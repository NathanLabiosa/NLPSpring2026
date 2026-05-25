#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --exclude=a02-15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80GB
#SBATCH --time=30:00:00
#SBATCH --array=1-15%4
#SBATCH --output=/home1/labiosa/NLPSpring2026/logs/phase1_gemma2_%a_%j.out
#SBATCH --error=/home1/labiosa/NLPSpring2026/logs/phase1_gemma2_%a_%j.err

# Phase 1 — Gemma-2-9B (base) layer sweep under fixed harness.
# 5 windows × 3 seeds = 15 array tasks.
# Gemma-2 needs eager attention. Windows widen to 6 layers (Gemma is 42 layers).

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project_gemma
export LD_LIBRARY_PATH=/home1/labiosa/miniconda3/envs/my_project/lib:$LD_LIBRARY_PATH

# HF cache was migrated from ~/.cache/huggingface/hub to scratch1.
export HF_HOME=/scratch1/labiosa/huggingface_cache
export HF_HUB_CACHE=$HF_HOME/hub
# Gemma is gated under a separate HF account; prefer the Gemma-specific token.
if [ -f $HF_HOME/token_gemma ]; then
    export HF_TOKEN=$(cat $HF_HOME/token_gemma)
elif [ -f $HF_HOME/token ]; then
    export HF_TOKEN=$(cat $HF_HOME/token)
fi

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_DOWNLOAD_TIMEOUT=120

MODEL_ID="google/gemma-2-9b"
MODEL_SLUG="gemma2_9b"
ATTN_IMPL="eager"
TARGET_MODULES=("q_proj" "v_proj")

SEEDS=(42 43 44)
# Plan windows: L00-04 (5 layers), L06-11 (6), L15-20 (6), L25-30 (6), L35-40 (6).
WINDOW_STARTS=( 0  6 15 25 35)
WINDOW_ENDS=(   4 11 20 30 40)
N_WINDOWS=${#WINDOW_STARTS[@]}

SEED_IDX=$(( (SLURM_ARRAY_TASK_ID - 1) / N_WINDOWS ))
WIN_IDX=$(( (SLURM_ARRAY_TASK_ID - 1) % N_WINDOWS ))
SEED=${SEEDS[$SEED_IDX]}
WIN_START=${WINDOW_STARTS[$WIN_IDX]}
WIN_END=${WINDOW_ENDS[$WIN_IDX]}
WIDTH=$(( WIN_END - WIN_START + 1 ))
WIN_TAG=$(printf "L%02d-%02d" $WIN_START $WIN_END)

ROOT=/home1/labiosa/NLPSpring2026
PY=/home1/labiosa/miniconda3/envs/my_project_gemma/bin/python
RUN_TAG="phase1_${MODEL_SLUG}_${WIN_TAG}_seed${SEED}"
TRAIN_DIR=$ROOT/stabilizer_weights/${RUN_TAG}
EVAL_OUT=$ROOT/results/fixed_harness/v1/layer_sweep/${MODEL_SLUG}_${WIN_TAG}_seed${SEED}.json
mkdir -p $(dirname $EVAL_OUT)

echo "=========================================="
echo "Phase 1 ${MODEL_SLUG}  task ${SLURM_ARRAY_TASK_ID}/$(( 3 * N_WINDOWS ))"
echo "  window=${WIN_TAG}  seed=${SEED}"
echo "=========================================="
nvidia-smi

$PY $ROOT/train_lrd_lora_v14.py \
    --model              $MODEL_ID \
    --dataset            gsm8k \
    --lora_rank          4 \
    --lora_alpha         8 \
    --lora_dropout       0.05 \
    --lora_layers        $WIDTH \
    --lora_layers_start  $WIN_START \
    --target_modules     "${TARGET_MODULES[@]}" \
    --lambda_acc         1.0 \
    --lambda_stab        0.0 \
    --stab_layer         $WIN_END \
    --n_per_condition    857 \
    --clean_fraction     0.25 \
    --epochs             30 \
    --max_steps          300 \
    --batch_size         2 \
    --grad_accum_steps   8 \
    --lr                 5e-5 \
    --warmup_ratio       0.05 \
    --max_seq_len        512 \
    --log_every          50 \
    --save_every         300 \
    --eval_every         0 \
    --clean_eval_every   300 \
    --clean_eval_n       200 \
    --early_stop_delta   5.0 \
    --no_eval_after_training \
    --seed               $SEED \
    --output_dir         $TRAIN_DIR

if [ ! -d "${TRAIN_DIR}/lora_final" ]; then
    echo "ERROR: training did not produce ${TRAIN_DIR}/lora_final — aborting eval"
    exit 1
fi

$PY $ROOT/eval_fixed_harness.py \
    --base_model     $MODEL_ID \
    --checkpoint     ${TRAIN_DIR}/lora_final \
    --n_samples      500 \
    --seed           42 \
    --attn_impl      $ATTN_IMPL \
    --output_file    $EVAL_OUT

echo "--- done: ${WIN_TAG} seed=${SEED} ---"
