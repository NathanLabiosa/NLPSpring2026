#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --exclude=a02-15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --time=10:00:00
#SBATCH --array=1-18%6
#SBATCH --output=/home1/labiosa/NLPSpring2026/logs/phase1_qwen_%a_%j.out
#SBATCH --error=/home1/labiosa/NLPSpring2026/logs/phase1_qwen_%a_%j.err

# Phase 1 — Qwen2.5-7B-Instruct layer sweep under fixed harness.
# 6 windows × 3 seeds = 18 array tasks.
# Includes L08-11: falsifies the C3/C4 post-hoc prediction (per plan).

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project_gemma
export LD_LIBRARY_PATH=/home1/labiosa/miniconda3/envs/my_project/lib:$LD_LIBRARY_PATH

# HF cache was migrated from ~/.cache/huggingface/hub to scratch1.
export HF_HOME=/scratch1/labiosa/huggingface_cache
export HF_HUB_CACHE=$HF_HOME/hub
[ -f $HF_HOME/token ] && export HF_TOKEN=$(cat $HF_HOME/token)

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_DOWNLOAD_TIMEOUT=120

MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
MODEL_SLUG="qwen2.5_7b"
ATTN_IMPL="sdpa"
TARGET_MODULES=("q_proj" "v_proj")

SEEDS=(42 43 44)
# L00-04, L05-09, L08-11, L15-19, L20-23, L24-27 (Qwen-7B has 28 layers).
WINDOW_STARTS=(0  5  8 15 20 24)
WINDOW_ENDS=(  4  9 11 19 23 27)
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

$PY $ROOT/src/training/train_lrd_lora_v14.py \
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
    --batch_size         4 \
    --grad_accum_steps   4 \
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

$PY $ROOT/src/eval/eval_fixed_harness.py \
    --base_model     $MODEL_ID \
    --checkpoint     ${TRAIN_DIR}/lora_final \
    --n_samples      500 \
    --seed           42 \
    --attn_impl      $ATTN_IMPL \
    --output_file    $EVAL_OUT

echo "--- done: ${WIN_TAG} seed=${SEED} ---"
