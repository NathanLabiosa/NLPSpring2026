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
#SBATCH --array=1-2
#SBATCH --output=/home1/labiosa/NLPSpring2026/logs/mistral_config_%a_%j.out
#SBATCH --error=/home1/labiosa/NLPSpring2026/logs/mistral_config_%a_%j.err

# Mistral config check — decides whether Mistral row in §6 is a real diagnostic
# finding or a config artifact.
#
# Phase 1 Mistral used identical LoRA config to Qwen (rank 4, alpha 8, q+v, lr 5e-5)
# but landed catastrophically negative at every window. Best window was L27-31
# at mean perturbed Δ = -4.3pp (seed-42 alone: clean Δ=-3.8, mean perturbed Δ=-4.5).
#
# Two alternate configs, both at L27-31 seed 42 (Mistral's best Phase 1 cell):
#   task 1: lr 5e-5 → 1e-5 (5× lower — most common "model overshoots" fix)
#   task 2: target_modules q+v → q+k+v+o (full attention coverage)
#
# Decision rule:
#   - If either config flips sign OR pulls mean Δ above -1pp → Mistral is config-fragile,
#     not diagnostic-fragile. Rerun full Mistral Phase 1 with the working config.
#   - If both still ≤ -3pp → Mistral is genuinely an outlier; report as-is in paper.

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

MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3"
MODEL_SLUG="mistral_7b_v03"
ATTN_IMPL="sdpa"
WIN_START=27
WIN_END=31
WIDTH=5
WIN_TAG=$(printf "L%02d-%02d" $WIN_START $WIN_END)
SEED=42

# Branch on task id: 1 = low-lr; 2 = wide-targets
if [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then
    VARIANT_TAG="lr1e-5"
    LR="1e-5"
    TARGET_MODULES=("q_proj" "v_proj")
elif [ "$SLURM_ARRAY_TASK_ID" -eq 2 ]; then
    VARIANT_TAG="qkvo"
    LR="5e-5"
    TARGET_MODULES=("q_proj" "k_proj" "v_proj" "o_proj")
else
    echo "ERROR: unexpected SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
    exit 1
fi

ROOT=/home1/labiosa/NLPSpring2026
PY=/home1/labiosa/miniconda3/envs/my_project_gemma/bin/python
RUN_TAG="mistral_cfgcheck_${VARIANT_TAG}_${WIN_TAG}_seed${SEED}"
TRAIN_DIR=$ROOT/stabilizer_weights/${RUN_TAG}
RESULTS=$ROOT/results/fixed_harness/v1/mistral_config_check
EVAL_OUT=$RESULTS/${VARIANT_TAG}_${WIN_TAG}_seed${SEED}.json
mkdir -p $RESULTS

echo "=========================================="
echo "Mistral config check  variant=${VARIANT_TAG}"
echo "  lr=${LR}  target_modules=${TARGET_MODULES[@]}"
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
    --batch_size         4 \
    --grad_accum_steps   4 \
    --lr                 $LR \
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

echo "--- done: variant=${VARIANT_TAG} ${WIN_TAG} seed=${SEED} ---"
