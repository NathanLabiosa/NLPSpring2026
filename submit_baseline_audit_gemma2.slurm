#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --exclude=a02-15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80GB
#SBATCH --time=10:00:00
#SBATCH --output=/home1/labiosa/NLPSpring2026/logs/audit_gemma2_%j.out
#SBATCH --error=/home1/labiosa/NLPSpring2026/logs/audit_gemma2_%j.err

# Phase 0 baseline audit: Gemma-2-9B (base), no LoRA, fixed harness.
# Published GSM8K (Google): ~68% (8-shot CoT) for the base; instruct higher.
# Note: this is the base model used in the panel; if the panel uses the
# instruct variant, swap the model_id.

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project_gemma
export LD_LIBRARY_PATH=/home1/labiosa/miniconda3/envs/my_project/lib:$LD_LIBRARY_PATH

# HF cache was migrated from ~/.cache/huggingface/hub to scratch1. Point
# HF_HOME at the new root so cached weights (already present) are reused
# instead of re-downloaded into the now-missing home cache.
export HF_HOME=/scratch1/labiosa/huggingface_cache
export HF_HUB_CACHE=$HF_HOME/hub

# Gemma is gated under a different HF account than Llama/Mistral; use the
# Gemma-specific token if it exists, otherwise fall back to the main token.
# Token files live under the migrated HF_HOME.
if [ -f $HF_HOME/token_gemma ]; then
    export HF_TOKEN=$(cat $HF_HOME/token_gemma)
elif [ -f $HF_HOME/token ]; then
    export HF_TOKEN=$(cat $HF_HOME/token)
elif [ -f /home1/labiosa/.cache/huggingface/token_gemma ]; then
    export HF_TOKEN=$(cat /home1/labiosa/.cache/huggingface/token_gemma)
fi
export PYTHONUNBUFFERED=1
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_ETAG_TIMEOUT=30

echo "=========================================="
echo "Baseline audit: Gemma-2-9B (no LoRA)"
echo "=========================================="
nvidia-smi

ROOT=/home1/labiosa/NLPSpring2026
PY=/home1/labiosa/miniconda3/envs/my_project_gemma/bin/python
RESULTS=$ROOT/results/fixed_harness/v1/baseline_audit
mkdir -p $RESULTS

# Gemma2 needs eager attention (Phi-3.5-style restriction).
$PY $ROOT/eval_fixed_harness.py \
    --base_model google/gemma-2-9b \
    --n_samples 500 \
    --seed 42 \
    --attn_impl eager \
    --output_file $RESULTS/baseline_gemma2_9b.json

echo "--- done ---"
