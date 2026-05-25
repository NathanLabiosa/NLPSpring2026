#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --exclude=a02-15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --time=08:00:00
#SBATCH --array=1-2
#SBATCH --output=/home1/labiosa/NLPSpring2026/logs/heldout_lrd_%a_%j.out
#SBATCH --error=/home1/labiosa/NLPSpring2026/logs/heldout_lrd_%a_%j.err

# Phase 2.4 — Held-out perturbation LRD profiles.
# Picks one model from each regime: Phi-3.5 (S&S) and Llama-3 (Late).
# Perturbations are isolated single-op variants (char_insert, char_swap, qwerty)
# that were not used in the main analysis or in any LoRA training.

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

# task 1 = Phi-3.5 (S&S regime), task 2 = Llama-3 (Late regime)
if [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then
    MODEL_ID="microsoft/Phi-3.5-mini-instruct"
    MODEL_SLUG="phi35"
    ATTN_IMPL="eager"
elif [ "$SLURM_ARRAY_TASK_ID" -eq 2 ]; then
    MODEL_ID="meta-llama/Meta-Llama-3-8B-Instruct"
    MODEL_SLUG="llama3_8b"
    ATTN_IMPL="eager"
else
    echo "ERROR: unexpected SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
    exit 1
fi

ROOT=/home1/labiosa/NLPSpring2026
PY=/home1/labiosa/miniconda3/envs/my_project_gemma/bin/python
RESULTS=$ROOT/results/heldout_perturbation_lrd
OUTPUT=$RESULTS/${MODEL_SLUG}.json
mkdir -p $RESULTS

echo "=========================================="
echo "Held-out LRD profiling  model=${MODEL_SLUG}"
echo "  attn_impl=${ATTN_IMPL}  output=${OUTPUT}"
echo "=========================================="
nvidia-smi

$PY $ROOT/src/eval/eval_heldout_lrd.py \
    --base_model     $MODEL_ID \
    --output_file    $OUTPUT \
    --n_samples      200 \
    --prefilter_pool 800 \
    --seed           42 \
    --attn_impl      $ATTN_IMPL \
    --max_new_tokens 256

echo "--- done: ${MODEL_SLUG} ---"
