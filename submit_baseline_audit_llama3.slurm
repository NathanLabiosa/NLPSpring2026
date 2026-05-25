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
#SBATCH --output=/home1/labiosa/NLPSpring2026/logs/audit_llama3_%j.out
#SBATCH --error=/home1/labiosa/NLPSpring2026/logs/audit_llama3_%j.err

# Phase 0 baseline audit: Llama-3-8B-Instruct, no LoRA, fixed harness.
# Published GSM8K (Meta): ~79% on 8-shot CoT. Compare clean baseline here.

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project_gemma
export LD_LIBRARY_PATH=/home1/labiosa/miniconda3/envs/my_project/lib:$LD_LIBRARY_PATH

# HF cache was migrated from ~/.cache/huggingface/hub to scratch1.
export HF_HOME=/scratch1/labiosa/huggingface_cache
export HF_HUB_CACHE=$HF_HOME/hub
[ -f $HF_HOME/token ] && export HF_TOKEN=$(cat $HF_HOME/token)

export PYTHONUNBUFFERED=1
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_ETAG_TIMEOUT=30

echo "=========================================="
echo "Baseline audit: Llama-3-8B-Instruct (no LoRA)"
echo "=========================================="
nvidia-smi

ROOT=/home1/labiosa/NLPSpring2026
PY=/home1/labiosa/miniconda3/envs/my_project_gemma/bin/python
RESULTS=$ROOT/results/fixed_harness/v1/baseline_audit
mkdir -p $RESULTS

$PY $ROOT/eval_fixed_harness.py \
    --base_model meta-llama/Meta-Llama-3-8B-Instruct \
    --n_samples 500 \
    --seed 42 \
    --output_file $RESULTS/baseline_llama3_8b_instruct.json

echo "--- done ---"
