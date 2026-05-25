#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80GB
#SBATCH --time=48:00:00
#SBATCH --output=/home1/labiosa/NLPSpring2026/logs/qwen14b_lrd_%j.out
#SBATCH --error=/home1/labiosa/NLPSpring2026/logs/qwen14b_lrd_%j.err

# Qwen2.5-14B LRD Profiling
# Tests whether late-accumulation regime persists at larger Qwen scale

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project_gemma

export LD_LIBRARY_PATH=/home1/labiosa/miniconda3/envs/my_project/lib:$LD_LIBRARY_PATH
[ -f /home1/labiosa/.cache/huggingface/token ] && export HF_TOKEN=$(cat /home1/labiosa/.cache/huggingface/token)
export PYTHONUNBUFFERED=1
export HF_HUB_DOWNLOAD_TIMEOUT=30
export HF_HUB_ETAG_TIMEOUT=10

echo "=========================================="
echo "Qwen2.5-14B LRD Profiling"
echo "=========================================="
nvidia-smi

ROOT=/home1/labiosa/NLPSpring2026
export PYTHONPATH=$ROOT/scripts:$PYTHONPATH
PY=/home1/labiosa/miniconda3/envs/my_project_gemma/bin/python
cd $ROOT/Qwen2.5

$PY lrd_diagnostics.py \
    --dataset gsm8k \
    --model Qwen/Qwen2.5-14B-Instruct \
    --n_samples 500 \
    --output_dir $ROOT/Qwen2.5/scale_experiments/lrd_results_14B \
    --max_new_tokens 768 \
    --run_taxonomy_variance

echo "--- Qwen2.5-14B LRD complete ---"
