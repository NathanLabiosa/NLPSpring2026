#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --exclude=a02-15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --time=24:00:00
#SBATCH --output=/home1/labiosa/NLPSpring2026/logs/llama32_1b_lrd_%j.out
#SBATCH --error=/home1/labiosa/NLPSpring2026/logs/llama32_1b_lrd_%j.err

# Llama-3.2-1B LRD Profiling
# Tests within-family scale variation on a second architecture family

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

echo "=========================================="
echo "Llama-3.2-1B LRD Profiling"
echo "=========================================="
nvidia-smi

ROOT=/home1/labiosa/NLPSpring2026
export PYTHONPATH=$ROOT/scripts:$PYTHONPATH
PY=/home1/labiosa/miniconda3/envs/my_project_gemma/bin/python
cd $ROOT/Llama

$PY lrd_diagnostics.py \
    --dataset gsm8k \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --n_samples 500 \
    --output_dir $ROOT/models/llama/lrd_results_3.2_1B \
    --max_new_tokens 768 \
    --run_taxonomy_variance

echo "--- Llama-3.2-1B LRD complete ---"
