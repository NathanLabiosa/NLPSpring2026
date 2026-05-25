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
#SBATCH --output=/home1/labiosa/NLPSpring2026/logs/regression_%j.out
#SBATCH --error=/home1/labiosa/NLPSpring2026/logs/regression_%j.err

# Harness regression test — re-runs Qwen stab and asserts mean perturbed Δ
# is within ±1pp of the recorded reference. Run before every batch of jobs.

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project_gemma
export LD_LIBRARY_PATH=/home1/labiosa/miniconda3/envs/my_project/lib:$LD_LIBRARY_PATH

# HF cache was migrated from ~/.cache/huggingface/hub to scratch1.
export HF_HOME=/scratch1/labiosa/huggingface_cache
export HF_HUB_CACHE=$HF_HOME/hub
[ -f $HF_HOME/token ] && export HF_TOKEN=$(cat $HF_HOME/token)

export PYTHONUNBUFFERED=1

echo "=========================================="
echo "Harness regression test (Qwen stab checkpoint)"
echo "=========================================="
nvidia-smi

cd /home1/labiosa/NLPSpring2026
/home1/labiosa/miniconda3/envs/my_project_gemma/bin/python regression_test_harness.py

echo "--- done ---"
