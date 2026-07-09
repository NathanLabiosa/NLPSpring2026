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
#SBATCH --output=/home1/labiosa/NLPSpring2026/models/qwen2.5/logs/lrd_%j.out
#SBATCH --error=/home1/labiosa/NLPSpring2026/models/qwen2.5/logs/lrd_%j.err

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project_gemma
export LD_LIBRARY_PATH=/home1/labiosa/miniconda3/envs/my_project/lib:$LD_LIBRARY_PATH

echo "--- Qwen2.5-7B LRD Diagnostic ---"
nvidia-smi

cd /home1/labiosa/NLPSpring2026/Qwen2.5

# Qwen2.5 lrd_diagnostics.py is already present in this directory
/home1/labiosa/miniconda3/envs/my_project_gemma/bin/python lrd_diagnostics.py \
    --dataset gsm8k \
    --model Qwen/Qwen2.5-7B-Instruct \
    --n_samples 500 \
    --output_dir lrd_results/qwen_gsm8k \
    --max_new_tokens 768 \
    --run_taxonomy_variance
