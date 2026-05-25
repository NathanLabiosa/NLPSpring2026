#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --exclude=a02-15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --time=02:00:00
#SBATCH --output=/home1/labiosa/NLPSpring2026/Qwen2.5/logs/bugfix_%j.out
#SBATCH --error=/home1/labiosa/NLPSpring2026/Qwen2.5/logs/bugfix_%j.err

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project_gemma
export LD_LIBRARY_PATH=/home1/labiosa/miniconda3/envs/my_project/lib:$LD_LIBRARY_PATH

echo "--- Qwen2.5-7B Bug Fix Investigation (Priority 0) ---"
nvidia-smi

cd /home1/labiosa/NLPSpring2026/Qwen2.5
/home1/labiosa/miniconda3/envs/my_project_gemma/bin/python qwen_bugfix.py
