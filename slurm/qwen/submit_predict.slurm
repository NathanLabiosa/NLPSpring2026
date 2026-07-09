#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --exclude=a02-15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --time=04:00:00
#SBATCH --output=/home1/labiosa/NLPSpring2026/models/qwen2.5/logs/predict_%j.out
#SBATCH --error=/home1/labiosa/NLPSpring2026/models/qwen2.5/logs/predict_%j.err

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project_gemma
export LD_LIBRARY_PATH=/home1/labiosa/miniconda3/envs/my_project/lib:$LD_LIBRARY_PATH

echo "--- Qwen2.5-7B C3/C4 Prediction (Step B3) ---"
echo "IMPORTANT: This must run AFTER LRD diagnostic and BEFORE sweep."
nvidia-smi

ROOT=/home1/labiosa/NLPSpring2026
PY=/home1/labiosa/miniconda3/envs/my_project_gemma/bin/python

$PY $ROOT/predict_optimal_window.py --model Qwen25 --compute_c3c4

echo "--- Prediction file written. Now safe to submit sweep. ---"
