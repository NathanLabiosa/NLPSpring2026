#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --time=04:00:00
#SBATCH --output=/home1/labiosa/NLPSpring2026/Gemma2/logs/predict_%j.out
#SBATCH --error=/home1/labiosa/NLPSpring2026/Gemma2/logs/predict_%j.err

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project_gemma
export LD_LIBRARY_PATH=/home1/labiosa/miniconda3/envs/my_project/lib:$LD_LIBRARY_PATH
export HF_TOKEN=$(cat /home1/labiosa/.cache/huggingface/token)

echo "--- Gemma-2-9B C3/C4 Prediction (Step A4) ---"
echo "IMPORTANT: This must run AFTER LRD diagnostic and BEFORE sweep."
nvidia-smi

ROOT=/home1/labiosa/NLPSpring2026
PY=/home1/labiosa/miniconda3/envs/my_project_gemma/bin/python

$PY $ROOT/predict_optimal_window.py --model Gemma2 --compute_c3c4

echo "--- Prediction file written. Now safe to submit sweep. ---"
