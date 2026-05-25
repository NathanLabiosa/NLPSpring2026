#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --output=/home1/labiosa/NLPSpring2026/TinyLlama/logs/predict_%j.out
#SBATCH --error=/home1/labiosa/NLPSpring2026/TinyLlama/logs/predict_%j.err

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

echo "--- TinyLlama C3/C4 Prediction (Step C4) ---"
echo "IMPORTANT: This must run AFTER LRD diagnostic and BEFORE sweep."
nvidia-smi

ROOT=/home1/labiosa/NLPSpring2026
PY=/home1/labiosa/miniconda3/envs/my_project/bin/python

$PY $ROOT/predict_optimal_window.py --model TinyLlama --compute_c3c4

echo "--- Prediction file written. Now safe to submit sweep. ---"
