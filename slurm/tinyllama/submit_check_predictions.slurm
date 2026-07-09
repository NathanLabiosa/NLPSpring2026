#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=1:00:00
#SBATCH --output=logs/check_predict_%j.out
#SBATCH --error=logs/check_predict_%j.err

# Run AFTER all sweep jobs complete to check predictions.
# No GPU needed (just reads JSON) but submitted to same partition for simplicity.

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd /home1/labiosa/NLPSpring2026
PY=/home1/labiosa/miniconda3/envs/my_project/bin/python

echo "=== Checking C3/C4 Predictions ==="

$PY src/analysis/predict_optimal_window.py --model TinyLlama --check_prediction
$PY src/analysis/predict_optimal_window.py --model Gemma2    --check_prediction
$PY src/analysis/predict_optimal_window.py --model Qwen25    --check_prediction

echo "--- Check complete. See results/predictions/*.json for results. ---"
