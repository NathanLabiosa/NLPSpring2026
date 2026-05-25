#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=8:00:00
#SBATCH --output=logs/predict_%j.out
#SBATCH --error=logs/predict_%j.err

# Experiment 4: Predictive Validation of C3/C4
# Run BEFORE submitting sweep jobs for new models.
# Loads each new model, computes C3 (weight effective rank) and C4 (gradient norm),
# classifies regime from LRD diagnostic, and writes timestamped prediction to
# predictions/{model}_{timestamp}.json BEFORE seeing sweep results.
#
# Submit order: submit_predict_new_models.slurm → (wait for LRD + patch jobs) →
#               then submit_sweep.slurm for each model.
# After sweep: re-run with --check_prediction to validate.

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project_gemma
export LD_LIBRARY_PATH=/home1/labiosa/miniconda3/envs/my_project/lib:$LD_LIBRARY_PATH
export HF_TOKEN=$(cat /home1/labiosa/.cache/huggingface/token)

echo "--- Exp 4: C3/C4 Prediction ---"
nvidia-smi

cd /home1/labiosa/NLPSpring2026
PY=/home1/labiosa/miniconda3/envs/my_project_gemma/bin/python
mkdir -p predictions

echo "=== TinyLlama ==="
$PY predict_optimal_window.py --model TinyLlama --compute_c3c4 --n_samples 50

echo "=== Gemma2 ==="
$PY predict_optimal_window.py --model Gemma2 --compute_c3c4 --n_samples 50

echo "=== Qwen25 ==="
$PY predict_optimal_window.py --model Qwen25 --compute_c3c4 --n_samples 50

echo "--- Prediction phase complete. Now submit sweep jobs. ---"
