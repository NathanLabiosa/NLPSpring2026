#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=1:00:00
#SBATCH --output=logs/p3_bootstrap_%j.out
#SBATCH --error=logs/p3_bootstrap_%j.err

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project

export PYTHONUNBUFFERED=1

echo "--- P3: Bootstrap CIs on LRD diagnostics ---"
mkdir -p logs

# Priority 3: Bootstrap diagnostics — no GPU needed, just resampling.
# Runs bootstrap CIs on point-biserial r, recovery rates, cascade slope
# for all models that have existing LRD results.

# Phi-3.5 (hardened results with per-token LRD)
python src/analysis/bootstrap_diagnostics.py \
    --lrd_results_dir ./models/phi3.5/lrd_results/phi_hardening2 \
    --patching_dir ./models/phi3.5/lrd_results/phi_hardening2 \
    --dataset gsm8k \
    --n_bootstrap 10000 \
    --output ./results/bootstrap/phi35

# Llama-3
python src/analysis/bootstrap_diagnostics.py \
    --lrd_results_dir ./models/llama/lrd_results \
    --dataset gsm8k \
    --n_bootstrap 10000 \
    --output ./results/bootstrap/llama

# Qwen
python src/analysis/bootstrap_diagnostics.py \
    --lrd_results_dir ./models/qwen2.5/lrd_results \
    --dataset gsm8k \
    --n_bootstrap 10000 \
    --output ./results/bootstrap/qwen

echo "--- Bootstrap analysis complete ---"
