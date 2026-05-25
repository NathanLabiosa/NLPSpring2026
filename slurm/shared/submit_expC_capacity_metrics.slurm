#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=8:00:00
#SBATCH --output=logs/expC_%j.out
#SBATCH --error=logs/expC_%j.err

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd /home1/labiosa/NLPSpring2026

echo "=== Exp C: Layer-Wise Capacity Metrics ==="
nvidia-smi

# Phi3.5 (includes C4 gradient norms, ~2–3 hrs on V100)
python src/experiments/expC_capacity_metrics.py --model phi35   --n_samples 200

# Llama3
python src/experiments/expC_capacity_metrics.py --model llama3  --n_samples 200

# Mistral
python src/experiments/expC_capacity_metrics.py --model mistral --n_samples 200

echo "=== DONE ==="
