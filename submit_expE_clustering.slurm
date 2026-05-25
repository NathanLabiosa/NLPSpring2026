#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=20:00:00
#SBATCH --output=logs/expE_%j.out
#SBATCH --error=logs/expE_%j.err

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd /home1/labiosa/NLPSpring2026

echo "=== Exp E: Per-Example LRD Trajectory Clustering ==="
nvidia-smi

# n=500 examples × 2 perturbations + pre-filtering pass
# ~15-18 hrs on V100 (individual forward passes — no batching for LRD)
python expE_clustering.py --n_samples 500 --k_min 3 --k_max 5 --seed 42

echo "=== DONE ==="
