#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=10:00:00
#SBATCH --output=logs/train_stabilizer_%j.out
#SBATCH --error=logs/train_stabilizer_%j.err

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

echo "--- GATE ABLATION ---"
nvidia-smi

mkdir -p logs

python src/experiments/ablate_gate.py
