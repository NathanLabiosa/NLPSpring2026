#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --output=/home1/labiosa/NLPSpring2026/models/tinyllama/logs/viability_%j.out
#SBATCH --error=/home1/labiosa/NLPSpring2026/models/tinyllama/logs/viability_%j.err

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

echo "--- TinyLlama-1.1B-Chat Viability Check (C1) ---"
nvidia-smi

cd /home1/labiosa/NLPSpring2026/TinyLlama
/home1/labiosa/miniconda3/envs/my_project/bin/python viability_check.py
