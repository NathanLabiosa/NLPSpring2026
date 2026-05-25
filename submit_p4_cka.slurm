#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --output=logs/p4_cka_%j.out
#SBATCH --error=logs/p4_cka_%j.err

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "--- P4: CKA comparison + per-position LRD ---"
nvidia-smi
mkdir -p logs

# Priority 4: CKA validation of LRD metric
# Computes linear CKA and per-position LRD for Phi-3.5 on GSM8K.
# CKA validates LRD as a simpler proxy; per-position LRD resolves
# the homophones paradox from the main text.

python compute_cka.py \
    --model microsoft/Phi-3.5-mini-instruct \
    --dataset gsm8k \
    --n_samples 100 \
    --output ./cka_results/phi35
