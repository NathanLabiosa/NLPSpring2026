#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --exclude=a02-15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --output=logs/qwen_speech_patch_%j.out
#SBATCH --error=logs/qwen_speech_patch_%j.err

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1

echo "--- Qwen2.5-7B activation patching: speech 10% (Experiment 4) ---"
nvidia-smi
mkdir -p logs

# Experiment 4: Qwen patching with speech perturbation (replaces OCR).
# OCR patching gave 60% identity ceiling due to BPE misalignment.
# Speech perturbation preserves tokenization better → expect >80% identity.
# If confirmed, upgrades the Claim 4 dissociability finding from exploratory
# to confirmed and identifies the broader Qwen causal window (expected peak ~layer 12).

cd /home1/labiosa/NLPSpring2026/Qwen2.5
mkdir -p lrd_results_speech logs

python patch_only_speech.py
