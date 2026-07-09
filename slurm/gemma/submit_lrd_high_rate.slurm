#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --time=24:00:00
#SBATCH --output=/home1/labiosa/NLPSpring2026/models/gemma2/logs/lrd_high_rate_%j.out
#SBATCH --error=/home1/labiosa/NLPSpring2026/models/gemma2/logs/lrd_high_rate_%j.err

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project_gemma
export LD_LIBRARY_PATH=/home1/labiosa/miniconda3/envs/my_project/lib:$LD_LIBRARY_PATH
export HF_TOKEN=$(cat /home1/labiosa/.cache/huggingface/token)

echo "--- Gemma-2-9B base LRD Diagnostic (HIGH PERTURBATION RATES) ---"
echo "Rates: Typos 10%, OCR 10%, Whitespace 20%, Case 20%, Homophones 40%, Speech 20%"
nvidia-smi

ROOT=/home1/labiosa/NLPSpring2026
export PYTHONPATH=$ROOT/scripts:$PYTHONPATH
cd /home1/labiosa/NLPSpring2026/Gemma2
/home1/labiosa/miniconda3/envs/my_project_gemma/bin/python lrd_run_high_rate.py

echo "--- High-rate LRD diagnostic complete ---"
