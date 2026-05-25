#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=6:00:00
#SBATCH --output=logs/expF_%j.out
#SBATCH --error=logs/expF_%j.err

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd /home1/labiosa/NLPSpring2026

echo "=== Exp F: LoRA-Induced Clean Disruption ==="
nvidia-smi

# Run all three models sequentially (each loads/unloads model weights)
python src/experiments/expF_clean_disruption.py --model phi35   --n_samples 200
python src/experiments/expF_clean_disruption.py --model llama3  --n_samples 200
python src/experiments/expF_clean_disruption.py --model mistral --n_samples 200

echo "=== DONE ==="
