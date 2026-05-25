#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --output=logs/aug_baseline_%j.out
#SBATCH --error=logs/aug_baseline_%j.err

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

echo "--- Augmentation Baseline (Exp 3) ---"
nvidia-smi

cd /home1/labiosa/NLPSpring2026

# Run both variants sequentially: L15-19 and all-32-layers
/home1/labiosa/miniconda3/envs/my_project/bin/python src/training/train_augmentation_baseline.py \
    --variant both \
    --lora_rank 4 \
    --lora_alpha 8 \
    --n_steps 2000 \
    --n_per_condition 300 \
    --n_eval 500

echo "--- Augmentation baseline complete ---"
