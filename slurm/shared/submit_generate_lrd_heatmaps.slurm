#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=00:10:00
#SBATCH --output=/home1/labiosa/NLPSpring2026/logs/generate_lrd_heatmaps_%j.out
#SBATCH --error=/home1/labiosa/NLPSpring2026/logs/generate_lrd_heatmaps_%j.err

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project

echo "--- Generating LRD Heatmaps ---"
python3 /home1/labiosa/NLPSpring2026/generate_lrd_heatmaps.py

echo "--- Done ---"
ls -la /home1/labiosa/NLPSpring2026/figures/lrd_heatmap_*.pdf
