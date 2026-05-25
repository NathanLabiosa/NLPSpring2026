#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=01:00:00
#SBATCH --output=/home1/labiosa/NLPSpring2026/logs/three_map_new_%j.out
#SBATCH --error=/home1/labiosa/NLPSpring2026/logs/three_map_new_%j.err

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

echo "--- Three-Map Analysis + Block Bootstrap for New Models (A6/B5/C6) ---"

cd /home1/labiosa/NLPSpring2026
/home1/labiosa/miniconda3/envs/my_project/bin/python three_map_new.py

echo "--- Also checking predictions vs sweep results ---"
/home1/labiosa/miniconda3/envs/my_project/bin/python predict_optimal_window.py --model Gemma2 --check_prediction
/home1/labiosa/miniconda3/envs/my_project/bin/python predict_optimal_window.py --model Qwen25 --check_prediction
/home1/labiosa/miniconda3/envs/my_project/bin/python predict_optimal_window.py --model TinyLlama --check_prediction

echo "--- Analysis complete ---"
