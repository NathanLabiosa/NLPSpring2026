#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=1:00:00
#SBATCH --output=logs/workstream_ac_%j.out
#SBATCH --error=logs/workstream_ac_%j.err

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1

mkdir -p logs

echo "=== A1: Interaction Test ==="
python3 experiments/workstream_a/interaction_test.py

echo ""
echo "=== A2: Whitespace Analysis ==="
python3 experiments/workstream_a/whitespace_analysis.py

echo ""
echo "=== A3: Power Analysis ==="
python3 experiments/workstream_a/power_analysis.py

echo ""
echo "=== C: Generate Figures ==="
python3 experiments/workstream_c/generate_figures.py

echo ""
echo "=== ALL DONE ==="
