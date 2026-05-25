#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=6:00:00
#SBATCH --output=logs/eval_bootstrap_%j.out
#SBATCH --error=logs/eval_bootstrap_%j.err

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1

echo "--- BOOTSTRAP CONFIDENCE INTERVAL EVAL (v17) ---"
nvidia-smi
mkdir -p logs

python src/eval/eval_bootstrap.py \
    --checkpoint_dir ./stabilizer_weights/lora_v17/lora_final \
    --n_samples   500 \
    --n_bootstrap 10000 \
    --output      ./stabilizer_weights/lora_v17/bootstrap_results.json
