#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --output=logs/eval_transfer_%j.out
#SBATCH --error=logs/eval_transfer_%j.err

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1

echo "--- MMLU + BBH TRANSFER EVAL (v17) ---"
nvidia-smi
mkdir -p logs

# n_samples=200 per benchmark, 20 examples per MMLU subject / BBH task
python src/eval/eval_transfer.py \
    --checkpoint_dir ./stabilizer_weights/lora_v17/lora_final \
    --n_samples    200 \
    --n_per_subject 20 \
    --n_per_task    20 \
    --output ./stabilizer_weights/lora_v17/transfer_results.json
