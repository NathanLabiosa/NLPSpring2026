#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --array=1-2
#SBATCH --output=logs/paired_bootstrap_%a_%j.out
#SBATCH --error=logs/paired_bootstrap_%a_%j.err

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs

# Task 1: Headline config — cosine v2, stab=24, λ=3.0
# Task 2: MSE comparison — MSE, stab=24, λ=3.0
if [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then
    echo "--- Paired bootstrap: Cosine v2, L15-19, stab=24, λ=3.0 (headline) ---"
    python eval_paired_bootstrap.py \
        --checkpoint_dir ./stabilizer_weights/cosv2_L15_stab24_high/lora_final \
        --output_dir     ./bootstrap_results/cosv2_L15_stab24_high \
        --n_samples 500 --n_bootstrap 10000
elif [ "$SLURM_ARRAY_TASK_ID" -eq 2 ]; then
    echo "--- Paired bootstrap: MSE, L15-19, stab=24, λ=3.0 (comparison) ---"
    python eval_paired_bootstrap.py \
        --checkpoint_dir ./stabilizer_weights/mse_L15_stab24_high/lora_final \
        --output_dir     ./bootstrap_results/mse_L15_stab24_high \
        --n_samples 500 --n_bootstrap 10000
fi
