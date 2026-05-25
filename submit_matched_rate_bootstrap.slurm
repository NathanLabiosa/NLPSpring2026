#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=8:00:00
#SBATCH --array=1-2
#SBATCH --output=logs/matched_rate_bootstrap_%a_%j.out
#SBATCH --error=logs/matched_rate_bootstrap_%a_%j.err

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs

# Paired bootstrap CIs for matched-rate (10%) experiment.
# Evaluates at 10% for all perturbation types to match training.

if [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then
    echo "--- Bootstrap: Cosine matched-rate ---"
    python eval_paired_bootstrap.py \
        --model microsoft/Phi-3.5-mini-instruct \
        --checkpoint_dir ./stabilizer_weights/phi_matched_cosine_stab24/lora_final \
        --output_dir ./bootstrap_results/phi_matched_cosine_stab24 \
        --n_samples 500 --n_bootstrap 10000 \
        --uniform_rate 0.10

elif [ "$SLURM_ARRAY_TASK_ID" -eq 2 ]; then
    echo "--- Bootstrap: MSE matched-rate ---"
    python eval_paired_bootstrap.py \
        --model microsoft/Phi-3.5-mini-instruct \
        --checkpoint_dir ./stabilizer_weights/phi_matched_mse_stab24/lora_final \
        --output_dir ./bootstrap_results/phi_matched_mse_stab24 \
        --n_samples 500 --n_bootstrap 10000 \
        --uniform_rate 0.10
fi
