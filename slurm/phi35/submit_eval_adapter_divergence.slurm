#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=8:00:00
#SBATCH --output=logs/eval_adapter_divergence_%j.out
#SBATCH --error=logs/eval_adapter_divergence_%j.err

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "--- Clean-input adapter divergence analysis ---"
nvidia-smi
mkdir -p logs

# Measures how much each adapter rewrites the base model on CLEAN inputs.
# Loads each adapter one at a time (to avoid OOM), runs 200 clean GSM8K
# prompts through both adapter-enabled and adapter-disabled, computes
# per-layer cosine distance.
#
# If random-layer adapters show high divergence and v17 doesn't, that's
# evidence L_stab preserves clean behavior and the random runs are
# "cheating" by rewriting the model more aggressively.

python src/eval/eval_adapter_divergence.py \
    --model microsoft/Phi-3.5-mini-instruct \
    --adapters \
        v17=./stabilizer_weights/lora_v17/lora_final \
        clean_only=./stabilizer_weights/p1_clean_only/lora_final \
        rand_s1=./stabilizer_weights/p1_random_layers_seed1/lora_final \
        rand_s2=./stabilizer_weights/p1_random_layers_seed2/lora_final \
        rand_s3=./stabilizer_weights/p1_random_layers_seed3/lora_final \
        alllayer_r4=./stabilizer_weights/phi35_naive_alllayers/lora_final \
        alllayer_r8=./stabilizer_weights/p1_data_aug_r8/lora_final \
    --n_samples 200 \
    --output ./adapter_divergence_results.json
