#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=8:00:00
#SBATCH --output=logs/eval_lrd_adapter_%j.out
#SBATCH --error=logs/eval_lrd_adapter_%j.err

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1

echo "--- LRD analysis of v17 adapter: does it reduce divergence at the causal window? ---"
nvidia-smi
mkdir -p logs

# Experiment 6: Per-perturbation-type LRD analysis of the v17 adapter.
# For each eval condition, measures per-layer cosine distance:
#   - clean input → base model                   (reference)
#   - perturbed   → base model (no adapter)      (baseline LRD)
#   - perturbed   → adapted model (with adapter) (adapted LRD)
#
# If reduction is concentrated at layers 0–4 (the LoRA layers / causal window),
# this closes the narrative loop: diagnostics → placement → measured reduction → gains.
# No new training. Uses existing v17/lora_final checkpoint.

python eval_lrd_adapter.py \
    --checkpoint_dir ./stabilizer_weights/lora_v17/lora_final \
    --n_samples 100 \
    --output    ./stabilizer_weights/lora_v17/lrd_adapter_analysis.json
