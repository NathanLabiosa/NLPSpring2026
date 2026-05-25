#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --output=logs/p7_compensatory_%j.out
#SBATCH --error=logs/p7_compensatory_%j.err

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "--- P7: Compensatory computation analysis ---"
nvidia-smi
mkdir -p logs

# Priority 7: Compensatory computation analysis
# Measures LRD with/without adapter at causal window layers (0–4),
# and trains linear probes on layer-4 hidden states to test whether
# the adapter changes what information is available.

python src/experiments/compensatory_analysis.py \
    --checkpoint_dir ./stabilizer_weights/lora_v17/lora_final \
    --model microsoft/Phi-3.5-mini-instruct \
    --n_samples 200 \
    --causal_layers 0 1 2 3 4 \
    --probe_layer 5 \
    --output ./compensatory_results
