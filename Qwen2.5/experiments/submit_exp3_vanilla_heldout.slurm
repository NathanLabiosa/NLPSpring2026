#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --time=08:00:00
#SBATCH --output=/home1/labiosa/NLPSpring2026/logs/qwen_exp3_vanilla_heldout_%j.out
#SBATCH --error=/home1/labiosa/NLPSpring2026/logs/qwen_exp3_vanilla_heldout_%j.err

# Experiment 3: Held-out evaluation on vanilla Qwen checkpoint
# Tests whether the +11.6% in-distribution delta transfers to:
# (a) held-out perturbation type (speech)
# (b) held-out severity (typos at 10%, 15%; OCR at 10%)

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project_gemma
export LD_LIBRARY_PATH=/home1/labiosa/miniconda3/envs/my_project/lib:$LD_LIBRARY_PATH
[ -f /home1/labiosa/.cache/huggingface/token ] && export HF_TOKEN=$(cat /home1/labiosa/.cache/huggingface/token)
export PYTHONUNBUFFERED=1
export HF_HUB_DOWNLOAD_TIMEOUT=30
export HF_HUB_ETAG_TIMEOUT=10

echo "=========================================="
echo "Experiment 3: Vanilla Held-out Evaluation"
echo "=========================================="
nvidia-smi

ROOT=/home1/labiosa/NLPSpring2026
PY=/home1/labiosa/miniconda3/envs/my_project_gemma/bin/python
SW=$ROOT/stabilizer_weights
RESULTS=$ROOT/Qwen2.5/experiments/results

mkdir -p $RESULTS

# Use the original vanilla checkpoint (seed 42)
VANILLA_CKPT=$SW/qwen_sweep_L24_27/lora_final

# Evaluate on held-out severity
echo "--- Evaluating vanilla on held-out severities ---"
$PY $ROOT/Qwen2.5/experiments/exp_qwen_heldout_eval.py \
    --checkpoint $VANILLA_CKPT \
    --mode severity \
    --include_indist \
    --n_samples 500 \
    --seed 42 \
    --output_file $RESULTS/exp3_vanilla_heldout_severity.json

# Evaluate on held-out type (speech)
echo "--- Evaluating vanilla on held-out type (speech) ---"
$PY $ROOT/Qwen2.5/experiments/exp_qwen_heldout_eval.py \
    --checkpoint $VANILLA_CKPT \
    --mode type \
    --include_indist \
    --n_samples 500 \
    --seed 42 \
    --output_file $RESULTS/exp3_vanilla_heldout_type.json

echo "--- Experiment 3 complete ---"
echo "Results saved to: $RESULTS/"
