#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --time=02:00:00
#SBATCH --output=/home1/labiosa/NLPSpring2026/logs/qwen_heldout_severity_%j.out
#SBATCH --error=/home1/labiosa/NLPSpring2026/logs/qwen_heldout_severity_%j.err

# Experiment 2a: Held-out SEVERITY evaluation
# Evaluates existing vanilla and stabilizer checkpoints on perturbation rates
# NOT seen in training:
#   - typos @ 10% and 15% (training had 5%)
#   - OCR @ 10% (training had 5% and 15%, this is interpolation)

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project_gemma

export LD_LIBRARY_PATH=/home1/labiosa/miniconda3/envs/my_project/lib:$LD_LIBRARY_PATH
export HF_TOKEN=$(cat /home1/labiosa/.cache/huggingface/token)

echo "=========================================="
echo "Experiment 2a: Held-out Severity Evaluation"
echo "=========================================="
nvidia-smi

ROOT=/home1/labiosa/NLPSpring2026
PY=/home1/labiosa/miniconda3/envs/my_project_gemma/bin/python
SW=$ROOT/stabilizer_weights
RESULTS=$ROOT/models/qwen2.5/experiments/results

mkdir -p $RESULTS

# Evaluate vanilla LoRA (seed=42)
echo "--- Evaluating vanilla LoRA on held-out severities ---"
$PY $ROOT/models/qwen2.5/experiments/exp_qwen_heldout_eval.py \
    --checkpoint $SW/qwen_sweep_L24_27/lora_final \
    --mode severity \
    --include_indist \
    --n_samples 500 \
    --seed 42 \
    --output_file $RESULTS/vanilla_heldout_severity.json

# Evaluate stabilizer (seed=42)
echo "--- Evaluating stabilizer on held-out severities ---"
$PY $ROOT/models/qwen2.5/experiments/exp_qwen_heldout_eval.py \
    --checkpoint $SW/qwen25_7b_L24_27_stab/lora_final \
    --mode severity \
    --include_indist \
    --n_samples 500 \
    --seed 42 \
    --output_file $RESULTS/stabilizer_heldout_severity.json

echo "--- Held-out severity evaluation complete ---"
echo "Results saved to: $RESULTS/"
