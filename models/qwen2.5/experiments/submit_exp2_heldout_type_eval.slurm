#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --time=02:00:00
#SBATCH --output=/home1/labiosa/NLPSpring2026/logs/qwen_heldout_type_eval_%j.out
#SBATCH --error=/home1/labiosa/NLPSpring2026/logs/qwen_heldout_type_eval_%j.err

# Experiment 2b: Held-out TYPE evaluation
# Evaluates retrained checkpoints (no speech in training) on speech perturbation
# Run this AFTER submit_exp2_heldout_type_train.slurm completes

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project_gemma

export LD_LIBRARY_PATH=/home1/labiosa/miniconda3/envs/my_project/lib:$LD_LIBRARY_PATH
export HF_TOKEN=$(cat /home1/labiosa/.cache/huggingface/token)

echo "=========================================="
echo "Experiment 2b: Held-out Type Evaluation"
echo "Testing on speech (held out from training)"
echo "=========================================="
nvidia-smi

ROOT=/home1/labiosa/NLPSpring2026
PY=/home1/labiosa/miniconda3/envs/my_project_gemma/bin/python
SW=$ROOT/stabilizer_weights
RESULTS=$ROOT/models/qwen2.5/experiments/results

mkdir -p $RESULTS

# Evaluate vanilla (trained without speech)
echo "--- Evaluating vanilla (no speech training) on speech ---"
$PY $ROOT/models/qwen2.5/experiments/exp_qwen_heldout_eval.py \
    --checkpoint $SW/qwen_vanilla_nospeech/lora_final \
    --mode type \
    --include_indist \
    --n_samples 500 \
    --seed 42 \
    --output_file $RESULTS/vanilla_nospeech_heldout_type.json

# Evaluate stabilizer (trained without speech)
echo "--- Evaluating stabilizer (no speech training) on speech ---"
$PY $ROOT/models/qwen2.5/experiments/exp_qwen_heldout_eval.py \
    --checkpoint $SW/qwen_stab_nospeech/lora_final \
    --mode type \
    --include_indist \
    --n_samples 500 \
    --seed 42 \
    --output_file $RESULTS/stabilizer_nospeech_heldout_type.json

echo "--- Held-out type evaluation complete ---"
echo "Results saved to: $RESULTS/"
