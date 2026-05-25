#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --time=04:00:00
#SBATCH --output=/home1/labiosa/NLPSpring2026/Gemma2/logs/eval_only_v100_%j.out
#SBATCH --error=/home1/labiosa/NLPSpring2026/Gemma2/logs/eval_only_v100_%j.err

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project_gemma
export LD_LIBRARY_PATH=/home1/labiosa/miniconda3/envs/my_project/lib:$LD_LIBRARY_PATH
export HF_TOKEN=$(cat /home1/labiosa/.cache/huggingface/token)

echo "--- Gemma-2-9B Eval-Only (all 7 windows, V100) ---"
nvidia-smi

ROOT=/home1/labiosa/NLPSpring2026
PY=/home1/labiosa/miniconda3/envs/my_project_gemma/bin/python
MODEL=google/gemma-2-9b
SW=$ROOT/stabilizer_weights

# Evaluate all 7 saved adapters
$PY $ROOT/eval_only_lora_window.py \
    --model $MODEL --output_dir $SW/gemma2_9b_sweep_L00_05 \
    --eval_batch_size 8 --n_eval 500

$PY $ROOT/eval_only_lora_window.py \
    --model $MODEL --output_dir $SW/gemma2_9b_sweep_L06_11 \
    --eval_batch_size 8 --n_eval 500

$PY $ROOT/eval_only_lora_window.py \
    --model $MODEL --output_dir $SW/gemma2_9b_sweep_L12_17 \
    --eval_batch_size 8 --n_eval 500

$PY $ROOT/eval_only_lora_window.py \
    --model $MODEL --output_dir $SW/gemma2_9b_sweep_L18_23 \
    --eval_batch_size 8 --n_eval 500

$PY $ROOT/eval_only_lora_window.py \
    --model $MODEL --output_dir $SW/gemma2_9b_sweep_L24_29 \
    --eval_batch_size 8 --n_eval 500

$PY $ROOT/eval_only_lora_window.py \
    --model $MODEL --output_dir $SW/gemma2_9b_sweep_L30_35 \
    --eval_batch_size 8 --n_eval 500

$PY $ROOT/eval_only_lora_window.py \
    --model $MODEL --output_dir $SW/gemma2_9b_sweep_L36_41 \
    --eval_batch_size 8 --n_eval 500

echo "--- Gemma-2-9B eval complete ---"
