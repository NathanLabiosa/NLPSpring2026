#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --time=04:00:00
#SBATCH --output=/home1/labiosa/NLPSpring2026/models/gemma2/logs/sweep_%j.out
#SBATCH --error=/home1/labiosa/NLPSpring2026/models/gemma2/logs/sweep_%j.err

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project_gemma
export LD_LIBRARY_PATH=/home1/labiosa/miniconda3/envs/my_project/lib:$LD_LIBRARY_PATH
export HF_TOKEN=$(cat /home1/labiosa/.cache/huggingface/token)

echo "--- Gemma-2-9B base LoRA Window Sweep (42 layers -> 7 windows of 6) ---"
echo "--- Step A5: LoRA rank 4, alpha 8, q_proj+v_proj, 1000 steps ---"
nvidia-smi

ROOT=/home1/labiosa/NLPSpring2026
PY=/home1/labiosa/miniconda3/envs/my_project_gemma/bin/python
MODEL=google/gemma-2-9b
SW=$ROOT/stabilizer_weights

# Gemma-2-9B has 42 layers -> 7 windows of 6 layers each
# 1000 steps per plan (7B+ model)

$PY $ROOT/sweep_lora_window.py \
    --model $MODEL --layer_start 0  --layer_end 5  \
    --n_steps 1000 --output_dir $SW/gemma2_9b_sweep_L00_05 \
    --batch_size 1

$PY $ROOT/sweep_lora_window.py \
    --model $MODEL --layer_start 6  --layer_end 11 \
    --n_steps 1000 --output_dir $SW/gemma2_9b_sweep_L06_11 \
    --batch_size 1

$PY $ROOT/sweep_lora_window.py \
    --model $MODEL --layer_start 12 --layer_end 17 \
    --n_steps 1000 --output_dir $SW/gemma2_9b_sweep_L12_17 \
    --batch_size 1

$PY $ROOT/sweep_lora_window.py \
    --model $MODEL --layer_start 18 --layer_end 23 \
    --n_steps 1000 --output_dir $SW/gemma2_9b_sweep_L18_23 \
    --batch_size 1

$PY $ROOT/sweep_lora_window.py \
    --model $MODEL --layer_start 24 --layer_end 29 \
    --n_steps 1000 --output_dir $SW/gemma2_9b_sweep_L24_29 \
    --batch_size 1

$PY $ROOT/sweep_lora_window.py \
    --model $MODEL --layer_start 30 --layer_end 35 \
    --n_steps 1000 --output_dir $SW/gemma2_9b_sweep_L30_35 \
    --batch_size 1

$PY $ROOT/sweep_lora_window.py \
    --model $MODEL --layer_start 36 --layer_end 41 \
    --n_steps 1000 --output_dir $SW/gemma2_9b_sweep_L36_41 \
    --batch_size 1

echo "--- Gemma-2-9B sweep complete ---"
