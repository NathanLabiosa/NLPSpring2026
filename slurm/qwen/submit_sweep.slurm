#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --exclude=a02-15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --time=48:00:00
#SBATCH --output=/home1/labiosa/NLPSpring2026/models/qwen2.5/logs/sweep_%j.out
#SBATCH --error=/home1/labiosa/NLPSpring2026/models/qwen2.5/logs/sweep_%j.err

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project_gemma
export LD_LIBRARY_PATH=/home1/labiosa/miniconda3/envs/my_project/lib:$LD_LIBRARY_PATH

echo "--- Qwen2.5-7B LoRA Window Sweep (28 layers -> 7 windows of 4) ---"
echo "--- Step B4: LoRA rank 4, alpha 8, q_proj+v_proj, 1000 steps ---"
nvidia-smi

ROOT=/home1/labiosa/NLPSpring2026
PY=/home1/labiosa/miniconda3/envs/my_project_gemma/bin/python
MODEL=Qwen/Qwen2.5-7B-Instruct
SW=$ROOT/stabilizer_weights

# Qwen2.5-7B has 28 layers -> 7 non-overlapping 4-layer windows
# 1000 steps (7B model per plan)

$PY $ROOT/sweep_lora_window.py \
    --model $MODEL --layer_start 0  --layer_end 3  \
    --n_steps 1000 --output_dir $SW/qwen_sweep_L00_03

$PY $ROOT/sweep_lora_window.py \
    --model $MODEL --layer_start 4  --layer_end 7  \
    --n_steps 1000 --output_dir $SW/qwen_sweep_L04_07

$PY $ROOT/sweep_lora_window.py \
    --model $MODEL --layer_start 8  --layer_end 11 \
    --n_steps 1000 --output_dir $SW/qwen_sweep_L08_11

$PY $ROOT/sweep_lora_window.py \
    --model $MODEL --layer_start 12 --layer_end 15 \
    --n_steps 1000 --output_dir $SW/qwen_sweep_L12_15

$PY $ROOT/sweep_lora_window.py \
    --model $MODEL --layer_start 16 --layer_end 19 \
    --n_steps 1000 --output_dir $SW/qwen_sweep_L16_19

$PY $ROOT/sweep_lora_window.py \
    --model $MODEL --layer_start 20 --layer_end 23 \
    --n_steps 1000 --output_dir $SW/qwen_sweep_L20_23

$PY $ROOT/sweep_lora_window.py \
    --model $MODEL --layer_start 24 --layer_end 27 \
    --n_steps 1000 --output_dir $SW/qwen_sweep_L24_27

echo "--- Qwen2.5-7B sweep complete ---"
