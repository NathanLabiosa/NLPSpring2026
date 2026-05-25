#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --output=/home1/labiosa/NLPSpring2026/TinyLlama/logs/sweep_%j.out
#SBATCH --error=/home1/labiosa/NLPSpring2026/TinyLlama/logs/sweep_%j.err

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

echo "--- TinyLlama LoRA Window Sweep (22 layers -> 5 windows of 4) ---"
echo "--- Step C5: LoRA rank 4, alpha 8, q_proj+v_proj, 300 steps ---"
nvidia-smi

ROOT=/home1/labiosa/NLPSpring2026
PY=/home1/labiosa/miniconda3/envs/my_project/bin/python
MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0
SW=$ROOT/stabilizer_weights

# TinyLlama has 22 layers -> 5 non-overlapping 4-layer windows (skip last 2 layers)
# 300 steps (sub-3B model per plan)

$PY $ROOT/sweep_lora_window.py \
    --model $MODEL --layer_start 0  --layer_end 3  \
    --n_steps 300 --output_dir $SW/tinyllama_sweep_L00_03

$PY $ROOT/sweep_lora_window.py \
    --model $MODEL --layer_start 4  --layer_end 7  \
    --n_steps 300 --output_dir $SW/tinyllama_sweep_L04_07

$PY $ROOT/sweep_lora_window.py \
    --model $MODEL --layer_start 8  --layer_end 11 \
    --n_steps 300 --output_dir $SW/tinyllama_sweep_L08_11

$PY $ROOT/sweep_lora_window.py \
    --model $MODEL --layer_start 12 --layer_end 15 \
    --n_steps 300 --output_dir $SW/tinyllama_sweep_L12_15

$PY $ROOT/sweep_lora_window.py \
    --model $MODEL --layer_start 16 --layer_end 19 \
    --n_steps 300 --output_dir $SW/tinyllama_sweep_L16_19

echo "--- TinyLlama sweep complete ---"
