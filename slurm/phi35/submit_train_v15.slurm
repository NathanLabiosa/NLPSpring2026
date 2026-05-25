#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_v15_%j.out
#SBATCH --error=logs/train_v15_%j.err

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "--- STARTING LoRA TRAINING (v15) ---"
nvidia-smi
mkdir -p logs

# v15: GSM8K-only LoRA fine-tuning
#
# Changes from v14:
#   - dataset: gsm8k only (removes MMLU/C4 format confusion)
#   - max_steps: 5000 (was 25000; ~23 epochs on 875 GSM8K pairs)
#   - lr: 1e-4 (was 2e-4; reduces overfitting risk on small dataset)
#   - early_stop_delta: 5.0 (stop if clean drops >5% below pre-training baseline)
#   - eval_n_samples: 200 for final eval
#
# Architecture unchanged from v14:
#   LoRA rank=16, alpha=32, layers 0–8, qkv_proj + o_proj

python src/training/train_lrd_lora_v14.py \
    --model    microsoft/Phi-3.5-mini-instruct \
    --dataset  gsm8k \
    --output_dir ./stabilizer_weights/lora_v15 \
    --lora_rank    16 \
    --lora_alpha   32 \
    --lora_dropout 0.05 \
    --lora_layers  9 \
    --target_modules qkv_proj o_proj \
    --lambda_acc   0.7 \
    --lambda_stab  0.3 \
    --stab_layer   9 \
    --n_per_condition 500 \
    --clean_fraction  0.25 \
    --epochs 30 \
    --max_steps 5000 \
    --batch_size 4 \
    --grad_accum_steps 4 \
    --lr 1e-4 \
    --warmup_ratio 0.05 \
    --max_seq_len 512 \
    --log_every 25 \
    --save_every 500 \
    --eval_every 500 \
    --clean_eval_every 500 \
    --early_stop_delta 5.0 \
    --clean_eval_n 500 \
    --eval_after_training \
    --eval_n_samples 500
