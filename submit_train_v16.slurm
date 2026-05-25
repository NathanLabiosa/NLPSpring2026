#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_v16_%j.out
#SBATCH --error=logs/train_v16_%j.err

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "--- STARTING LoRA TRAINING (v16) ---"
nvidia-smi
mkdir -p logs

# v16: Highly constrained LoRA to reduce overfitting (Path A)
#
# Changes from v15:
#   - lora_rank: 4 (was 16) — ~10x fewer trainable params
#   - lora_alpha: 8 (was 32) — keep effective LR scaling proportional to rank
#   - lora_layers: 5 (was 9) — layers 0–4 only
#   - lr: 5e-5 (was 1e-4) — lower to avoid overwriting early layers
#   - max_steps: 500 (was 5000) — speech/typos signal peaked at step 500
#   - n_per_condition: 857 (was 500) — ~7500 total training examples
#     (7 conditions × 857 × 1.25 = 7498)
#   - eval_every: 100 (was 500) — finer-grained view within 500-step window
#   - save_every: 100 (was 500)
#   - early_stop_delta: 5.0, clean_eval_n: 500 unchanged

python train_lrd_lora_v14.py \
    --model    microsoft/Phi-3.5-mini-instruct \
    --dataset  gsm8k \
    --output_dir ./stabilizer_weights/lora_v17 \
    --lora_rank    4 \
    --lora_alpha   8 \
    --lora_dropout 0.05 \
    --lora_layers  5 \
    --target_modules qkv_proj o_proj \
    --lambda_acc   0.7 \
    --lambda_stab  0.3 \
    --stab_layer   9 \
    --n_per_condition 857 \
    --clean_fraction  0.25 \
    --epochs 30 \
    --max_steps 300 \
    --batch_size 4 \
    --grad_accum_steps 4 \
    --lr 5e-5 \
    --warmup_ratio 0.05 \
    --max_seq_len 512 \
    --log_every 25 \
    --save_every 300 \
    --eval_every 0 \
    --clean_eval_every 300 \
    --early_stop_delta 5.0 \
    --clean_eval_n 500 \
    --eval_after_training \
    --eval_n_samples 500
