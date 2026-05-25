#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=4:00:00
#SBATCH --output=logs/diag_%j.out
#SBATCH --error=logs/diag_%j.err

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

echo "--- DIAGNOSTIC RUN: teacher-forcing fix check ---"
nvidia-smi

mkdir -p logs

# 2000-step diagnostic to confirm the teacher-forcing shortcut is closed.
# Expected signal: L_acc should stay in ~0.3-0.6 throughout rather than
# collapsing toward 0.07 as in v1/v2. If L_acc dives below 0.15 before
# step 500, the shortcut is still open.
#
# Curriculum is active (reg_mult: 3.0→1.0, stab warmup 25%=500 steps).
# Using rank-16 LoRA and rebalanced weights matching the full v3 run,
# so this diagnostic is predictive of full training behavior.
python src/lib/train_lrd_stabilizer.py \
    --model    microsoft/Phi-3.5-mini-instruct \
    --dataset  gsm8k \
    --output_dir ./stabilizer_weights/phi35_gsm8k_diag \
    --inject_layers 1 2 3 4 5 6 7 8 \
    --probe_layers 9 12 \
    --lora_rank      16 \
    --lambda_stab    0.30 \
    --lambda_acc     0.40 \
    --lambda_prop    0.10 \
    --lambda_dir     0.00 \
    --lambda_gate    0.30 \
    --lambda_suppress 0.30 \
    --lambda_adapter  0.20 \
    --lambda_reg_start 3.0 \
    --stab_warmup_frac 0.25 \
    --epochs 10 \
    --max_steps 2000 \
    --batch_size 4 \
    --grad_accum_steps 4 \
    --n_per_condition 500 \
    --max_seq_len 512 \
    --log_every 10 \
    --save_every 500 \
    --no_eval_after_training
