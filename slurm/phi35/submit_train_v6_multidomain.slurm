#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_v6_multidomain_%j.out
#SBATCH --error=logs/train_v6_multidomain_%j.err

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

echo "--- STARTING STABILIZER TRAINING (v6 multidomain) ---"
nvidia-smi

mkdir -p logs

# v6_multidomain: same as v6 (LRD-supervised gate + gate input augmentation)
# but trained on a mixed dataset: 40% GSM8K / 30% MMLU / 30% C4.
# Evaluation remains GSM8K-only so results are comparable to v1-v5 and v6.
#
# Key differences vs v6 (phi35_gsm8k_mlp_v6):
#   - dataset=mixed  (40% GSM8K, 30% MMLU, 30% C4 per perturbation condition)
#   - output_dir → phi35_mixed_mlp_v6
#   - All other hyperparameters identical to v6.
#
# Effective weights at step 0:    gate=0.90, suppress=0.90, adapter=0.60, stab=0.00, lrd=0.40
# Effective weights at step 3750:  gate=0.60, suppress=0.60, adapter=0.40, stab=0.30 (full)
# Effective weights at step 25000: gate=0.30, suppress=0.30, adapter=0.20, stab=0.30, lrd=0.40
python src/lib/train_lrd_stabilizer.py \
    --model    microsoft/Phi-3.5-mini-instruct \
    --dataset  mixed \
    --output_dir ./stabilizer_weights/phi35_mixed_mlp_v6 \
    --inject_layers 1 2 3 4 5 6 7 8 \
    --probe_layers 9 12 \
    --bottleneck_dim 128 \
    --max_norm       1.0 \
    --lambda_stab    0.30 \
    --lambda_acc     0.40 \
    --lambda_prop    0.10 \
    --lambda_dir     0.00 \
    --lambda_lrd     0.40 \
    --lambda_gate    0.30 \
    --lambda_suppress 0.30 \
    --lambda_adapter  0.20 \
    --lambda_reg_start 3.0 \
    --stab_warmup_frac 0.25 \
    --gate_aug \
    --ema_decay      0.99 \
    --gate_aug_warmup 500 \
    --epochs 20 \
    --max_steps 25000 \
    --batch_size 4 \
    --grad_accum_steps 4 \
    --n_per_condition 500 \
    --max_seq_len 512 \
    --log_every 25 \
    --save_every 500 \
    --eval_after_training \
    --eval_n_samples 100
