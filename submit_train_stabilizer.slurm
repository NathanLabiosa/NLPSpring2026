#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_stabilizer_%j.out
#SBATCH --error=logs/train_stabilizer_%j.err

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

echo "--- STARTING STABILIZER TRAINING ---"
nvidia-smi

mkdir -p logs

# v6: LRD-supervised gate + gate input augmentation
#
# Key changes vs v5:
#   - lambda_lrd=0.4: gate supervised toward batch-normalised cosine LRD per layer
#     (noisy steps only). Forces gate to open proportionally to hidden-state divergence.
#   - gate_aug=True: gate input becomes concat([h_norm, |h_norm - h_clean_mean|]) (2D).
#     h_clean_mean is a per-layer EMA (decay=0.99) over clean steps. Gives the gate an
#     explicit divergence-from-clean signal without requiring clean states at inference.
#   - gate_aug_warmup=500: gate reads [h_norm, zeros] for first 500 steps while EMA
#     stabilises; augmentation kicks in at step 500.
#   - All v5 settings preserved: bottleneck MLP, norm clip, curriculum, whitespace
#     downweighting, per-condition gate logging, 25k steps.
#   - output_dir → phi35_gsm8k_mlp_v6
#
# Effective weights at step 0:    gate=0.90, suppress=0.90, adapter=0.60, stab=0.00, lrd=0.40
# Effective weights at step 3750:  gate=0.60, suppress=0.60, adapter=0.40, stab=0.30 (full)
# Effective weights at step 25000: gate=0.30, suppress=0.30, adapter=0.20, stab=0.30, lrd=0.40
python train_lrd_stabilizer.py \
    --model    microsoft/Phi-3.5-mini-instruct \
    --dataset  gsm8k \
    --output_dir ./stabilizer_weights/phi35_gsm8k_mlp_v6 \
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

# lambda_lrd sweep (if v6 gate_cond_ocr still lags):
# python train_lrd_stabilizer.py ... --lambda_lrd 0.5 \
#     --output_dir ./stabilizer_weights/phi35_gsm8k_mlp_v6_lrd05 [same args as above]
#
# no_gate_aug ablation (to isolate LRD supervision contribution):
# python train_lrd_stabilizer.py ... --no_gate_aug --lambda_lrd 0.4 \
#     --output_dir ./stabilizer_weights/phi35_gsm8k_mlp_v6_noaug [same args as above]
