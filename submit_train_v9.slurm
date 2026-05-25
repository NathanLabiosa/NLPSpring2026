#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_v9_%j.out
#SBATCH --error=logs/train_v9_%j.err

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

echo "--- STARTING STABILIZER TRAINING (v9) ---"
nvidia-smi

mkdir -p logs

# v9 Architecture: Replace contrastive loss with embed MSE at stage 1
#
#   Core change — swap L_contrastive → L_embed_mse at stage 1:
#     v7/v8 used: L_ctr = relu(d_corr - d_noisy + margin)
#     This requires a detectable shift in cosine distance. With max_norm=1.0
#     and embedding magnitude ~55, correction is 0.45% of embedding scale,
#     so d_corr ≈ d_noisy always → gradient underflows to zero. Result: ctr
#     pinned at margin=0.10 for all 8725 steps of v8 with r1_max ≈ 1.000.
#
#     v9 uses: L_embed_mse = MSE(h_corrected[:, :T_min], h_clean[:, :T_min])
#     MSE gives gradient proportional to (h_corr - h_clean), which at positions
#     where the token changed is O(20-50) — a strong signal regardless of
#     correction magnitude. T_min truncation handles tokenization-length
#     mismatches (typo splitting one token into three, etc.).
#
#   max_norm 1.0 → 0.05 (semantic change: fraction of input norm, not absolute):
#     v7/v8 max_norm was an absolute cap (delta ≤ 1.0 units) regardless of the
#     embedding magnitude. With ‖h‖≈55 for Phi-3.5 embeddings, that was a cap of
#     ~1.8% of embedding magnitude. Now max_norm is a fraction: delta ≤ max_norm*‖h‖.
#     With max_norm=0.05, the cap is 0.05*55 ≈ 2.75 units (~5% of embedding norm).
#     This scales automatically with the local embedding magnitude at each layer.
#
#   All other hyperparameters unchanged from v8:
#     lambda_suppress=0.35, dataset=mixed, stage 2 config, curriculum, etc.
#
# KEY DIAGNOSTIC: Check emb_mse at step 3000.
#   Target: emb_mse DECREASING from step 0 (any downward trend confirms gradient flows)
#   Target: r1_max > 1.3  (gate opening more on noisy than clean steps)
#   Target: gnorm > 0 on most noisy steps (no more mass zero-gradient steps)
#   Failure: emb_mse flat/pinned or gnorm still mostly 0.000 → correction scale issue
#
# Training data: 40% GSM8K / 30% MMLU / 30% C4 (multidomain)
# Evaluation: GSM8K-only (comparable to v1-v8)
python train_lrd_stabilizer.py \
    --model    microsoft/Phi-3.5-mini-instruct \
    --dataset  mixed \
    --output_dir ./stabilizer_weights/phi35_mixed_mlp_v9 \
    --inject_layers 2 4 \
    --bottleneck_dim 128 \
    --stage2_bottleneck_dim 32 \
    --max_norm 0.05 \
    --gate_dim 64 \
    --lambda_embed_mse   0.35 \
    --lambda_stab        0.15 \
    --lambda_acc         0.40 \
    --lambda_prop        0.10 \
    --lambda_gate        0.15 \
    --lambda_suppress    0.35 \
    --lambda_adapter     0.15 \
    --probe_layers 9 \
    --ema_decay 0.99 \
    --reg_warmup_steps    2000 \
    --reg_decay_end_steps 8000 \
    --stab_warmup_frac    0.25 \
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
