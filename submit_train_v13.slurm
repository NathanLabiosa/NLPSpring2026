#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_v13_%j.out
#SBATCH --error=logs/train_v13_%j.err

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

echo "--- STARTING STABILIZER TRAINING (v13) ---"
nvidia-smi

mkdir -p logs

# v13: Gateless MLP at layers 2 and 4 (v6 topology + all training improvements)
#
# Motivation:
#   v9–v12 all failed due to the gate bootstrap deadlock:
#     fc2 random → MSE gradient to α incoherent → suppress wins → gate collapses.
#     Hard-freezing α=1.0 (v12) correctly isolated fc2 training, but embed_mse
#     (0.35 × 0.002 = 0.0007) was drowned out by L_stab (0.15 × 0.4 = 0.06) and
#     L_acc (~0.4), giving fc2 < 0.2% of the total gradient budget — no learning.
#
# Design (v13):
#   Remove the gate entirely and Stage 1.  Return to v6's topology:
#     - Stage 2 only: bottleneck MLP at layers 2 and 4
#     - Delta applied directly: h_out = h + delta  (no gate, no alpha)
#     - Norm clip at max_norm=0.025 (2.5% of ‖h‖ ≈ 1.4 units) controls
#       clean regression risk.  In v6 the regression was caused by an
#       unconstrained MLP; adaptive norm clip was not in place then.
#   All training fixes from v9–v12 are retained:
#     - Per-example loss routing (noisy → L_stab; both → L_acc/L_prop)
#     - Balanced perturbation pool (7 conditions, homophones collapsed,
#       OCR at 5% and 15%)
#     - Changed-token masking in embed_mse (unused here but code is present)
#     - Mixed dataset (40% GSM8K / 30% MMLU / 30% C4)
#
# Clean regression risk:
#   delta is always applied on every forward pass (noisy AND clean).
#   The norm clip limits the maximum perturbation to 2.5% of ‖h‖ per token.
#   L_stab (cosine MSE) gives fc2 gradient on noisy steps to correct toward
#   h_clean.  On clean steps, fc2 receives gradient from L_acc + L_prop only
#   (L_stab ≈ 0 when h_noisy ≈ h_clean).  fc2 is zero-init → exact no-op at
#   step 0.  The norm clip prevents runaway corrections.
#
# Key diagnostic:
#   L_stab should decrease monotonically as fc2 learns.  If it plateaus above
#   0.3 after 5000 steps, max_norm=0.025 may be too tight — try 0.04.
#   Monitor clean-set accuracy (eval) to check regression.
#
python train_lrd_stabilizer.py \
    --model    microsoft/Phi-3.5-mini-instruct \
    --dataset  mixed \
    --output_dir ./stabilizer_weights/phi35_mixed_mlp_v13 \
    --inject_layers 2 4 \
    --stage2_bottleneck_dim 32 \
    --no_gate \
    --no_embed_stage \
    --max_norm 0.025 \
    --lambda_embed_mse   0.0 \
    --lambda_stab        0.5 \
    --lambda_acc         0.4 \
    --lambda_prop        0.1 \
    --lambda_gate        0.0 \
    --lambda_suppress    0.0 \
    --lambda_adapter     0.0 \
    --probe_layers 9 \
    --stab_warmup_frac   0.10 \
    --gate_warmup_steps  0 \
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
