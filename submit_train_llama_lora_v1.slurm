#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=16:00:00
#SBATCH --output=logs/train_llama_lora_v1_%j.out
#SBATCH --error=logs/train_llama_lora_v1_%j.err

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Llama-3-8B is gated — token needed on first download; cached thereafter.
[ -f $HF_HOME/token ] && export HF_TOKEN=$(cat $HF_HOME/token)

echo "--- Llama-3 8B LoRA v1: informed placement (layers 0–8, causal window) ---"
nvidia-smi
mkdir -p logs

# Experiment 1 v2: LoRA at the confirmed causal window (layers 0–8, identity=90%).
#
# KEY FIX vs. prior runs: --stab_layer 12 (was 9).
# With LoRA on layers 0–8 and stab_layer=9, the hook lands on layers[8] — the
# LAST LoRA layer. Clean inputs show near-zero divergence there directly, so
# L_stab ≈ 0 throughout training and provides no regularization. This allowed
# the LoRA to freely disrupt downstream representations, causing -3 to -9%
# degradation across perturbed conditions.
# Fix: stab_layer=12 (layers[11]) is downstream of all LoRA layers. It measures
# propagated effects on clean representations — the same design as Phi-3.5 v17
# where stab_layer=9 was beyond the LoRA layers (0–4).
#
# Also: max_steps 600, lr 2e-5 — larger model needs more careful training.

python train_lrd_lora_v14.py \
    --model    meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset  gsm8k \
    --output_dir ./stabilizer_weights/llama_lora_v4 \
    --lora_rank    4 \
    --lora_alpha   8 \
    --lora_dropout 0.05 \
    --lora_layers  9 \
    --lora_layers_start 0 \
    --target_modules q_proj v_proj \
    --lambda_acc   0.7 \
    --lambda_stab  0.3 \
    --stab_layer   12 \
    --n_per_condition 857 \
    --clean_fraction  0.25 \
    --epochs 30 \
    --max_steps 300 \
    --batch_size 2 \
    --grad_accum_steps 8 \
    --lr 2e-5 \
    --warmup_ratio 0.05 \
    --max_seq_len 512 \
    --log_every 25 \
    --save_every 50 \
    --eval_every 0 \
    --clean_eval_every 100 \
    --early_stop_delta 15.0 \
    --clean_eval_n 200 \
    --eval_after_training \
    --eval_n_samples 500
