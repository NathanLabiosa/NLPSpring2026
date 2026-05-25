#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_v14_%j.out
#SBATCH --error=logs/train_v14_%j.err

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "--- STARTING LoRA TRAINING (v14) ---"
nvidia-smi

mkdir -p logs

# v14: LoRA fine-tuning on perturbed data
#
# Motivation:
#   v1–v13 all used external correction modules (MLPs, gates) injected into
#   the frozen model's residual stream. The single-type diagnostic (v13b)
#   showed these modules cannot learn per-type correction directions — even
#   trained on a single perturbation type they fail to help their own type.
#   Root cause: Phi-3.5's spike-and-suppress pattern has already collapsed
#   the perturbation-specific signal at layers 2–4.
#
# Design (v14):
#   LoRA adapters on layers 0–8 (q_proj, v_proj), rank=16, alpha=32.
#   No external modules — the model's own attention mechanism becomes the
#   correction, working with the architecture rather than against it.
#   L_acc (task CE on perturbed input) is the primary signal.
#   L_stab (MSE of hidden states at layer 9 on clean inputs) prevents
#   clean-input regression.
#
# Success criteria:
#   Clean baseline Δ ≥ -1%
#   No perturbation type worse than -2%
#   ≥ 3 perturbation types with Δ > +2%
#   Typos and OCR both ≥ 0%

python train_lrd_lora_v14.py \
    --model    microsoft/Phi-3.5-mini-instruct \
    --dataset  mixed \
    --output_dir ./stabilizer_weights/lora_v14 \
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
    --epochs 20 \
    --max_steps 25000 \
    --batch_size 4 \
    --grad_accum_steps 4 \
    --lr 2e-4 \
    --warmup_ratio 0.05 \
    --max_seq_len 512 \
    --log_every 25 \
    --save_every 2500 \
    --clean_eval_every 500 \
    --eval_after_training \
    --eval_n_samples 200
