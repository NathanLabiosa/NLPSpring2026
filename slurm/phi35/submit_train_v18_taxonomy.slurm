#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_v18_taxonomy_%j.out
#SBATCH --error=logs/train_v18_taxonomy_%j.err

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "--- TAXONOMY ABLATION: v17 + taxonomy loss weights ---"
nvidia-smi
mkdir -p logs

# Taxonomy ablation: identical to v17 except --taxonomy_weights flag.
# whitespace and case examples are weighted at 0.3x CE; directional types
# (typos, OCR, speech, homophones) at 1.0x.
# Hypothesis: downweighting uniform perturbation types focuses LoRA capacity
# on failure-predictive types and improves robustness on those conditions.
# v17 baseline uses flat 1.0x weighting (no taxonomy) → compare directly.

python src/training/train_lrd_lora_v14.py \
    --model    microsoft/Phi-3.5-mini-instruct \
    --dataset  gsm8k \
    --output_dir ./stabilizer_weights/lora_v18_taxonomy \
    --lora_rank    4 \
    --lora_alpha   8 \
    --lora_dropout 0.05 \
    --lora_layers  5 \
    --lora_layers_start 0 \
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
    --taxonomy_weights \
    --eval_after_training \
    --eval_n_samples 500
