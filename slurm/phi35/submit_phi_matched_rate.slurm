#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --array=1-2
#SBATCH --output=logs/phi_matched_rate_%a_%j.out
#SBATCH --error=logs/phi_matched_rate_%a_%j.err

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs

# B2: Matched-rate perturbation comparison on Phi-3.5
# All 6 perturbation types at 10% rate (eliminating rate as confound).
# 2×2 design: cosine vs MSE at stab=24, layers 15-19, lambda=3.0.
# Everything else identical to headline config.

COMMON_ARGS="--model microsoft/Phi-3.5-mini-instruct \
    --dataset gsm8k \
    --lora_rank 4 \
    --lora_alpha 8 \
    --lora_dropout 0.05 \
    --lora_layers 5 \
    --lora_layers_start 15 \
    --target_modules qkv_proj o_proj \
    --lambda_acc 0.7 \
    --lambda_stab 3.0 \
    --stab_layer 24 \
    --uniform_rate 0.10 \
    --n_per_condition 857 \
    --clean_fraction 0.25 \
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
    --eval_n_samples 500"

if [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then
    echo "--- Matched-rate: Cosine v2, L15-19, stab=24, lambda=3.0, all@10% ---"
    python src/training/train_lrd_lora_v14.py $COMMON_ARGS \
        --stab_cosine \
        --output_dir ./stabilizer_weights/phi_matched_cosine_stab24

elif [ "$SLURM_ARRAY_TASK_ID" -eq 2 ]; then
    echo "--- Matched-rate: MSE, L15-19, stab=24, lambda=3.0, all@10% ---"
    python src/training/train_lrd_lora_v14.py $COMMON_ARGS \
        --output_dir ./stabilizer_weights/phi_matched_mse_stab24
fi
