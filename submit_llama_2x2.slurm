#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --array=1-4
#SBATCH --output=logs/llama_2x2_%a_%j.out
#SBATCH --error=logs/llama_2x2_%a_%j.err

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -f $HF_HOME/token ] && export HF_TOKEN=$(cat $HF_HOME/token)

mkdir -p logs

# Llama-3 8B 2×2: (cosine vs MSE) × (close=29 vs far=31)
# Best window from sweep: layers 20–24.
# close stab_layer = 24+5 = 29
# far   stab_layer = 24+7 = 31 (capped; 24+11=35 exceeds 31)
# 3000 steps (sweep at 1000 was marginal for 8B model).

COMMON_ARGS="--model meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset gsm8k \
    --lora_rank 4 \
    --lora_alpha 8 \
    --lora_dropout 0.05 \
    --lora_layers 5 \
    --lora_layers_start 20 \
    --target_modules q_proj v_proj \
    --lambda_acc 0.7 \
    --lambda_stab 3.0 \
    --n_per_condition 857 \
    --clean_fraction 0.25 \
    --epochs 30 \
    --max_steps 3000 \
    --batch_size 2 \
    --grad_accum_steps 8 \
    --lr 2e-5 \
    --warmup_ratio 0.05 \
    --max_seq_len 512 \
    --log_every 50 \
    --save_every 3000 \
    --eval_every 0 \
    --clean_eval_every 500 \
    --early_stop_delta 15.0 \
    --clean_eval_n 200 \
    --eval_after_training \
    --eval_n_samples 500"

if [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then
    echo "--- Llama 2x2: Cosine, close (stab=29) ---"
    python train_lrd_lora_v14.py $COMMON_ARGS \
        --stab_cosine --stab_layer 29 \
        --output_dir ./stabilizer_weights/llama_2x2_cos_close

elif [ "$SLURM_ARRAY_TASK_ID" -eq 2 ]; then
    echo "--- Llama 2x2: Cosine, far (stab=31) ---"
    python train_lrd_lora_v14.py $COMMON_ARGS \
        --stab_cosine --stab_layer 31 \
        --output_dir ./stabilizer_weights/llama_2x2_cos_far

elif [ "$SLURM_ARRAY_TASK_ID" -eq 3 ]; then
    echo "--- Llama 2x2: MSE, close (stab=29) ---"
    python train_lrd_lora_v14.py $COMMON_ARGS \
        --stab_layer 29 \
        --output_dir ./stabilizer_weights/llama_2x2_mse_close

elif [ "$SLURM_ARRAY_TASK_ID" -eq 4 ]; then
    echo "--- Llama 2x2: MSE, far (stab=31) ---"
    python train_lrd_lora_v14.py $COMMON_ARGS \
        --stab_layer 31 \
        --output_dir ./stabilizer_weights/llama_2x2_mse_far
fi
