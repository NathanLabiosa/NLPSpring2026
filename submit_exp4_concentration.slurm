#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --array=1-3
#SBATCH --output=logs/exp4_concentration_%a_%j.out
#SBATCH --error=logs/exp4_concentration_%a_%j.err

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs

# Exp 4: Concentration comparison at lambda_stab=3.0 on Phi-3.5.
# Prior concentration experiment used lambda_stab=0.3 (effectively off).
# Tests: concentrated 5-layer (L15-19) vs all-layer rank-4 vs all-layer rank-8.
# Success criterion: concentrated 5-layer still beats both all-layer configs.

COMMON_ARGS="--model microsoft/Phi-3.5-mini-instruct \
    --dataset gsm8k \
    --lora_dropout 0.05 \
    --target_modules qkv_proj o_proj \
    --lambda_acc 0.7 \
    --lambda_stab 3.0 \
    --stab_cosine \
    --n_per_condition 857 \
    --clean_fraction 0.25 \
    --epochs 30 \
    --max_steps 300 \
    --batch_size 4 \
    --grad_accum_steps 4 \
    --lr 5e-5 \
    --warmup_ratio 0.05 \
    --max_seq_len 512 \
    --log_every 50 \
    --save_every 300 \
    --eval_every 0 \
    --clean_eval_every 300 \
    --early_stop_delta 5.0 \
    --clean_eval_n 200 \
    --eval_after_training \
    --eval_n_samples 500 \
    --seed 42"

if [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then
    echo "--- Exp4: Concentrated 5-layer (L15-19), rank=4, lambda_stab=3.0 ---"
    python train_lrd_lora_v14.py $COMMON_ARGS \
        --lora_rank 4 --lora_alpha 8 \
        --lora_layers 5 --lora_layers_start 15 \
        --stab_layer 24 \
        --output_dir ./stabilizer_weights/exp4_concentrated_r4

elif [ "$SLURM_ARRAY_TASK_ID" -eq 2 ]; then
    echo "--- Exp4: All-layer, rank=4, lambda_stab=3.0 ---"
    python train_lrd_lora_v14.py $COMMON_ARGS \
        --lora_rank 4 --lora_alpha 8 \
        --lora_layers 32 --lora_layers_start 0 \
        --stab_layer 24 \
        --output_dir ./stabilizer_weights/exp4_alllayer_r4

elif [ "$SLURM_ARRAY_TASK_ID" -eq 3 ]; then
    echo "--- Exp4: All-layer, rank=8, lambda_stab=3.0 ---"
    python train_lrd_lora_v14.py $COMMON_ARGS \
        --lora_rank 8 --lora_alpha 16 \
        --lora_layers 32 --lora_layers_start 0 \
        --stab_layer 24 \
        --output_dir ./stabilizer_weights/exp4_alllayer_r8
fi
