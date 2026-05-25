#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=16:00:00
#SBATCH --array=1-6
#SBATCH --output=logs/llama_layer_sweep_%a_%j.out
#SBATCH --error=logs/llama_layer_sweep_%a_%j.err

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -f $HF_HOME/token ] && export HF_TOKEN=$(cat $HF_HOME/token)

mkdir -p logs

# Llama-3 8B layer sweep: 6 windows of 5 layers, 1000 steps each.
# Uses cosine L_stab at lambda=3.0 (best Phi config).
# stab_layer = last_lora_layer + 5 (or capped at 31 for late windows).

COMMON_ARGS="--model meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset gsm8k \
    --lora_rank 4 \
    --lora_alpha 8 \
    --lora_dropout 0.05 \
    --lora_layers 5 \
    --target_modules q_proj v_proj \
    --lambda_acc 0.7 \
    --lambda_stab 3.0 \
    --stab_cosine \
    --n_per_condition 857 \
    --clean_fraction 0.25 \
    --epochs 30 \
    --max_steps 1000 \
    --batch_size 2 \
    --grad_accum_steps 8 \
    --lr 2e-5 \
    --warmup_ratio 0.05 \
    --max_seq_len 512 \
    --log_every 50 \
    --save_every 1000 \
    --eval_every 0 \
    --clean_eval_every 200 \
    --early_stop_delta 15.0 \
    --clean_eval_n 200 \
    --eval_after_training \
    --eval_n_samples 500"

if [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then
    echo "--- Llama-3 8B: Layers 0-4, stab_layer=9 ---"
    python train_lrd_lora_v14.py $COMMON_ARGS \
        --lora_layers_start 0 --stab_layer 9 \
        --output_dir ./stabilizer_weights/llama_sweep_L00_04

elif [ "$SLURM_ARRAY_TASK_ID" -eq 2 ]; then
    echo "--- Llama-3 8B: Layers 5-9, stab_layer=14 ---"
    python train_lrd_lora_v14.py $COMMON_ARGS \
        --lora_layers_start 5 --stab_layer 14 \
        --output_dir ./stabilizer_weights/llama_sweep_L05_09

elif [ "$SLURM_ARRAY_TASK_ID" -eq 3 ]; then
    echo "--- Llama-3 8B: Layers 10-14, stab_layer=19 ---"
    python train_lrd_lora_v14.py $COMMON_ARGS \
        --lora_layers_start 10 --stab_layer 19 \
        --output_dir ./stabilizer_weights/llama_sweep_L10_14

elif [ "$SLURM_ARRAY_TASK_ID" -eq 4 ]; then
    echo "--- Llama-3 8B: Layers 15-19, stab_layer=24 ---"
    python train_lrd_lora_v14.py $COMMON_ARGS \
        --lora_layers_start 15 --stab_layer 24 \
        --output_dir ./stabilizer_weights/llama_sweep_L15_19

elif [ "$SLURM_ARRAY_TASK_ID" -eq 5 ]; then
    echo "--- Llama-3 8B: Layers 20-24, stab_layer=29 ---"
    python train_lrd_lora_v14.py $COMMON_ARGS \
        --lora_layers_start 20 --stab_layer 29 \
        --output_dir ./stabilizer_weights/llama_sweep_L20_24

elif [ "$SLURM_ARRAY_TASK_ID" -eq 6 ]; then
    echo "--- Llama-3 8B: Layers 27-31, stab_layer=31 (no L_stab — can't go beyond last layer) ---"
    python train_lrd_lora_v14.py $COMMON_ARGS \
        --lora_layers_start 27 --stab_layer 31 \
        --lambda_stab 0.0 --lambda_acc 1.0 \
        --output_dir ./stabilizer_weights/llama_sweep_L27_31
fi
