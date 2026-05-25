#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --array=1-6
#SBATCH --output=logs/exp1c_mistral_rank_%a_%j.out
#SBATCH --error=logs/exp1c_mistral_rank_%a_%j.err

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -f $HF_HOME/token ] && export HF_TOKEN=$(cat $HF_HOME/token)

mkdir -p logs

# Exp 1c: Mistral-7B rank scaling at layers 20-24 (LRD-predicted optimum).
# Mirror of Exp 1a for Mistral. Success criterion: avg perturbed Δ > +2%.

COMMON_ARGS="--model mistralai/Mistral-7B-Instruct-v0.3 \
    --dataset gsm8k \
    --lora_layers 5 \
    --lora_layers_start 20 \
    --stab_layer 29 \
    --target_modules q_proj v_proj \
    --lora_dropout 0.05 \
    --lambda_acc 0.7 \
    --lambda_stab 3.0 \
    --stab_cosine \
    --n_per_condition 857 \
    --clean_fraction 0.25 \
    --epochs 30 \
    --batch_size 2 \
    --grad_accum_steps 8 \
    --lr 2e-5 \
    --warmup_ratio 0.05 \
    --max_seq_len 512 \
    --log_every 100 \
    --save_every 5000 \
    --eval_every 0 \
    --clean_eval_every 500 \
    --early_stop_delta 15.0 \
    --clean_eval_n 200 \
    --eval_after_training \
    --eval_n_samples 500 \
    --seed 42"

if [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then
    echo "--- Exp1c: Mistral-7B, rank=16, steps=3000 ---"
    python train_lrd_lora_v14.py $COMMON_ARGS \
        --lora_rank 16 --lora_alpha 32 --max_steps 3000 \
        --output_dir ./stabilizer_weights/exp1c_mistral_r16_s3000

elif [ "$SLURM_ARRAY_TASK_ID" -eq 2 ]; then
    echo "--- Exp1c: Mistral-7B, rank=16, steps=5000 ---"
    python train_lrd_lora_v14.py $COMMON_ARGS \
        --lora_rank 16 --lora_alpha 32 --max_steps 5000 \
        --output_dir ./stabilizer_weights/exp1c_mistral_r16_s5000

elif [ "$SLURM_ARRAY_TASK_ID" -eq 3 ]; then
    echo "--- Exp1c: Mistral-7B, rank=32, steps=3000 ---"
    python train_lrd_lora_v14.py $COMMON_ARGS \
        --lora_rank 32 --lora_alpha 64 --max_steps 3000 \
        --output_dir ./stabilizer_weights/exp1c_mistral_r32_s3000

elif [ "$SLURM_ARRAY_TASK_ID" -eq 4 ]; then
    echo "--- Exp1c: Mistral-7B, rank=32, steps=5000 ---"
    python train_lrd_lora_v14.py $COMMON_ARGS \
        --lora_rank 32 --lora_alpha 64 --max_steps 5000 \
        --output_dir ./stabilizer_weights/exp1c_mistral_r32_s5000

elif [ "$SLURM_ARRAY_TASK_ID" -eq 5 ]; then
    echo "--- Exp1c: Mistral-7B, rank=64, steps=3000 ---"
    python train_lrd_lora_v14.py $COMMON_ARGS \
        --lora_rank 64 --lora_alpha 128 --max_steps 3000 \
        --output_dir ./stabilizer_weights/exp1c_mistral_r64_s3000

elif [ "$SLURM_ARRAY_TASK_ID" -eq 6 ]; then
    echo "--- Exp1c: Mistral-7B, rank=64, steps=5000 ---"
    python train_lrd_lora_v14.py $COMMON_ARGS \
        --lora_rank 64 --lora_alpha 128 --max_steps 5000 \
        --output_dir ./stabilizer_weights/exp1c_mistral_r64_s5000
fi
