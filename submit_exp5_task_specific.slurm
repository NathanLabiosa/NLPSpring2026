#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --array=1-1
#SBATCH --output=logs/exp5_task_specific_%a_%j.out
#SBATCH --error=logs/exp5_task_specific_%a_%j.err

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs

# Exp 5: Task-specific LoRA on MMLU using Phi-3.5, layers 15-19.
# Tests whether training on target-task perturbations fixes the negative
# transfer seen when using the GSM8K adapter on MMLU (-1.5% to -4.5%).
# Success criterion: positive avg perturbed Δ on MMLU.
#
# Note: BBH dataset builder not yet implemented in train_lrd_stabilizer.py.
# Add build_bbh_pairs() there and extend DATASET_BUILDERS to run BBH.

if [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then
    echo "--- Exp5: Task-specific MMLU, Phi-3.5, L15-19 ---"
    python train_lrd_lora_v14.py \
        --model microsoft/Phi-3.5-mini-instruct \
        --dataset mmlu \
        --lora_rank 4 \
        --lora_alpha 8 \
        --lora_dropout 0.05 \
        --lora_layers 5 \
        --lora_layers_start 15 \
        --stab_layer 24 \
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
        --seed 42 \
        --output_dir ./stabilizer_weights/exp5_mmlu_phi_L15_19
fi
