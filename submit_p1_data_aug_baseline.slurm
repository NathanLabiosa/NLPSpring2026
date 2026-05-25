#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --output=logs/p1_data_aug_baseline_%j.out
#SBATCH --error=logs/p1_data_aug_baseline_%j.err

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "--- P1: Standard data augmentation baseline (all-layer LoRA, perturbed data) ---"
nvidia-smi
mkdir -p logs

# Priority 1, Experiment 4: Standard data augmentation baselines
#
# What a practitioner would try first: LoRA on ALL 32 layers with
# perturbed GSM8K data, no diagnostic guidance.
#
# Two variants run back-to-back:
#
#   (a) Rank 4 — apples-to-apples rank comparison with v17.
#       Same rank/alpha per layer, but 32 layers vs 5 → 6.4× more params.
#       Already covered by submit_train_phi35_naive_alllayers.slurm;
#       this script runs the rank-8 variant only.
#
#   (b) Rank 8 — simulates "practitioner best effort" with more capacity.
#       32 layers × rank 8 = ~12.8× more params than v17's 5 layers × rank 4.
#       NOTE FOR PAPER: call out that this has significantly more parameters.
#       If v17 matches or beats this despite 12.8× fewer params, it's a strong
#       argument for diagnostic-informed placement.

python train_lrd_lora_v14.py \
    --model    microsoft/Phi-3.5-mini-instruct \
    --dataset  gsm8k \
    --output_dir ./stabilizer_weights/p1_data_aug_r8 \
    --lora_rank    8 \
    --lora_alpha   16 \
    --lora_dropout 0.05 \
    --lora_layers  32 \
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
    --eval_after_training \
    --eval_n_samples 500
