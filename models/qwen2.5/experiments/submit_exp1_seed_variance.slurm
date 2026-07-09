#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --time=16:00:00
#SBATCH --array=0-2
#SBATCH --output=/home1/labiosa/NLPSpring2026/logs/qwen_exp1_seed%a_%j.out
#SBATCH --error=/home1/labiosa/NLPSpring2026/logs/qwen_exp1_seed%a_%j.err

# Experiment 1: Seed variance on +1 vanilla (Qwen2.5-7B)
# Replicates original qwen_sweep_L24_27 with stab_layer=28 (lora_end+1)
# Seeds 42, 43, 44 for variance estimation

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project_gemma
export LD_LIBRARY_PATH=/home1/labiosa/miniconda3/envs/my_project/lib:$LD_LIBRARY_PATH
[ -f /home1/labiosa/.cache/huggingface/token ] && export HF_TOKEN=$(cat /home1/labiosa/.cache/huggingface/token)
export PYTHONUNBUFFERED=1
export HF_HUB_DOWNLOAD_TIMEOUT=30
export HF_HUB_ETAG_TIMEOUT=10

SEEDS=(42 43 44)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "=========================================="
echo "Experiment 1: Seed Variance on +1 Vanilla"
echo "Seed: $SEED"
echo "Array Task: $SLURM_ARRAY_TASK_ID"
echo "=========================================="
nvidia-smi

ROOT=/home1/labiosa/NLPSpring2026
PY=/home1/labiosa/miniconda3/envs/my_project_gemma/bin/python
MODEL=Qwen/Qwen2.5-7B-Instruct
SW=$ROOT/stabilizer_weights

OUTPUT_DIR=$SW/qwen_exp1_plus1_seed${SEED}

# Identical hyperparameters to original sweep:
# lr=2e-4, batch_size=4, n_per_condition=150, clean_fraction=0.20, max_steps=1000
# Window L24-27, stab_layer=28 (lora_end+1), lambda_stab=1.0
$PY $ROOT/sweep_lora_window.py \
    --model $MODEL \
    --layer_start 24 \
    --layer_end 27 \
    --n_steps 1000 \
    --lora_rank 4 \
    --lora_alpha 8 \
    --target_modules q_proj v_proj \
    --n_per_condition 150 \
    --clean_fraction 0.20 \
    --batch_size 4 \
    --grad_accum_steps 4 \
    --lr 2e-4 \
    --n_eval 500 \
    --seed $SEED \
    --lambda_stab 1.0 \
    --stab_layer 28 \
    --output_dir $OUTPUT_DIR

echo "--- Seed $SEED complete ---"
