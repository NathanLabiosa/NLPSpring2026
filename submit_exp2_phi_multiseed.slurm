#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --array=1-30
#SBATCH --output=logs/exp2_phi_multiseed_%a_%j.out
#SBATCH --error=logs/exp2_phi_multiseed_%a_%j.err

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs

# Exp 2: Multi-seed Phi-3.5 layer sweep (5 seeds x 6 windows = 30 tasks).
# Array task layout: tasks 1-6 = seed 42, tasks 7-12 = seed 123, etc.
# Window within each seed block: 1=L00-04, 2=L05-09, ..., 6=L27-31.
# Success criterion: L15-19 CI non-overlapping with at least early windows.

SEEDS=(42 123 456 789 1234)
WINDOW_STARTS=(0 5 10 15 20 27)
WINDOW_TAGS=("L00_04" "L05_09" "L10_14" "L15_19" "L20_24" "L27_31")
STAB_LAYERS=(9 14 19 24 29 31)

# task_id 1-30: seed_idx = (task-1)/6, window_idx = (task-1)%6
SEED_IDX=$(( (SLURM_ARRAY_TASK_ID - 1) / 6 ))
WIN_IDX=$(( (SLURM_ARRAY_TASK_ID - 1) % 6 ))

SEED=${SEEDS[$SEED_IDX]}
START=${WINDOW_STARTS[$WIN_IDX]}
TAG=${WINDOW_TAGS[$WIN_IDX]}
STAB=${STAB_LAYERS[$WIN_IDX]}

echo "--- Exp2: Phi-3.5, seed=${SEED}, window=${TAG}, stab_layer=${STAB} ---"
nvidia-smi

# For the last window (L27-31), disable L_stab (no valid downstream layer)
if [ "$WIN_IDX" -eq 5 ]; then
    LAMBDA_STAB=0.0
    LAMBDA_ACC=1.0
else
    LAMBDA_STAB=3.0
    LAMBDA_ACC=0.7
fi

python train_lrd_lora_v14.py \
    --model microsoft/Phi-3.5-mini-instruct \
    --dataset gsm8k \
    --lora_rank 4 \
    --lora_alpha 8 \
    --lora_dropout 0.05 \
    --lora_layers 5 \
    --lora_layers_start ${START} \
    --stab_layer ${STAB} \
    --target_modules qkv_proj o_proj \
    --lambda_acc ${LAMBDA_ACC} \
    --lambda_stab ${LAMBDA_STAB} \
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
    --seed ${SEED} \
    --output_dir ./stabilizer_weights/exp2_phi_seed${SEED}_${TAG}
