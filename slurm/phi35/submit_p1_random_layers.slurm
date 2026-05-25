#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --output=logs/p1_random_layers_%j.out
#SBATCH --error=logs/p1_random_layers_%j.err
#SBATCH --array=1-3

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "--- P1: Random-layer LoRA baseline (seed ${SLURM_ARRAY_TASK_ID}) ---"
nvidia-smi
mkdir -p logs

# Priority 1, Experiment 2: Random-layer LoRA baseline (3 seeds)
#
# Same budget as v17 (rank 4, alpha 8, 5 layers, 300 steps),
# but LoRA placed on 5 randomly sampled layers that are NOT
# in the causal window (0–4) or the uninformed window (20–24).
# Tests whether *any* 5 layers work, or only the diagnostic-informed ones.
#
# Three stratified-random layer sets (avoiding 0–4 and 20–24).
# Available pool: {5..19, 25..31} = 22 layers. Divided into 5 bands:
#   Band A (5–8), Band B (9–13), Band C (14–18), Band D (19,25–27), Band E (28–31)
# Each seed picks one layer per band. No two seeds share any layer.
#
#   Seed 1: 5,  9, 14, 25, 28  (low end of each band)
#   Seed 2: 7, 12, 16, 26, 30  (mid of each band)
#   Seed 3: 8, 11, 18, 27, 31  (high end of each band)
#
# All 15 selected layers are unique across the 3 seeds.
#
# L_stab disabled (lambda_stab=0): non-contiguous layers spanning
# the full model have no valid downstream stab_layer. Since the
# layer sweep (P2) tests all contiguous windows with L_stab, this
# isolates the effect of placement alone.

if [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then
    LAYERS="5 9 14 25 28"
    SEED_TAG="seed1"
elif [ "$SLURM_ARRAY_TASK_ID" -eq 2 ]; then
    LAYERS="7 12 16 26 30"
    SEED_TAG="seed2"
else
    LAYERS="8 11 18 27 31"
    SEED_TAG="seed3"
fi

python src/training/train_lrd_lora_v14.py \
    --model    microsoft/Phi-3.5-mini-instruct \
    --dataset  gsm8k \
    --output_dir ./stabilizer_weights/p1_random_layers_${SEED_TAG} \
    --lora_rank    4 \
    --lora_alpha   8 \
    --lora_dropout 0.05 \
    --lora_layer_indices $LAYERS \
    --target_modules qkv_proj o_proj \
    --lambda_acc   1.0 \
    --lambda_stab  0.0 \
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
