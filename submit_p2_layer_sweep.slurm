#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --output=logs/p2_layer_sweep_%j_%a.out
#SBATCH --error=logs/p2_layer_sweep_%j_%a.err
#SBATCH --array=1-6

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "--- P2: Layer sweep (window ${SLURM_ARRAY_TASK_ID}/6) ---"
nvidia-smi
mkdir -p logs

# Priority 2: Multi-window layer sweep
#
# Train and evaluate LoRA at each of 6 non-overlapping 5-layer windows,
# all else equal to v17. Tests whether the diagnostic-informed placement
# (layers 0–4) is specifically optimal, or whether early layers in general work.
#
# Windows:
#   1: layers  0– 4 (diagnostic-informed, replicates v17)
#   2: layers  5– 9
#   3: layers 10–14
#   4: layers 15–19
#   5: layers 20–24 (uninformed baseline)
#   6: layers 27–31 (final layers)

case $SLURM_ARRAY_TASK_ID in
    1) START=0;  TAG="L00_04" ;;
    2) START=5;  TAG="L05_09" ;;
    3) START=10; TAG="L10_14" ;;
    4) START=15; TAG="L15_19" ;;
    5) START=20; TAG="L20_24" ;;
    6) START=27; TAG="L27_31" ;;
esac

# For window 6 (layers 27–31), only 5 layers
NLAYERS=5

# stab_layer must be downstream of LoRA layers to provide meaningful
# regularization. stab_layer=N hooks model.layers[N-1].
# Phi-3.5 has 32 layers (indices 0–31), so max valid stab_layer = 32.
#
# Strategy: place stab_layer 4 layers downstream of the last LoRA layer.
#   Window 0–4:   stab = 9  (layers[8],  4 downstream of layer 4)
#   Window 5–9:   stab = 14 (layers[13], 4 downstream of layer 9)
#   Window 10–14: stab = 19 (layers[18], 4 downstream of layer 14)
#   Window 15–19: stab = 24 (layers[23], 4 downstream of layer 19)
#   Window 20–24: stab = 29 (layers[28], 4 downstream of layer 24) ← valid, within 0–31
#   Window 27–31: no downstream layer possible → disable L_stab

LAST_LORA=$((START + NLAYERS - 1))
STAB=$((LAST_LORA + 5))

# Disable L_stab if stab_layer would exceed model depth (32 layers)
if [ $STAB -gt 32 ]; then
    LAMBDA_STAB=0.0
    LAMBDA_ACC=1.0
    STAB=9  # placeholder, won't be used with lambda_stab=0
    echo "Window ${TAG}: L_stab disabled (no valid downstream layer)"
else
    LAMBDA_STAB=0.3
    LAMBDA_ACC=0.7
    echo "Window ${TAG}: stab_layer=${STAB} (layers[$((STAB - 1))])"
fi

python train_lrd_lora_v14.py \
    --model    microsoft/Phi-3.5-mini-instruct \
    --dataset  gsm8k \
    --output_dir ./stabilizer_weights/p2_sweep_${TAG} \
    --lora_rank    4 \
    --lora_alpha   8 \
    --lora_dropout 0.05 \
    --lora_layers  $NLAYERS \
    --lora_layers_start $START \
    --target_modules qkv_proj o_proj \
    --lambda_acc   $LAMBDA_ACC \
    --lambda_stab  $LAMBDA_STAB \
    --stab_layer   $STAB \
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
