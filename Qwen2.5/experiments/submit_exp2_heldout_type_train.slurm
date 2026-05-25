#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --time=08:00:00
#SBATCH --array=0-1
#SBATCH --output=/home1/labiosa/NLPSpring2026/logs/qwen_nospeech_%a_%j.out
#SBATCH --error=/home1/labiosa/NLPSpring2026/logs/qwen_nospeech_%a_%j.err

# Experiment 2b: Held-out TYPE retraining
# Retrains vanilla and stabilizer WITHOUT "speech" perturbation
# to allow evaluation on speech as held-out type

module purge
source /home1/labiosa/miniconda3/etc/profile.d/conda.sh
conda activate my_project_gemma

export LD_LIBRARY_PATH=/home1/labiosa/miniconda3/envs/my_project/lib:$LD_LIBRARY_PATH
export HF_TOKEN=$(cat /home1/labiosa/.cache/huggingface/token)

# Array: 0 = vanilla, 1 = stabilizer
METHODS=("vanilla" "stabilizer")
METHOD=${METHODS[$SLURM_ARRAY_TASK_ID]}

echo "=========================================="
echo "Experiment 2b: Held-out Type Training"
echo "Method: $METHOD (no speech in training)"
echo "=========================================="
nvidia-smi

ROOT=/home1/labiosa/NLPSpring2026
PY=/home1/labiosa/miniconda3/envs/my_project_gemma/bin/python
MODEL=Qwen/Qwen2.5-7B-Instruct
SW=$ROOT/stabilizer_weights

if [ "$METHOD" == "vanilla" ]; then
    OUTPUT_DIR=$SW/qwen_vanilla_nospeech

    # Vanilla LoRA without speech
    $PY $ROOT/sweep_lora_window.py \
        --model $MODEL \
        --layer_start 24 \
        --layer_end 27 \
        --n_steps 1000 \
        --lora_rank 4 \
        --lora_alpha 8 \
        --target_modules q_proj v_proj \
        --output_dir $OUTPUT_DIR \
        --n_eval 500 \
        --n_per_condition 150 \
        --clean_fraction 0.20 \
        --batch_size 4 \
        --eval_batch_size 8 \
        --grad_accum_steps 4 \
        --lr 2e-4 \
        --max_seq_len 512 \
        --seed 42 \
        --heldout_perturbation speech

else
    OUTPUT_DIR=$SW/qwen_stab_nospeech

    # Stabilizer without speech
    # Use --perturbation_types to include all except speech
    $PY $ROOT/train_lrd_lora_v14.py \
        --model $MODEL \
        --dataset gsm8k \
        --lora_rank 4 \
        --lora_alpha 8 \
        --lora_dropout 0.05 \
        --lora_layers_start 24 \
        --lora_layers 4 \
        --stab_layer 27 \
        --target_modules q_proj v_proj \
        --lambda_acc 0.7 \
        --lambda_stab 3.0 \
        --stab_cosine \
        --perturbation_types typos ocr homophones whitespace case \
        --n_per_condition 857 \
        --clean_fraction 0.25 \
        --epochs 30 \
        --max_steps 1000 \
        --batch_size 1 \
        --grad_accum_steps 4 \
        --lr 2e-5 \
        --warmup_ratio 0.05 \
        --max_seq_len 512 \
        --log_every 50 \
        --save_every 1000 \
        --eval_every 0 \
        --clean_eval_every 200 \
        --early_stop_delta 15.0 \
        --clean_eval_n 200 \
        --eval_n_samples 500 \
        --seed 42 \
        --output_dir $OUTPUT_DIR
fi

echo "--- $METHOD (no speech) training complete ---"
