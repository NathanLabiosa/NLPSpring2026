"""
exp_qwen_heldout_type_train.py — Train Qwen with "speech" held out.

Retrains vanilla LoRA and stabilizer WITHOUT "speech" perturbation in the
training pool. This allows evaluation on speech as a held-out perturbation
type to test generalization vs memorization.

This creates modified versions of the training data pools that exclude speech.

Usage:
    # Train vanilla (no stability loss) without speech
    python exp_qwen_heldout_type_train.py \
        --method vanilla \
        --seed 42 \
        --output_dir ./qwen_vanilla_no_speech_seed42

    # Train stabilizer without speech
    python exp_qwen_heldout_type_train.py \
        --method stabilizer \
        --seed 42 \
        --output_dir ./qwen_stab_no_speech_seed42
"""

import os
import sys
import argparse
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_vanilla_training(seed: int, output_dir: str, dry_run: bool = False):
    """Train vanilla LoRA without speech perturbation."""

    # Create a modified training script that excludes speech
    cmd = [
        sys.executable, os.path.join(ROOT, "sweep_lora_window.py"),
        "--model", "Qwen/Qwen2.5-7B-Instruct",
        "--layer_start", "24",
        "--layer_end", "27",
        "--n_steps", "1000",
        "--lora_rank", "4",
        "--lora_alpha", "8",
        "--target_modules", "q_proj", "v_proj",
        "--output_dir", output_dir,
        "--n_eval", "500",
        "--n_per_condition", "150",
        "--clean_fraction", "0.20",
        "--batch_size", "4",
        "--eval_batch_size", "8",
        "--grad_accum_steps", "4",
        "--lr", "2e-4",
        "--max_seq_len", "512",
        "--seed", str(seed),
        "--heldout_perturbation", "speech",  # New flag to hold out
    ]

    print(f"\n{'='*60}")
    print(f"Training VANILLA (no speech) with seed={seed}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")

    if dry_run:
        print("DRY RUN - would execute:")
        print(" ".join(cmd))
        return 0

    result = subprocess.run(cmd, cwd=ROOT)
    return result.returncode


def run_stabilizer_training(seed: int, output_dir: str, dry_run: bool = False):
    """Train stabilizer LoRA without speech perturbation."""

    cmd = [
        sys.executable, os.path.join(ROOT, "train_lrd_lora_v14.py"),
        "--model", "Qwen/Qwen2.5-7B-Instruct",
        "--dataset", "gsm8k",
        "--lora_rank", "4",
        "--lora_alpha", "8",
        "--lora_dropout", "0.05",
        "--lora_layers_start", "24",
        "--lora_layers", "4",
        "--stab_layer", "27",
        "--target_modules", "q_proj", "v_proj",
        "--lambda_acc", "0.7",
        "--lambda_stab", "3.0",
        "--stab_cosine",
        "--n_per_condition", "857",
        "--clean_fraction", "0.25",
        "--epochs", "30",
        "--max_steps", "1000",
        "--batch_size", "1",
        "--grad_accum_steps", "4",
        "--lr", "2e-5",
        "--warmup_ratio", "0.05",
        "--max_seq_len", "512",
        "--log_every", "50",
        "--save_every", "1000",
        "--eval_every", "0",
        "--clean_eval_every", "200",
        "--early_stop_delta", "15.0",
        "--clean_eval_n", "200",
        "--eval_n_samples", "500",
        "--seed", str(seed),
        "--output_dir", output_dir,
        "--heldout_perturbation", "speech",  # New flag to hold out
    ]

    print(f"\n{'='*60}")
    print(f"Training STABILIZER (no speech) with seed={seed}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")

    if dry_run:
        print("DRY RUN - would execute:")
        print(" ".join(cmd))
        return 0

    result = subprocess.run(cmd, cwd=ROOT)
    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True,
                        choices=["vanilla", "stabilizer"],
                        help="Training method")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.method == "vanilla":
        return run_vanilla_training(args.seed, args.output_dir, args.dry_run)
    else:
        return run_stabilizer_training(args.seed, args.output_dir, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
