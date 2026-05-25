"""
exp_qwen_lambda0_ablation.py — Clean λ_stab=0 ablation for Qwen2.5-7B.

Trains Qwen2.5-7B with LoRA at L24-27, using the exact same training
infrastructure as the stabilizer (train_lrd_lora_v14.py) but with λ_stab=0.

This is NOT the same as the vanilla sweep (sweep_lora_window.py) which:
  - Uses different training code
  - Has stab_layer offset hack that puts stab outside model

This ablation:
  - Uses train_lrd_lora_v14.py
  - Explicitly sets lambda_stab=0
  - Runs 3 seeds for variance estimation

Usage:
    python exp_qwen_lambda0_ablation.py --seed 42 --output_dir ./qwen_lambda0_seed42
"""

import os
import sys
import argparse
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_training(seed: int, output_dir: str, dry_run: bool = False):
    """Run training with lambda_stab=0."""

    cmd = [
        sys.executable, os.path.join(ROOT, "train_lrd_lora_v14.py"),
        "--model", "Qwen/Qwen2.5-7B-Instruct",
        "--dataset", "gsm8k",
        "--lora_rank", "4",
        "--lora_alpha", "8",
        "--lora_dropout", "0.05",
        "--lora_layers_start", "24",
        "--lora_layers", "4",
        "--stab_layer", "28",  # Last layer (Qwen has 28 layers, indices 0-27)
        "--target_modules", "q_proj", "v_proj",
        "--lambda_acc", "1.0",  # All weight on accuracy
        "--lambda_stab", "0.0",  # Explicit zero
        "--n_per_condition", "857",  # Same as stabilizer
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
    ]

    print(f"\n{'='*60}")
    print(f"Running λ_stab=0 ablation with seed={seed}")
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
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--dry_run", action="store_true", help="Print command without running")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    return run_training(args.seed, args.output_dir, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
