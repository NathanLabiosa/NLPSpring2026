"""
exp_qwen_heldout_eval.py — Held-out generalization evaluation for Qwen2.5-7B.

Two held-out conditions to test memorization vs generalization:

1. Held-out severity: Evaluate existing checkpoints on perturbation rates
   NOT seen in training:
   - Training has typos @ 5%, evaluate at 10%
   - Training has OCR @ 5% and 15%, evaluate at 10%

2. Held-out perturbation type: After retraining without "speech" in the
   training pool, evaluate on speech perturbation.

Usage:
    # Evaluate held-out severity on existing checkpoints
    python exp_qwen_heldout_eval.py \
        --checkpoint ./stabilizer_weights/qwen_sweep_L24_27/lora_final \
        --mode severity \
        --output_file ./qwen_vanilla_heldout_severity.json

    # Evaluate held-out type (requires retrained checkpoint)
    python exp_qwen_heldout_eval.py \
        --checkpoint ./stabilizer_weights/qwen_vanilla_no_speech/lora_final \
        --mode type \
        --output_file ./qwen_vanilla_heldout_type.json
"""

import os
import sys
import json
import random
import argparse
from typing import List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from train_lrd_stabilizer import (
    format_question_prompt,
    _extract_gsm8k_answer, _extract_gsm8k_truth, _gsm8k_correct,
)

try:
    from perturbations import PerturbationEngine
except ImportError:
    sys.path.insert(0, os.path.join(ROOT, "Phi3.5"))
    from perturbations import PerturbationEngine


# Held-out severity conditions (rates not in training)
HELDOUT_SEVERITY_CONDITIONS = [
    ("typos", 0.10),      # Training has 5%, test at 10%
    ("typos", 0.15),      # Training has 5%, test at 15%
    ("ocr", 0.10),        # Training has 5% and 15%, test at 10% (interpolation)
]

# Held-out type condition
HELDOUT_TYPE_CONDITIONS = [
    ("speech", 0.10),     # Speech held out from training
    ("speech", 0.20),     # Higher rate for robustness check
]

# In-distribution conditions for comparison
INDIST_CONDITIONS = [
    ("typos", 0.05),
    ("ocr", 0.05),
    ("homophones", 0.20),
    ("whitespace", 0.10),
    ("case", 0.10),
]


@torch.no_grad()
def evaluate_conditions(
    model,
    tokenizer,
    perturber: PerturbationEngine,
    conditions: List[Tuple[str, float]],
    n_samples: int,
    device: torch.device,
    batch_size: int = 8,
) -> dict:
    """Evaluate model on specified perturbation conditions."""

    model.eval()

    # Load test data
    test_ds = load_dataset("openai/gsm8k", "main", split="test")
    items = list(test_ds)
    random.shuffle(items)
    items = items[:n_samples]
    truths = [_extract_gsm8k_truth(it["answer"]) for it in items]

    gen_kw = dict(
        max_new_tokens=512, do_sample=False,
        temperature=None, top_p=None, top_k=None,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True
    )

    def gen_batch(prompts: List[str], use_adapter: bool) -> List[str]:
        out = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=1024).to(device)
            if use_adapter:
                ids = model.generate(**enc, **gen_kw)
            else:
                with model.disable_adapter():
                    ids = model.generate(**enc, **gen_kw)
            new_ids = ids[:, enc["input_ids"].shape[1]:]
            out.extend(tokenizer.batch_decode(new_ids, skip_special_tokens=True))
        return out

    results = {}

    for method, rate in conditions:
        cond = f"{method}_{int(rate * 100)}pct"
        print(f"\nEvaluating: {cond}")
        sys.stdout.flush()

        prompts_p = [
            format_question_prompt(tokenizer, perturber.apply(it["question"], method, rate))
            for it in items
        ]
        prompts_c = [format_question_prompt(tokenizer, it["question"]) for it in items]

        # Pass 1: No adapter on perturbed
        print("  [1/3] No adapter on perturbed...")
        preds_no = gen_batch(prompts_p, use_adapter=False)

        # Pass 2: With adapter on perturbed
        print("  [2/3] With adapter on perturbed...")
        preds_yes = gen_batch(prompts_p, use_adapter=True)

        # Pass 3: No adapter on clean (baseline)
        print("  [3/3] No adapter on clean...")
        preds_cln = gen_batch(prompts_c, use_adapter=False)

        acc_no = sum(_gsm8k_correct(_extract_gsm8k_answer(p), t)
                     for p, t in zip(preds_no, truths)) / len(truths) * 100
        acc_yes = sum(_gsm8k_correct(_extract_gsm8k_answer(p), t)
                      for p, t in zip(preds_yes, truths)) / len(truths) * 100
        acc_cln = sum(_gsm8k_correct(_extract_gsm8k_answer(p), t)
                      for p, t in zip(preds_cln, truths)) / len(truths) * 100
        delta = acc_yes - acc_no

        print(f"  No adapter:   {acc_no:.1f}%")
        print(f"  With adapter: {acc_yes:.1f}%  (Δ={delta:+.1f}%)")

        results[cond] = {
            "method": method,
            "rate": rate,
            "n_samples": n_samples,
            "acc_no_adapter": round(acc_no, 2),
            "acc_with_adapter": round(acc_yes, 2),
            "acc_clean_baseline": round(acc_cln, 2),
            "delta": round(delta, 2),
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to LoRA checkpoint directory")
    parser.add_argument("--mode", type=str, required=True, choices=["severity", "type", "both"],
                        help="Which held-out conditions to evaluate")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Output JSON file")
    parser.add_argument("--n_samples", type=int, default=500,
                        help="Number of test samples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include_indist", action="store_true",
                        help="Also evaluate in-distribution conditions for comparison")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print(f"\nLoading base model: Qwen/Qwen2.5-7B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    # Load LoRA adapter
    print(f"Loading LoRA adapter: {args.checkpoint}")
    model = PeftModel.from_pretrained(base_model, args.checkpoint)

    perturber = PerturbationEngine()

    # Select conditions based on mode
    conditions = []
    if args.mode in ["severity", "both"]:
        conditions.extend(HELDOUT_SEVERITY_CONDITIONS)
    if args.mode in ["type", "both"]:
        conditions.extend(HELDOUT_TYPE_CONDITIONS)
    if args.include_indist:
        conditions.extend(INDIST_CONDITIONS)

    print(f"\nEvaluating {len(conditions)} conditions:")
    for method, rate in conditions:
        print(f"  - {method} @ {rate*100:.0f}%")

    results = evaluate_conditions(
        model, tokenizer, perturber,
        conditions, args.n_samples, device
    )

    # Add metadata
    output = {
        "checkpoint": args.checkpoint,
        "mode": args.mode,
        "seed": args.seed,
        "n_samples": args.n_samples,
        "results": results,
    }

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {args.output_file}")


if __name__ == "__main__":
    main()
