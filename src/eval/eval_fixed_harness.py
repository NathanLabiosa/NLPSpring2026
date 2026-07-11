# === lib path shim (added by repo reorg) ===
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir, 'lib'))
# === end lib path shim ===

"""
eval_fixed_harness.py — Unified GSM8K evaluation harness for cross-checkpoint
comparison under a single fixed evaluation protocol.

Reuses the exp3 held-out evaluation flow (max_new_tokens=512, chat-template
prompt via train_lrd_stabilizer.format_question_prompt, n_samples=500 from
the GSM8K test split). Generalized to take an arbitrary base model so we can
compare Qwen and Phi-3.5 checkpoints under the same code path.

Evaluates the standard 7 conditions: clean + 6 perturbation types matched to
the original training pool (typos@5%, ocr@5%, speech@10%, homophones@20%,
whitespace@10%, case@10%).

Usage:
    python eval_fixed_harness.py \
        --base_model Qwen/Qwen2.5-7B-Instruct \
        --checkpoint ./stabilizer_weights/qwen_sweep_L24_27/lora_final \
        --output_file ./fixed_harness_qwen_vanilla.json
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

# Compatibility shims for transformers API changes
_compat_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib")
if _compat_dir not in sys.path:
    sys.path.insert(0, _compat_dir)
try:
    import compat  # noqa: F401
except ImportError:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from train_lrd_stabilizer import (
    format_question_prompt,
    _extract_gsm8k_answer, _extract_gsm8k_truth, _gsm8k_correct,
)

# Bump this when the harness changes in a way that invalidates prior results
# (prompt formatting, generation kwargs, condition set, sampling protocol).
# Every eval JSON records it so downstream tooling can detect drift.
HARNESS_VERSION = "v1"

try:
    from perturbations import PerturbationEngine
except ImportError:
    # Search canonical locations across cluster layouts
    for _p in [
        os.path.join(ROOT, "evaluation"),
        os.path.join(ROOT, "models", "phi3.5"),
        os.path.join(ROOT, "Phi3.5"),
    ]:
        if _p not in sys.path:
            sys.path.insert(0, _p)
    from perturbations import PerturbationEngine


# Standard in-distribution condition set (clean + 6 perturbations) at the
# rates the LoRAs were originally trained on.
CONDITIONS: List[Tuple[str, float]] = [
    ("none",       0.0),
    ("typos",      0.05),
    ("ocr",        0.05),
    ("speech",     0.10),
    ("homophones", 0.20),
    ("whitespace", 0.10),
    ("case",       0.10),
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
    max_new_tokens: int = 512,
    has_adapter: bool = True,
) -> dict:
    model.eval()

    test_ds = load_dataset("openai/gsm8k", "main", split="test")
    items = list(test_ds)
    random.shuffle(items)
    items = items[:n_samples]
    truths = [_extract_gsm8k_truth(it["answer"]) for it in items]

    gen_kw = dict(
        max_new_tokens=max_new_tokens, do_sample=False,
        temperature=None, top_p=None, top_k=None,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )

    def gen_batch(prompts: List[str], use_adapter: bool) -> List[str]:
        out = []
        n_batches = (len(prompts) + batch_size - 1) // batch_size
        for i in range(0, len(prompts), batch_size):
            batch_idx = i // batch_size + 1
            if batch_idx % 10 == 0 or batch_idx == 1:
                print(f"    batch {batch_idx}/{n_batches}")
                sys.stdout.flush()
            batch = prompts[i:i + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=1024).to(device)
            if use_adapter or not has_adapter:
                # has_adapter=False: model is the bare base; no toggling needed.
                ids = model.generate(**enc, **gen_kw)
            else:
                with model.disable_adapter():
                    ids = model.generate(**enc, **gen_kw)
            new_ids = ids[:, enc["input_ids"].shape[1]:]
            out.extend(tokenizer.batch_decode(new_ids, skip_special_tokens=True))
        return out

    results = {}

    prompts_c = [format_question_prompt(tokenizer, it["question"]) for it in items]

    for method, rate in conditions:
        cond = "clean_baseline" if method == "none" else f"{method}_{int(rate * 100)}pct"
        print(f"\n--- {cond} ---")
        sys.stdout.flush()

        if method == "none":
            prompts_p = prompts_c
        else:
            prompts_p = [
                format_question_prompt(tokenizer, perturber.apply(it["question"], method, rate))
                for it in items
            ]

        print("  [1/3] no adapter on perturbed")
        preds_no = gen_batch(prompts_p, use_adapter=False)
        if has_adapter:
            print("  [2/3] with adapter on perturbed")
            preds_yes = gen_batch(prompts_p, use_adapter=True)
        else:
            preds_yes = preds_no  # baseline-only audit: adapter columns mirror no-adapter
        if method == "none":
            preds_cln = preds_no
        else:
            print("  [3/3] no adapter on clean")
            preds_cln = gen_batch(prompts_c, use_adapter=False)

        acc_no = sum(_gsm8k_correct(_extract_gsm8k_answer(p), t)
                     for p, t in zip(preds_no, truths)) / len(truths) * 100
        acc_yes = sum(_gsm8k_correct(_extract_gsm8k_answer(p), t)
                      for p, t in zip(preds_yes, truths)) / len(truths) * 100
        acc_cln = sum(_gsm8k_correct(_extract_gsm8k_answer(p), t)
                      for p, t in zip(preds_cln, truths)) / len(truths) * 100
        delta = acc_yes - acc_no

        print(f"  no adapter:   {acc_no:.1f}%")
        if has_adapter:
            print(f"  with adapter: {acc_yes:.1f}%  (Δ={delta:+.1f}%)")

        results[cond] = {
            "method": method,
            "rate": rate,
            "n_samples": n_samples,
            "acc_no_adapter": round(acc_no, 2),
            "acc_with_adapter": round(acc_yes, 2) if has_adapter else None,
            "acc_clean_baseline": round(acc_cln, 2),
            "delta": round(delta, 2) if has_adapter else None,
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True,
                        help="HF model id for the base model")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to LoRA checkpoint directory. Omit to run a "
                             "no-adapter baseline audit on the bare base model.")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attn_impl", type=str, default="sdpa",
                        help="attn_implementation passed to from_pretrained (e.g. sdpa, eager)")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Base model: {args.base_model}")
    print(f"Checkpoint: {args.checkpoint}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
        attn_implementation=args.attn_impl,
    )

    has_adapter = args.checkpoint is not None
    if has_adapter:
        print(f"Loading LoRA adapter")
        model = PeftModel.from_pretrained(base_model, args.checkpoint)
    else:
        print("No checkpoint provided — running no-adapter baseline audit.")
        model = base_model

    perturber = PerturbationEngine()

    print(f"\nEvaluating {len(CONDITIONS)} conditions on {args.n_samples} GSM8K examples")
    for method, rate in CONDITIONS:
        label = "clean" if method == "none" else f"{method} @ {rate*100:.0f}%"
        print(f"  - {label}")

    results = evaluate_conditions(
        model, tokenizer, perturber,
        CONDITIONS, args.n_samples, device,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        has_adapter=has_adapter,
    )

    # Skip None deltas (baseline-audit mode has has_adapter=False → delta=None).
    perturbed_deltas = [r["delta"] for c, r in results.items()
                        if c != "clean_baseline" and r["delta"] is not None]
    mean_perturbed_delta = (sum(perturbed_deltas) / len(perturbed_deltas)
                             if perturbed_deltas else None)
    clean_baseline = results.get("clean_baseline", {}).get("acc_no_adapter")

    output = {
        "harness_version": HARNESS_VERSION,
        "base_model": args.base_model,
        "checkpoint": args.checkpoint,
        "seed": args.seed,
        "n_samples": args.n_samples,
        "max_new_tokens": args.max_new_tokens,
        "harness": "exp3_heldout_style_fixed",
        "clean_baseline": clean_baseline,
        "mean_perturbed_delta": (round(mean_perturbed_delta, 4)
                                 if mean_perturbed_delta is not None else None),
        "results": results,
    }

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {args.output_file}")

    print("\n" + "=" * 60)
    print(f"{'Condition':<22} {'No Adp':>8} {'W/ Adp':>8} {'Delta':>7}")
    print("-" * 60)
    for cond, r in results.items():
        marker = "" if cond == "clean_baseline" else "  *"
        adp_str = f"{r['acc_with_adapter']:>7.1f}%" if r["acc_with_adapter"] is not None else f"{'-':>8}"
        delta_str = f"{r['delta']:>+6.1f}%" if r["delta"] is not None else f"{'-':>7}"
        print(f"{cond:<22} {r['acc_no_adapter']:>7.1f}% {adp_str} {delta_str}{marker}")
    if mean_perturbed_delta is not None:
        print("-" * 60)
        print(f"{'mean perturbed Δ':<22} {'':>8} {'':>8} {mean_perturbed_delta:>+6.1f}%")


if __name__ == "__main__":
    main()
