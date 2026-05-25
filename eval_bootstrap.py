"""
eval_bootstrap.py — Paired bootstrap confidence intervals for v17 LoRA eval.

Reruns the full evaluation on the v17 checkpoint, saving per-example
(correct_no_adapter, correct_with_adapter) pairs, then runs 10,000 bootstrap
resamples to compute 95% CIs on each Δ.

NOTE (Phase 0 harness audit): The canonical accuracy path is
``eval_fixed_harness.py`` (max_new_tokens=512). This script is retained for
paired-bootstrap analyses only; max_new_tokens has been bumped to 512.

Usage:
    python eval_bootstrap.py \
        --checkpoint_dir ./stabilizer_weights/lora_v17/lora_final \
        --n_samples 500 \
        --n_bootstrap 10000 \
        --output eval_bootstrap_results.json
"""

import os, sys, json, random, argparse
import numpy as np
import torch
from typing import List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_lrd_stabilizer import (
    EVAL_CONDITIONS, KNOWN_BASELINES,
    GSM8K_SYSTEM_PROMPT, format_question_prompt,
    _extract_gsm8k_answer, _extract_gsm8k_truth, _gsm8k_correct,
    _import_perturbation_engine,
)
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

PerturbationEngine = _import_perturbation_engine()


def run_generation(model, tokenizer, prompts: List[str], use_adapter: bool,
                   device, gen_kwargs: dict) -> List[str]:
    outputs = []
    for i in range(0, len(prompts), 4):
        batch = prompts[i : i + 4]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=512).to(device)
        if use_adapter:
            out_ids = model.generate(**enc, **gen_kwargs)
        else:
            with model.disable_adapter():
                out_ids = model.generate(**enc, **gen_kwargs)
        new_ids = out_ids[:, enc["input_ids"].shape[1]:]
        outputs.extend(tokenizer.batch_decode(new_ids, skip_special_tokens=True))
    return outputs


@torch.no_grad()
def evaluate_with_per_example(model, tokenizer, perturber, n_samples, device):
    from datasets import load_dataset

    test_ds    = load_dataset("openai/gsm8k", "main", split="test")
    test_items = list(test_ds)
    random.shuffle(test_items)
    test_items = test_items[:n_samples]

    gen_kwargs = dict(max_new_tokens=512, do_sample=False,
                      pad_token_id=tokenizer.pad_token_id)

    model.eval()
    all_results = {}

    for method, rate in EVAL_CONDITIONS:
        cond_name = f"{method}_{int(rate * 100)}pct" if rate > 0 else "clean_baseline"
        print(f"\n--- Condition: {cond_name} (n={n_samples}) ---")

        prompts_perturbed, truths = [], []
        for item in test_items:
            q_clean = item["question"]
            q_noisy = perturber.apply(q_clean, method, rate) if rate > 0 else q_clean
            truths.append(_extract_gsm8k_truth(item["answer"]))
            prompts_perturbed.append(format_question_prompt(tokenizer, q_noisy))

        preds_no   = run_generation(model, tokenizer, prompts_perturbed,
                                    use_adapter=False, device=device, gen_kwargs=gen_kwargs)
        preds_with = run_generation(model, tokenizer, prompts_perturbed,
                                    use_adapter=True,  device=device, gen_kwargs=gen_kwargs)

        correct_no   = [int(_gsm8k_correct(_extract_gsm8k_answer(p), t))
                        for p, t in zip(preds_no, truths)]
        correct_with = [int(_gsm8k_correct(_extract_gsm8k_answer(p), t))
                        for p, t in zip(preds_with, truths)]

        acc_no   = sum(correct_no)   / n_samples * 100
        acc_with = sum(correct_with) / n_samples * 100
        delta    = acc_with - acc_no
        known    = KNOWN_BASELINES.get(method)
        known_str = f"  (known LRD baseline: {known:.1f}%)" if known else ""

        print(f"  No adapter:   {acc_no:.1f}%{known_str}")
        print(f"  With adapter: {acc_with:.1f}%  (Δ = {delta:+.1f}%)")

        all_results[cond_name] = {
            "correct_no_adapter":   correct_no,
            "correct_with_adapter": correct_with,
            "acc_no_adapter":       round(acc_no, 2),
            "acc_with_adapter":     round(acc_with, 2),
            "delta":                round(delta, 2),
            "known_baseline":       known,
        }

    return all_results


def bootstrap_ci(correct_no, correct_with, n_bootstrap=10000, ci=0.95):
    """Paired bootstrap CI on Δ = mean(correct_with) - mean(correct_no)."""
    rng = np.random.default_rng(42)
    n = len(correct_no)
    pairs = np.stack([correct_no, correct_with], axis=1)  # (n, 2)
    obs_delta = (pairs[:, 1].mean() - pairs[:, 0].mean()) * 100

    boot_deltas = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        sample = pairs[idx]
        boot_deltas[i] = (sample[:, 1].mean() - sample[:, 0].mean()) * 100

    alpha = 1 - ci
    lo = np.percentile(boot_deltas, alpha / 2 * 100)
    hi = np.percentile(boot_deltas, (1 - alpha / 2) * 100)
    p_value = min(
        np.mean(boot_deltas <= 0),
        np.mean(boot_deltas >= 0),
    ) * 2  # two-sided
    return obs_delta, lo, hi, p_value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str,
                        default="./stabilizer_weights/lora_v17/lora_final")
    parser.add_argument("--base_model", type=str,
                        default="microsoft/Phi-3.5-mini-instruct")
    parser.add_argument("--n_samples",   type=int, default=500)
    parser.add_argument("--n_bootstrap", type=int, default=10000)
    parser.add_argument("--output",      type=str,
                        default="./stabilizer_weights/lora_v17/bootstrap_results.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"\nLoading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map={"": 0},
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    print(f"Loading LoRA adapter: {args.checkpoint_dir}")
    model = PeftModel.from_pretrained(base_model, args.checkpoint_dir)
    model.eval()

    perturber = PerturbationEngine()

    print(f"\nRunning eval (n={args.n_samples}) to collect per-example predictions...")
    per_example = evaluate_with_per_example(
        model, tokenizer, perturber, args.n_samples, device
    )

    print(f"\n{'=' * 65}")
    print(f"Bootstrap CIs  (n={args.n_samples}, {args.n_bootstrap:,} resamples, 95% CI)")
    print(f"{'=' * 65}")
    print(f"{'Condition':<22} {'Δ':>6}  {'95% CI':^18}  {'p-value':>8}")
    print("-" * 65)

    output = {}
    for cond_name, res in per_example.items():
        cn  = np.array(res["correct_no_adapter"])
        cw  = np.array(res["correct_with_adapter"])
        obs, lo, hi, pval = bootstrap_ci(cn, cw, n_bootstrap=args.n_bootstrap)
        sig = "*" if pval < 0.05 else ("~" if pval < 0.10 else "")
        print(f"{cond_name:<22} {obs:>+5.1f}%  [{lo:>+5.1f}%, {hi:>+5.1f}%]  {pval:>8.4f} {sig}")
        output[cond_name] = {
            **res,
            "bootstrap_delta":  round(obs, 3),
            "ci_lo":            round(lo, 3),
            "ci_hi":            round(hi, 3),
            "p_value":          round(pval, 4),
        }
        # remove bulky per-example lists from JSON (keep counts)
        output[cond_name]["n_correct_no_adapter"]   = int(cn.sum())
        output[cond_name]["n_correct_with_adapter"] = int(cw.sum())
        del output[cond_name]["correct_no_adapter"]
        del output[cond_name]["correct_with_adapter"]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved: {args.output}")


if __name__ == "__main__":
    main()
