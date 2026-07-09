# === lib path shim (added by repo reorg) ===
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir, 'lib'))
# === end lib path shim ===

"""
Re-evaluate a LoRA checkpoint and save per-sample correct/incorrect labels,
then compute paired bootstrap CIs on the adapter delta.

NOTE (Phase 0 harness audit): The canonical accuracy path is
``eval_fixed_harness.py`` (max_new_tokens=512). This script is retained for
paired-bootstrap analyses only; max_new_tokens has been bumped to 512 to
match the canonical harness. Do not use the absolute accuracy numbers from
here in the paper without cross-checking against the fixed-harness eval.

Usage:
  python eval_paired_bootstrap.py \
      --checkpoint_dir stabilizer_weights/cosv2_L15_stab24_high/lora_final \
      --output_dir     results/bootstrap/cosv2_L15_stab24_high \
      --n_samples 500 --n_bootstrap 10000
"""

import argparse, json, os, random, sys
import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Reuse helpers from training script ──────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_lrd_stabilizer import (
    _extract_gsm8k_answer, _extract_gsm8k_truth, _gsm8k_correct,
    format_question_prompt, EVAL_CONDITIONS,
    _import_perturbation_engine,
)
PerturbationEngine = _import_perturbation_engine()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="microsoft/Phi-3.5-mini-instruct")
    p.add_argument("--checkpoint_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--n_samples", type=int, default=500)
    p.add_argument("--n_bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--uniform_rate", type=float, default=None,
                   help="Override all perturbation rates to this value (matched-rate eval)")
    return p.parse_args()


def run_generation(model, tokenizer, prompts, use_adapter, device):
    gen_kwargs = dict(max_new_tokens=512, do_sample=False,
                      pad_token_id=tokenizer.pad_token_id)
    outputs = []
    for i in range(0, len(prompts), 4):
        batch = prompts[i:i+4]
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


def paired_bootstrap_ci(correct_no_adapter, correct_with_adapter, n_bootstrap=10000,
                         ci_level=0.95):
    """Bootstrap CI on delta = mean(with) - mean(no), paired by sample index."""
    n = len(correct_no_adapter)
    rng = np.random.RandomState(42)
    deltas = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        no_boot  = np.array(correct_no_adapter)[idx].mean()
        with_boot = np.array(correct_with_adapter)[idx].mean()
        deltas.append(with_boot - no_boot)
    deltas = np.sort(deltas)
    alpha = 1 - ci_level
    lo = np.percentile(deltas, 100 * alpha / 2)
    hi = np.percentile(deltas, 100 * (1 - alpha / 2))
    point = np.mean(correct_with_adapter) - np.mean(correct_no_adapter)
    return point, lo, hi


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map=device,
        trust_remote_code=True)

    print(f"Loading adapter: {args.checkpoint_dir}")
    model = PeftModel.from_pretrained(base_model, args.checkpoint_dir)
    model.eval()

    perturber = PerturbationEngine()

    from datasets import load_dataset
    test_ds = load_dataset("openai/gsm8k", "main", split="test")
    test_items = list(test_ds)
    random.seed(args.seed)
    random.shuffle(test_items)
    test_items = test_items[:args.n_samples]

    all_results = {}

    if args.uniform_rate is not None:
        eval_conds = [("none", 0.0)]
        seen = set()
        for m, _ in EVAL_CONDITIONS:
            if m != "none" and m not in seen:
                eval_conds.append((m, args.uniform_rate))
                seen.add(m)
        print(f"Matched-rate eval conditions: {eval_conds}")
    else:
        eval_conds = EVAL_CONDITIONS

    for method, rate in eval_conds:
        cond_name = f"{method}_{int(rate*100)}pct" if rate > 0 else "clean_baseline"
        print(f"\n--- {cond_name} (n={args.n_samples}) ---")

        prompts_perturbed, truths = [], []
        for item in test_items:
            q = item["question"]
            q_noisy = perturber.apply(q, method, rate) if rate > 0 else q
            truths.append(_extract_gsm8k_truth(item["answer"]))
            prompts_perturbed.append(format_question_prompt(tokenizer, q_noisy))

        preds_no = run_generation(model, tokenizer, prompts_perturbed, False, device)
        preds_with = run_generation(model, tokenizer, prompts_perturbed, True, device)

        correct_no = [int(_gsm8k_correct(_extract_gsm8k_answer(p), t))
                      for p, t in zip(preds_no, truths)]
        correct_with = [int(_gsm8k_correct(_extract_gsm8k_answer(p), t))
                        for p, t in zip(preds_with, truths)]

        acc_no = np.mean(correct_no) * 100
        acc_with = np.mean(correct_with) * 100
        delta, lo, hi = paired_bootstrap_ci(correct_no, correct_with, args.n_bootstrap)

        print(f"  No adapter:   {acc_no:.1f}%")
        print(f"  With adapter: {acc_with:.1f}%  (Δ = {delta*100:+.1f}%)")
        print(f"  95% paired bootstrap CI: [{lo*100:+.1f}%, {hi*100:+.1f}%]")

        all_results[cond_name] = {
            "acc_no_adapter": round(acc_no, 2),
            "acc_with_adapter": round(acc_with, 2),
            "delta_pct": round(delta * 100, 2),
            "ci_lo_pct": round(lo * 100, 2),
            "ci_hi_pct": round(hi * 100, 2),
            "n_samples": args.n_samples,
            "n_bootstrap": args.n_bootstrap,
            "per_sample_no_adapter": correct_no,
            "per_sample_with_adapter": correct_with,
        }

    # Summary table
    print("\n" + "=" * 75)
    print(f"{'Condition':<18} {'No Adp':>8} {'W/ Adp':>8} {'Delta':>8} {'95% CI (paired)':>22}")
    print("-" * 75)
    perturbed_correct_no, perturbed_correct_with = [], []
    for cond_name, r in all_results.items():
        print(f"{cond_name:<18} {r['acc_no_adapter']:>7.1f}% {r['acc_with_adapter']:>7.1f}% "
              f"{r['delta_pct']:>+7.1f}%  [{r['ci_lo_pct']:>+6.1f}%, {r['ci_hi_pct']:>+6.1f}%]")
        if cond_name != "clean_baseline":
            perturbed_correct_no.extend(r["per_sample_no_adapter"])
            perturbed_correct_with.extend(r["per_sample_with_adapter"])

    # Pooled perturbed bootstrap
    delta, lo, hi = paired_bootstrap_ci(perturbed_correct_no, perturbed_correct_with,
                                         args.n_bootstrap)
    print("-" * 75)
    print(f"{'Avg perturbed':<18} {'':>8} {'':>8} {delta*100:>+7.1f}%  [{lo*100:>+6.1f}%, {hi*100:>+6.1f}%]")

    all_results["_pooled_perturbed"] = {
        "delta_pct": round(delta * 100, 2),
        "ci_lo_pct": round(lo * 100, 2),
        "ci_hi_pct": round(hi * 100, 2),
        "n_samples_total": len(perturbed_correct_no),
    }

    out_path = os.path.join(args.output_dir, "paired_bootstrap_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
