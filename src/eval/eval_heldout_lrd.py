# === lib path shim (added by repo reorg) ===
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir, 'lib'))
# === end lib path shim ===

"""
eval_heldout_lrd.py — LRD profiling under perturbation types HELD OUT of the
main analysis. Phase 2.4 of experiment_plan.md.

The main-analysis perturbation set is {typos, ocr, speech, homophones,
whitespace, case}. Held-out set isolates single-op variants:
  - char_insert : pure random-character insertion (no del/sub/swap)
  - char_swap   : pure adjacent-character swap (no del/sub/insert)
  - qwerty      : keyboard-adjacency substitution (existed in engine, not in main set)

For each held-out perturbation, this script computes the per-layer LRD
profile (mean-pooled cosine distance between clean and perturbed hidden
states) and the per-token LRD restricted to differing token positions —
matching Section 5's LRD methodology so the resulting profiles are
directly comparable to the main-analysis figures.

Output JSON schema mirrors expF / lrd_diagnostics so downstream plotting
code can ingest it without changes.

Usage:
    python eval_heldout_lrd.py \\
        --base_model microsoft/Phi-3.5-mini-instruct \\
        --output_file results/heldout_perturbation_lrd/phi35.json \\
        --n_samples 200 \\
        --attn_impl eager
"""

import os
import sys
import json
import random
import argparse
from typing import List, Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "Phi3.5"))

from perturbations import PerturbationEngine
from train_lrd_stabilizer import (
    format_question_prompt, GSM8K_SYSTEM_PROMPT,
    _extract_gsm8k_answer, _extract_gsm8k_truth, _gsm8k_correct,
)


HELDOUT_EXPERIMENTS = [
    {"name": "CharInsert_5%", "type": "char_insert", "rate": 0.05},
    {"name": "CharSwap_5%",   "type": "char_swap",   "rate": 0.05},
    {"name": "Qwerty_5%",     "type": "qwerty",      "rate": 0.05},
]


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    dot = float(np.dot(a, b))
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if norm < 1e-9 else 1.0 - dot / norm


def compute_lrd_profile(clean: np.ndarray, noisy: np.ndarray) -> np.ndarray:
    assert clean.shape == noisy.shape
    return np.array([cosine_distance(clean[l], noisy[l]) for l in range(clean.shape[0])])


def compute_perturbed_token_lrd(
    clean_per_tok: list, noisy_per_tok: list,
    clean_tokens: list, noisy_tokens: list,
) -> Optional[np.ndarray]:
    min_len = min(len(clean_tokens), len(noisy_tokens))
    diff_pos = [i for i in range(min_len) if clean_tokens[i] != noisy_tokens[i]]
    if not diff_pos:
        return None
    n_layers = len(clean_per_tok)
    out = []
    for l in range(n_layers):
        c, n = clean_per_tok[l], noisy_per_tok[l]
        vals = [cosine_distance(c[i], n[i]) for i in diff_pos
                if i < c.shape[0] and i < n.shape[0]]
        out.append(float(np.mean(vals)) if vals else 0.0)
    return np.array(out)


def cascade_slope(profile: np.ndarray) -> float:
    layers = np.arange(len(profile))
    return float(np.polyfit(layers, profile, deg=1)[0])


def recovery_test(profile: np.ndarray) -> dict:
    q = max(1, len(profile) // 4)
    early = float(profile[:q].mean())
    late = float(profile[-q:].mean())
    return {"early_lrd": early, "late_lrd": late, "recovered": late < early * 0.8}


@torch.no_grad()
def extract_hidden_states(model, tokenizer, prompt: str, device, max_len: int = 1024):
    enc = tokenizer(prompt, return_tensors="pt", truncation=True,
                    max_length=max_len, padding=False).to(device)
    out = model(**enc, output_hidden_states=True, return_dict=True)
    hs = out.hidden_states
    pooled = np.stack([h[0].mean(dim=0).float().cpu().numpy() for h in hs], axis=0)
    per_tok = [h[0].float().cpu().numpy() for h in hs]
    token_ids = enc["input_ids"][0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    return pooled, per_tok, tokens


@torch.no_grad()
def greedy_generate(model, tokenizer, prompt: str, device, max_new_tokens: int = 256) -> str:
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        use_cache=True,
    )
    return tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=200,
                        help="Target number of clean-correct examples to profile")
    parser.add_argument("--prefilter_pool", type=int, default=800,
                        help="Max GSM8K test examples to evaluate during pre-filter")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attn_impl", type=str, default="eager",
                        help="Eager is the default for hidden-state extraction (matches models/llama/lrd_diagnostics.py).")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Generation length for the pre-filter and accuracy probe")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:     {device}")
    print(f"Base model: {args.base_model}")
    print(f"attn_impl:  {args.attn_impl}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=args.attn_impl,
    )
    model.eval()

    test_ds = load_dataset("openai/gsm8k", "main", split="test")
    items = list(test_ds)
    random.shuffle(items)
    items = items[:args.prefilter_pool]
    truths = [_extract_gsm8k_truth(it["answer"]) for it in items]

    print(f"\nPre-filtering up to {args.prefilter_pool} GSM8K examples for clean-correct ...")
    clean_prompts = [format_question_prompt(tokenizer, it["question"]) for it in items]

    clean_correct_idx = []
    for i, prompt in enumerate(clean_prompts):
        try:
            gen = greedy_generate(model, tokenizer, prompt, device,
                                  max_new_tokens=args.max_new_tokens)
            if _gsm8k_correct(_extract_gsm8k_answer(gen), truths[i]):
                clean_correct_idx.append(i)
        except Exception as e:
            print(f"  [warn] pre-filter idx {i}: {e}")
        if (i + 1) % 50 == 0:
            print(f"  pre-filter {i + 1}/{len(items)}  clean_correct so far = {len(clean_correct_idx)}")
        if len(clean_correct_idx) >= args.n_samples:
            break

    print(f"  → {len(clean_correct_idx)} clean-correct examples (target {args.n_samples})")
    if len(clean_correct_idx) < 50:
        print("[WARNING] Fewer than 50 clean-correct examples — results will be underpowered.")

    perturber = PerturbationEngine()
    all_results = {}

    for exp in HELDOUT_EXPERIMENTS:
        print(f"\n{'─' * 60}\n{exp['name']}\n{'─' * 60}")
        records = []
        for n, idx in enumerate(clean_correct_idx):
            random.seed(args.seed + idx)
            perturbed_q = perturber.apply(items[idx]["question"], exp["type"], exp["rate"])
            cp = clean_prompts[idx]
            np_ = format_question_prompt(tokenizer, perturbed_q)
            try:
                c_pool, c_tok, c_tkns = extract_hidden_states(model, tokenizer, cp, device)
                n_pool, n_tok, n_tkns = extract_hidden_states(model, tokenizer, np_, device)

                lrd = compute_lrd_profile(c_pool, n_pool)
                per_tok_lrd = compute_perturbed_token_lrd(c_tok, n_tok, c_tkns, n_tkns)

                gen = greedy_generate(model, tokenizer, np_, device,
                                      max_new_tokens=args.max_new_tokens)
                is_correct = _gsm8k_correct(_extract_gsm8k_answer(gen), truths[idx])

                records.append({
                    "index":              idx,
                    "lrd_profile":        lrd.tolist(),
                    "final_lrd":          float(lrd[-1]),
                    "mean_lrd":           float(lrd.mean()),
                    "max_lrd":            float(lrd.max()),
                    "cascade_slope":      cascade_slope(lrd),
                    **{k: v for k, v in recovery_test(lrd).items()},
                    "is_correct":         bool(is_correct),
                    "per_tok_lrd_profile": per_tok_lrd.tolist() if per_tok_lrd is not None else None,
                    "per_tok_final_lrd":   float(per_tok_lrd[-1]) if per_tok_lrd is not None else None,
                    "n_differing_tokens":  sum(1 for a, b in zip(c_tkns, n_tkns) if a != b),
                })
            except Exception as e:
                print(f"  [warn] idx {idx}: {e}")
                continue
            if (n + 1) % 25 == 0:
                print(f"    {n + 1}/{len(clean_correct_idx)}")

        all_results[exp["name"]] = records
        if records:
            acc = float(np.mean([r["is_correct"] for r in records]))
            mlrd = float(np.mean([r["final_lrd"] for r in records]))
            print(f"  n={len(records)}  perturbed_acc={acc:.2%}  mean_final_LRD={mlrd:.4f}")

    # Mean-pooled and per-token LRD profile aggregates (matches expF figure schema).
    aggregates = {}
    for exp_name, records in all_results.items():
        if not records:
            continue
        prof_mat = np.array([r["lrd_profile"] for r in records])
        agg = {
            "n": len(records),
            "mean_lrd_profile": prof_mat.mean(axis=0).tolist(),
            "std_lrd_profile":  prof_mat.std(axis=0).tolist(),
            "mean_final_lrd":   float(prof_mat[:, -1].mean()),
            "perturbed_acc":    float(np.mean([r["is_correct"] for r in records])),
        }
        per_tok_records = [r for r in records if r["per_tok_lrd_profile"] is not None]
        if per_tok_records:
            pt_mat = np.array([r["per_tok_lrd_profile"] for r in per_tok_records])
            agg["mean_per_tok_lrd_profile"] = pt_mat.mean(axis=0).tolist()
        aggregates[exp_name] = agg

    out = {
        "harness_version":      "heldout_lrd_v1",
        "base_model":           args.base_model,
        "seed":                 args.seed,
        "n_samples":            args.n_samples,
        "n_clean_correct":      len(clean_correct_idx),
        "attn_impl":            args.attn_impl,
        "experiments":          [{"name": e["name"], "type": e["type"], "rate": e["rate"]}
                                 for e in HELDOUT_EXPERIMENTS],
        "aggregates":           aggregates,
        "per_example_records":  all_results,
    }

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {args.output_file}")


if __name__ == "__main__":
    main()
