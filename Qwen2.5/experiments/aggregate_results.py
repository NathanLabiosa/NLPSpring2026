#!/usr/bin/env python3
"""
aggregate_results.py — Aggregate Qwen experiment results into CSV and summary table.

Output format:
- Per-seed CSV: experiment, seed, condition, perturbation_type, clean_delta, perturbed_delta
- Summary table: mean ± std across seeds per experiment
- Flags for divergent seeds (>1.5σ from mean)

Usage:
    python aggregate_results.py
"""

import os
import json
import sys
from collections import defaultdict
import numpy as np

ROOT = "/home1/labiosa/NLPSpring2026"
SW = os.path.join(ROOT, "stabilizer_weights")
RESULTS_DIR = os.path.join(ROOT, "Qwen2.5/experiments/results")

SEEDS = [42, 43, 44]
PERTURBATION_TYPES = ["typos", "ocr", "speech", "homophones", "whitespace", "case"]


def load_eval_results(path):
    """Load eval_results.json and return dict."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def extract_deltas(results):
    """Extract clean and perturbed deltas from results dict."""
    if results is None:
        return None, None

    clean_delta = None
    perturbed_deltas = {}

    for key, val in results.items():
        if key == "clean_baseline":
            clean_delta = val.get("delta", 0)
        elif "method" in val:
            method = val["method"]
            if method != "none":
                perturbed_deltas[method] = val.get("delta", 0)

    return clean_delta, perturbed_deltas


def main():
    print("=" * 70)
    print("QWEN EXPERIMENT RESULTS AGGREGATION")
    print("=" * 70)

    # Experiment 1: Seed variance on +1 vanilla
    print("\n" + "-" * 70)
    print("EXPERIMENT 1: Seed Variance on +1 Vanilla (lambda=1.0, stab_layer=28)")
    print("-" * 70)

    exp1_data = []
    for seed in SEEDS:
        path = os.path.join(SW, f"qwen_exp1_plus1_seed{seed}", "eval_results.json")
        results = load_eval_results(path)
        if results is None:
            print(f"  Seed {seed}: NOT FOUND ({path})")
            continue

        clean_delta, perturbed_deltas = extract_deltas(results)
        avg_perturbed = np.mean([perturbed_deltas.get(p, 0) for p in PERTURBATION_TYPES if p in perturbed_deltas])

        print(f"  Seed {seed}: clean D={clean_delta:+.1f}%, avg perturbed D={avg_perturbed:+.1f}%")
        exp1_data.append({
            "seed": seed,
            "clean_delta": clean_delta,
            "perturbed_deltas": perturbed_deltas,
            "avg_perturbed": avg_perturbed,
        })

    if exp1_data:
        avg_clean = np.mean([d["clean_delta"] for d in exp1_data])
        std_clean = np.std([d["clean_delta"] for d in exp1_data])
        avg_perturbed = np.mean([d["avg_perturbed"] for d in exp1_data])
        std_perturbed = np.std([d["avg_perturbed"] for d in exp1_data])

        print(f"\n  SUMMARY: Clean D = {avg_clean:.1f} +/- {std_clean:.1f}%")
        print(f"           Avg Perturbed D = {avg_perturbed:.1f} +/- {std_perturbed:.1f}%")

        if std_perturbed > 4:
            print(f"  *** FLAG: std > 4% across seeds! ***")

        for d in exp1_data:
            z = abs(d["avg_perturbed"] - avg_perturbed) / (std_perturbed + 1e-8)
            if z > 1.5:
                print(f"  *** DIVERGENT SEED {d['seed']}: {d['avg_perturbed']:.1f}% (z={z:.2f}) ***")

    # Experiment 2: lambda=0 ablation
    print("\n" + "-" * 70)
    print("EXPERIMENT 2: lambda=0 Ablation (CE-only)")
    print("-" * 70)

    exp2_data = []
    for seed in SEEDS:
        path = os.path.join(SW, f"qwen_exp2_lambda0_seed{seed}", "eval_results.json")
        results = load_eval_results(path)
        if results is None:
            print(f"  Seed {seed}: NOT FOUND ({path})")
            continue

        clean_delta, perturbed_deltas = extract_deltas(results)
        avg_perturbed = np.mean([perturbed_deltas.get(p, 0) for p in PERTURBATION_TYPES if p in perturbed_deltas])

        print(f"  Seed {seed}: clean D={clean_delta:+.1f}%, avg perturbed D={avg_perturbed:+.1f}%")
        exp2_data.append({
            "seed": seed,
            "clean_delta": clean_delta,
            "perturbed_deltas": perturbed_deltas,
            "avg_perturbed": avg_perturbed,
        })

    if exp2_data:
        avg_clean = np.mean([d["clean_delta"] for d in exp2_data])
        std_clean = np.std([d["clean_delta"] for d in exp2_data])
        avg_perturbed = np.mean([d["avg_perturbed"] for d in exp2_data])
        std_perturbed = np.std([d["avg_perturbed"] for d in exp2_data])

        print(f"\n  SUMMARY: Clean D = {avg_clean:.1f} +/- {std_clean:.1f}%")
        print(f"           Avg Perturbed D = {avg_perturbed:.1f} +/- {std_perturbed:.1f}%")

        if std_perturbed > 4:
            print(f"  *** FLAG: std > 4% across seeds! ***")

    # Comparison
    if exp1_data and exp2_data:
        exp1_mean = np.mean([d["avg_perturbed"] for d in exp1_data])
        exp2_mean = np.mean([d["avg_perturbed"] for d in exp2_data])
        diff = exp1_mean - exp2_mean

        print("\n" + "-" * 70)
        print("COMPARISON: Exp 1 (+1) vs Exp 2 (lambda=0)")
        print("-" * 70)
        print(f"  Exp 1 avg perturbed D: {exp1_mean:.1f}%")
        print(f"  Exp 2 avg perturbed D: {exp2_mean:.1f}%")
        print(f"  Difference: {diff:+.1f}%")

        if abs(diff) >= 5:
            print(f"  *** MEANINGFUL DIFFERENCE (>=5pp) ***")

    # Experiment 3
    print("\n" + "-" * 70)
    print("EXPERIMENT 3: Vanilla Held-out Evaluation")
    print("-" * 70)

    sev_path = os.path.join(RESULTS_DIR, "exp3_vanilla_heldout_severity.json")
    if os.path.exists(sev_path):
        with open(sev_path) as f:
            sev_data = json.load(f)
        print("\n  Held-out Severity:")
        for key, val in sev_data.get("results", {}).items():
            print(f"    {key}: D={val.get('delta', 0):+.1f}%")
    else:
        print(f"  Severity results: NOT FOUND")

    type_path = os.path.join(RESULTS_DIR, "exp3_vanilla_heldout_type.json")
    if os.path.exists(type_path):
        with open(type_path) as f:
            type_data = json.load(f)
        print("\n  Held-out Type (speech):")
        for key, val in type_data.get("results", {}).items():
            print(f"    {key}: D={val.get('delta', 0):+.1f}%")
    else:
        print(f"  Type results: NOT FOUND")

    # Original comparison
    print("\n" + "-" * 70)
    print("COMPARISON TO ORIGINAL +11.6%")
    print("-" * 70)

    orig_path = os.path.join(SW, "qwen_sweep_L24_27", "eval_results.json")
    if os.path.exists(orig_path):
        orig_results = load_eval_results(orig_path)
        _, orig_perturbed = extract_deltas(orig_results)
        orig_avg = np.mean([orig_perturbed.get(p, 0) for p in PERTURBATION_TYPES if p in orig_perturbed])
        print(f"  Original (seed 42): avg perturbed D = {orig_avg:.1f}%")

        if exp2_data:
            exp2_seed42 = next((d for d in exp2_data if d["seed"] == 42), None)
            if exp2_seed42:
                new_avg = exp2_seed42["avg_perturbed"]
                print(f"  New lambda=0 seed 42: avg perturbed D = {new_avg:.1f}%")
                print(f"  Difference: {new_avg - orig_avg:+.1f}%")
    else:
        print(f"  Original results: NOT FOUND")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
