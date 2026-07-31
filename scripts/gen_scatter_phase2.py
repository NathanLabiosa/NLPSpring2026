#!/usr/bin/env python3
"""
gen_scatter_phase2.py — Generate disruption_<model>_scatter_phase2.pdf figures.

Both axes use phase2_scatter checkpoints:
  x = downstream clean disruption (from expF_clean_disruption_{model}.json)
  y = 3-seed mean acc_delta (from results/fixed_harness/v2/scatter/)

Models: phi35, qwen, llama3, mistral (4 panels).
Includes L10-14 for llama/mistral.
Output: figures/disruption_{model}_scatter_phase2.pdf
"""

import json
import math
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCATTER_DIR = os.path.join(ROOT, "results", "fixed_harness", "v2", "scatter")
RESULTS_DIR = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(ROOT, "figures")

MODELS = {
    "phi35": {
        "slug": "phi35",
        "display": "Phi-3.5-mini",
        "ckpt_prefix": "phase2_scatter_phi35",
        "windows": ["L00-04", "L05-09", "L10-14", "L15-19", "L20-24", "L27-31"],
    },
    "qwen": {
        "slug": "qwen2.5_7b",
        "display": "Qwen2.5-7B",
        "ckpt_prefix": "phase2_scatter_qwen2.5_7b",
        "windows": ["L00-04", "L05-09", "L08-11", "L15-19", "L20-23", "L24-27"],
    },
    "llama3": {
        "slug": "llama3_8b",
        "display": "Llama-3-8B",
        "ckpt_prefix": "phase2_scatter_llama3_8b",
        "windows": ["L00-04", "L05-09", "L10-14", "L15-19", "L20-24", "L27-31"],
    },
    "mistral": {
        "slug": "mistral_7b_v03",
        "display": "Mistral-7B-v0.3",
        "ckpt_prefix": "phase2_scatter_mistral_7b_v03",
        "windows": ["L00-04", "L05-09", "L10-14", "L15-19", "L20-24", "L27-31"],
    },
}
SEEDS = [42, 43, 44]


def load_per_seed_deltas(slug, windows):
    """Return {window: {seed: delta}} from v2/scatter JSON files."""
    by_win = {}
    for win in windows:
        by_win[win] = {}
        for seed in SEEDS:
            fp = os.path.join(SCATTER_DIR, f"{slug}_{win}_seed{seed}.json")
            if os.path.exists(fp):
                d = json.load(open(fp))
                by_win[win][seed] = d.get("mean_perturbed_delta")
    return by_win


def load_disruption(model_key, windows, ckpt_prefix):
    """Return {window: downstream_disruption} from disruption JSON."""
    cache_path = os.path.join(RESULTS_DIR, f"expF_clean_disruption_{model_key}.json")
    cache = json.load(open(cache_path))
    out = {}
    for win in windows:
        key = f"{ckpt_prefix}_{win}_seed42"
        if key in cache:
            out[win] = cache[key].get("downstream_disruption")
    return out


def mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    n = len(vals)
    m = sum(vals) / n
    if n == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in vals) / (n - 1)
    return m, math.sqrt(var)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    print(f"{'Model':<10} {'Window':<8} {'Disruption':>12} {'Mean Δ (pp)':>12} "
          f"{'s42':>8} {'s43':>8} {'s44':>8} {'±std':>8}")
    print("-" * 78)

    all_rho = {}

    for model_key, cfg in MODELS.items():
        slug = cfg["slug"]
        windows = cfg["windows"]
        ckpt_prefix = cfg["ckpt_prefix"]

        per_seed = load_per_seed_deltas(slug, windows)
        disruption = load_disruption(model_key, windows, ckpt_prefix)

        x_vals, y_vals = [], []
        rows = []

        for win in windows:
            disrupt = disruption.get(win)
            seed_deltas = per_seed.get(win, {})
            vals = [seed_deltas.get(s) for s in SEEDS]
            mn, sd = mean_std(vals)
            if disrupt is not None and mn is not None:
                x_vals.append(disrupt)
                y_vals.append(mn)
            rows.append((win, disrupt, mn, sd, vals))

        # Spearman ρ
        if len(x_vals) >= 3:
            rho, pval = spearmanr(x_vals, y_vals)
        else:
            rho, pval = float("nan"), float("nan")
        all_rho[model_key] = (rho, pval)

        # Print table
        for win, disrupt, mn, sd, vals in rows:
            s42 = f"{vals[0]:+.2f}" if vals[0] is not None else "   —"
            s43 = f"{vals[1]:+.2f}" if vals[1] is not None else "   —"
            s44 = f"{vals[2]:+.2f}" if vals[2] is not None else "   —"
            dis_s = f"{disrupt:.4f}" if disrupt is not None else "      —"
            mn_s = f"{mn:+.2f}" if mn is not None else "       —"
            sd_s = f"{sd:.2f}" if sd is not None else "    —"
            print(f"{model_key:<10} {win:<8} {dis_s:>12} {mn_s:>12} "
                  f"{s42:>8} {s43:>8} {s44:>8} {sd_s:>8}")
        print(f"{'':10} {'':8} {'':12} {'':12} "
              f"{'':8} {'':8} {'':8} ρ={rho:+.3f} (p={pval:.3f})")
        print()

        # Generate figure
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ax.scatter(x_vals, y_vals, s=90, color="tab:blue", edgecolor="black",
                   linewidths=0.7, zorder=3)
        for win, disrupt, mn, _, _ in rows:
            if disrupt is not None and mn is not None:
                ax.annotate(win, (disrupt, mn), textcoords="offset points",
                            xytext=(4, 3), fontsize=7.5, color="#444")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_xlabel("Downstream clean disruption", fontsize=11)
        ax.set_ylabel("Mean perturbed Δ (pp, 3 seeds)", fontsize=11)
        ax.set_title(f"{cfg['display']} (retrained adapters)\n"
                     f"Spearman ρ = {rho:+.3f}", fontsize=11)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        out_path = os.path.join(FIG_DIR, f"disruption_{model_key}_scatter_phase2.pdf")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")

    print("\n=== Spearman ρ summary ===")
    for model_key, (rho, pval) in all_rho.items():
        print(f"  {model_key:<10} ρ = {rho:+.3f}  (p = {pval:.3f})")


if __name__ == "__main__":
    main()
