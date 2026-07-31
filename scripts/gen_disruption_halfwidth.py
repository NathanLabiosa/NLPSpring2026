#!/usr/bin/env python3
"""
gen_disruption_halfwidth.py — Restyle disruption line-plots for qwen and gemma.

Targets 0.48\\textwidth columns (prints ~3.1 in at scale 0.63):
  figsize=(5, 3.75), fonts title=11 ax=10 ticks=9 legend=8,
  constrained_layout, legend ncol=2 inside upper-left.

Reads existing disruption JSONs — no GPU, no recompute.
Overwrites disruption_qwen.pdf / disruption_gemma.pdf in figures/.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(ROOT, "figures")

CONFIGS = {
    "phi35": {
        "json":       "expF_clean_disruption_phi35.json",
        "title":      "Phi-3.5-mini / GSM8K (n=200)",
        "out":        "disruption_phi35.pdf",
    },
    "qwen": {
        "json":       "expF_clean_disruption_qwen.json",
        "title":      "Qwen2.5-7B / GSM8K (n=200)",
        "out":        "disruption_qwen.pdf",
    },
    "gemma": {
        "json":       "expF_clean_disruption_gemma.json",
        "title":      "Gemma-2-9B / GSM8K (n=200)",
        "out":        "disruption_gemma.pdf",
    },
}


def remove_spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def gen_figure(model_key, cfg):
    cache_path = os.path.join(RESULTS_DIR, cfg["json"])
    all_results = json.load(open(cache_path))
    all_results.pop("_n_samples", None)

    n_layers = max(
        max(int(k) for k in r["disruption"]) + 1
        for r in all_results.values()
    )
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))

    fig, ax = plt.subplots(figsize=(5, 3.75), constrained_layout=True)

    for (sweep_name, res), color in zip(all_results.items(), colors):
        lo, hi = res["window"]
        lrd_vals = [res["disruption"].get(str(l), 0.0) for l in range(n_layers)]
        ax.plot(lrd_vals, label=f"L{lo}-{hi}", color=color, linewidth=1.8)

    ax.set_xlabel("Layer", fontsize=10)
    ax.set_ylabel("Mean cosine distance (clean input)", fontsize=10)
    ax.set_title(cfg["title"], fontsize=11)
    ax.tick_params(axis="both", labelsize=9)

    remove_spines(ax)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9, ncol=2)

    out_path = os.path.join(FIG_DIR, cfg["out"])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    for model_key, cfg in CONFIGS.items():
        gen_figure(model_key, cfg)


if __name__ == "__main__":
    main()
