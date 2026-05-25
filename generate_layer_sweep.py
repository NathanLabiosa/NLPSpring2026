"""
generate_layer_sweep.py — regenerate figures/layer_sweep.pdf from fixed-harness Phase 1 data.

For each (model, window) cell in results/fixed_harness/v1/layer_sweep/, plot
the mean perturbed Δ (with-adapter − no-adapter, averaged over 6 perturbations
× 500 GSM8K test items) versus depth-normalized window position. Error bars
are the std across 3 seeds (42, 43, 44); partial cells (fewer seeds) are
drawn with open markers.

Two panels:
  (a) Panel models in the diagnostic-track narrative: Phi-3.5, Qwen-7B,
      Llama-3-8B, Gemma-2-9B. Y-axis tuned for the [-8, +2] band these models
      actually occupy.
  (b) Mistral-7B-v0.3 alone, with the wider [-22, +1] range its catastrophic
      Δ requires.

Output: figures/layer_sweep.pdf
"""

import os
import sys
import json
import glob
import re
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from figure_style import apply_style, remove_spines

apply_style()

# Layer counts per base model (number of transformer layers).
LAYER_COUNT = {
    "phi35":           32,
    "qwen2.5_7b":      28,
    "llama3_8b":       32,
    "mistral_7b_v03":  32,
    "gemma2_9b":       42,
}

DISPLAY = {
    "phi35":          {"name": "Phi-3.5-mini-instruct",   "color": "#1F77B4", "marker": "o"},
    "qwen2.5_7b":     {"name": "Qwen2.5-7B-Instruct",     "color": "#2CA02C", "marker": "s"},
    "llama3_8b":      {"name": "Llama-3-8B-Instruct",     "color": "#FF7F0E", "marker": "D"},
    "gemma2_9b":      {"name": "Gemma-2-9B",              "color": "#9467BD", "marker": "^"},
    "mistral_7b_v03": {"name": "Mistral-7B-Instruct-v0.3", "color": "#D62728", "marker": "v"},
}

PANEL_A_MODELS = ["phi35", "qwen2.5_7b", "llama3_8b", "gemma2_9b"]
PANEL_B_MODELS = ["mistral_7b_v03"]


def load_results():
    files = sorted(glob.glob(os.path.join(
        ROOT, "results/fixed_harness/v1/layer_sweep/*.json")))
    pat = re.compile(r"(.+?)_(L(\d{2})-(\d{2}))_seed(\d+)\.json")
    groups = defaultdict(list)
    for fp in files:
        name = os.path.basename(fp)
        m = pat.match(name)
        if not m:
            continue
        slug, win_tag, ss, ee, seed = m.group(1), m.group(2), int(m.group(3)), int(m.group(4)), int(m.group(5))
        d = json.load(open(fp))
        groups[(slug, win_tag, ss, ee)].append((seed, d))
    return groups


def aggregate(groups):
    """Returns {slug: [(depth_frac, mean, std, n_seeds, win_tag, partial), ...]} sorted by depth."""
    by_model = defaultdict(list)
    for (slug, win_tag, ss, ee), rows in groups.items():
        L = LAYER_COUNT[slug]
        center = 0.5 * (ss + ee)
        depth = center / (L - 1)
        means = [d["mean_perturbed_delta"] for _, d in rows]
        mean = float(np.mean(means))
        std = float(np.std(means, ddof=0)) if len(means) > 1 else 0.0
        n_seeds = len(rows)
        partial = n_seeds < 3
        by_model[slug].append((depth, mean, std, n_seeds, win_tag, partial))
    for slug in by_model:
        by_model[slug].sort()
    return by_model


def plot_panel(ax, by_model, models, title, ylim, annotate_outliers=False):
    """Plot one panel with the given list of models."""
    for slug in models:
        if slug not in by_model:
            continue
        rows = by_model[slug]
        depths = [r[0] for r in rows]
        means = [r[1] for r in rows]
        stds = [r[2] for r in rows]
        partial_mask = [r[5] for r in rows]
        meta = DISPLAY[slug]

        # Solid line through ALL points (so the curve is continuous).
        ax.plot(depths, means, color=meta["color"], linewidth=1.8,
                alpha=0.85, zorder=2)

        # Filled markers for fully-seeded cells, open markers for partial.
        for i, (d, m, s, partial) in enumerate(zip(depths, means, stds, partial_mask)):
            if partial:
                ax.errorbar(d, m, yerr=s, color=meta["color"],
                            marker=meta["marker"], markersize=7,
                            markerfacecolor="white", markeredgecolor=meta["color"],
                            markeredgewidth=1.5, capsize=3, linestyle="none", zorder=3)
            else:
                ax.errorbar(d, m, yerr=s, color=meta["color"],
                            marker=meta["marker"], markersize=7,
                            capsize=3, linestyle="none", zorder=3)

        # One labelled handle per model for the legend (use a dummy line).
        ax.plot([], [], color=meta["color"], marker=meta["marker"], markersize=7,
                linewidth=1.8, label=meta["name"])

    # Zero reference line.
    ax.axhline(0.0, color="black", linewidth=0.9, linestyle="--", alpha=0.5, zorder=1)

    ax.set_xlabel("Depth-normalized window position")
    ax.set_ylabel(r"Mean perturbed $\Delta$ (pp)")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=12, loc="left")
    remove_spines(ax)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
    ax.legend(loc="lower right", frameon=True, fancybox=False, edgecolor="black",
              fontsize=9)


def main():
    groups = load_results()
    print(f"Loaded {sum(len(v) for v in groups.values())} JSONs across "
          f"{len(groups)} (model, window) cells")
    by_model = aggregate(groups)
    for slug in sorted(by_model):
        partials = [r for r in by_model[slug] if r[5]]
        full = [r for r in by_model[slug] if not r[5]]
        print(f"  {slug}: {len(full)} fully-seeded, {len(partials)} partial")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.0, 4.4),
                                    gridspec_kw={"width_ratios": [1.0, 0.60]})

    plot_panel(axA, by_model, PANEL_A_MODELS,
               title="(a) Panel models",
               ylim=(-9, 2.5))

    plot_panel(axB, by_model, PANEL_B_MODELS,
               title="(b) Mistral (config check pending)",
               ylim=(-23, 1))

    plt.tight_layout(w_pad=3.0)

    out = os.path.join(ROOT, "figures", "layer_sweep.pdf")
    plt.savefig(out)
    print(f"Saved: {out}")

    # Also write a PNG for quick web preview.
    out_png = os.path.join(ROOT, "figures", "layer_sweep.png")
    plt.savefig(out_png, dpi=200)
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
