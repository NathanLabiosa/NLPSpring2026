"""
generate_supplement_figures.py — supplemental figures for the paper rewrite.

Generates:
  1. figures/qwen14b_patching_per_layer.pdf — per-layer Qwen-14B recovery curve
  2. figures/layer_sweep_per_model_{slug}.pdf — one panel per model showing
     all 3 seeds + mean for each window (dispersion view of tab:layer_sweep_panel)
  3. figures/harness_artifact.pdf — paired-bar comparison of old vs fixed
     harness Δ for the Phi-3.5 L15-19 and Qwen L24-27 cells

Plus a LaTeX dump of the 48-row Qwen-14B per-layer recovery table.
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
FIG_DIR = os.path.join(ROOT, "figures")
sys.path.insert(0, ROOT)
from figure_style import apply_style, remove_spines

apply_style()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Qwen-14B per-layer patching recovery curve
# ─────────────────────────────────────────────────────────────────────────────

def figure_qwen14b_patching():
    d = json.load(open(os.path.join(
        ROOT, "Qwen2.5/scale_experiments/patch_results_14B/patching_gsm8k_typos.json")))
    layer_recovery = {int(k): 100.0 * sum(v) / len(v) for k, v in d["layer_recovery"].items()}
    layers = sorted(layer_recovery)
    vals = [layer_recovery[l] for l in layers]

    n = len(layers)
    third = n // 3
    early = np.mean(vals[:third])
    mid = np.mean(vals[third:2 * third])
    late = np.mean(vals[2 * third:])
    peak_l = max(layer_recovery, key=layer_recovery.get)
    worst_l = min(layer_recovery, key=layer_recovery.get)

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.bar(layers, vals, color="#FF7F0E", width=0.85, alpha=0.85, label="Per-layer recovery")
    ax.axhline(early, xmin=0, xmax=third / n, color="#444", linewidth=1.6,
               linestyle="-", alpha=0.7)
    ax.axhline(mid, xmin=third / n, xmax=2 * third / n, color="#444", linewidth=1.6,
               linestyle="-", alpha=0.7)
    ax.axhline(late, xmin=2 * third / n, xmax=1.0, color="#444", linewidth=1.6,
               linestyle="-", alpha=0.7)
    ax.text(third / 2, early + 2, f"early: {early:.1f}%", ha="center", fontsize=9, color="#222")
    ax.text(third + third / 2, mid + 2, f"mid: {mid:.1f}%", ha="center", fontsize=9, color="#222")
    ax.text(2 * third + third / 2, late + 2, f"late: {late:.1f}%", ha="center", fontsize=9, color="#222")

    # Annotate peak and worst
    ax.annotate(f"peak L{peak_l}={layer_recovery[peak_l]:.1f}%",
                xy=(peak_l, layer_recovery[peak_l]),
                xytext=(peak_l - 8, layer_recovery[peak_l] + 4),
                fontsize=9, color="#1F77B4",
                arrowprops=dict(arrowstyle="-", color="#1F77B4", lw=0.8))
    ax.annotate(f"worst L{worst_l}={layer_recovery[worst_l]:.1f}%",
                xy=(worst_l, layer_recovery[worst_l]),
                xytext=(worst_l - 16, layer_recovery[worst_l] - 8),
                fontsize=9, color="#D62728",
                arrowprops=dict(arrowstyle="-", color="#D62728", lw=0.8))

    ax.set_xlabel("Layer index (patched)")
    ax.set_ylabel("Recovery on perturbed input (%)")
    ax.set_title("Qwen2.5-14B per-layer activation-patching recovery (typos 5%, n=84 pairs)",
                 fontsize=11, loc="left")
    ax.set_ylim(0, 100)
    ax.set_xlim(-1, n)
    remove_spines(ax)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "qwen14b_patching_per_layer.pdf")
    plt.savefig(out)
    plt.savefig(out.replace(".pdf", ".png"), dpi=200)
    plt.close(fig)
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Per-model layer-sweep seed-dispersion panels
# ─────────────────────────────────────────────────────────────────────────────

LAYER_COUNT = {"phi35": 32, "qwen2.5_7b": 28, "llama3_8b": 32,
               "mistral_7b_v03": 32, "gemma2_9b": 42}
DISPLAY = {
    "phi35":          ("Phi-3.5-mini-instruct",   "#1F77B4"),
    "qwen2.5_7b":     ("Qwen2.5-7B-Instruct",     "#2CA02C"),
    "llama3_8b":      ("Llama-3-8B-Instruct",     "#FF7F0E"),
    "gemma2_9b":      ("Gemma-2-9B",              "#9467BD"),
    "mistral_7b_v03": ("Mistral-7B-Instruct-v0.3", "#D62728"),
}


def figure_seed_panels():
    pat = re.compile(r"(.+?)_(L\d{2}-\d{2})_seed(\d+)\.json")
    cells = defaultdict(list)  # (slug, win) → [(seed, mean_delta), ...]
    for fp in sorted(glob.glob(os.path.join(ROOT, "results/fixed_harness/v1/layer_sweep/*.json"))):
        name = os.path.basename(fp)
        m = pat.match(name)
        if not m:
            continue
        slug, win, seed = m.group(1), m.group(2), int(m.group(3))
        d = json.load(open(fp))
        cells[(slug, win)].append((seed, d["mean_perturbed_delta"]))

    by_model = defaultdict(dict)
    for (slug, win), rows in cells.items():
        rows.sort()
        by_model[slug][win] = rows

    seed_colors = {42: "#1F77B4", 43: "#2CA02C", 44: "#FF7F0E"}
    seed_markers = {42: "o", 43: "s", 44: "D"}

    for slug, windows in by_model.items():
        name, model_color = DISPLAY[slug]
        win_tags = sorted(windows.keys())
        x = np.arange(len(win_tags))
        fig, ax = plt.subplots(figsize=(7.5, 3.6))

        all_vals = []
        for s in (42, 43, 44):
            xs, ys = [], []
            for i, w in enumerate(win_tags):
                for seed, val in windows[w]:
                    if seed == s:
                        xs.append(i + (s - 43) * 0.12)
                        ys.append(val)
                        all_vals.append(val)
            ax.scatter(xs, ys, color=seed_colors[s], marker=seed_markers[s],
                       s=42, label=f"seed {s}", zorder=3, edgecolor="black", linewidth=0.4)

        # Mean overlay (computed across whatever seeds exist for each window)
        mean_xs, mean_ys = [], []
        for i, w in enumerate(win_tags):
            vals = [v for _, v in windows[w]]
            mean_xs.append(i)
            mean_ys.append(np.mean(vals))
        ax.plot(mean_xs, mean_ys, color=model_color, linewidth=2.0,
                alpha=0.7, label="mean across seeds", zorder=2)

        ax.axhline(0.0, color="black", linewidth=0.9, linestyle="--", alpha=0.4, zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels(win_tags, fontsize=10)
        ax.set_xlabel("Layer window")
        ax.set_ylabel(r"Mean perturbed $\Delta$ (pp)")
        ax.set_title(f"{name} — per-seed layer sweep ({LAYER_COUNT[slug]} layers)",
                     fontsize=11, loc="left")
        remove_spines(ax)
        ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
        ax.legend(loc="best", fontsize=9, frameon=True, edgecolor="black", fancybox=False)

        # Auto y-range with a touch of padding; honor the Mistral catastrophe.
        lo = min(all_vals) - 1.5
        hi = max(2.0, max(all_vals) + 1.0)
        ax.set_ylim(lo, hi)

        plt.tight_layout()
        out = os.path.join(FIG_DIR, f"layer_sweep_per_model_{slug}.pdf")
        plt.savefig(out)
        plt.savefig(out.replace(".pdf", ".png"), dpi=200)
        plt.close(fig)
        print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Harness artifact figure
# ─────────────────────────────────────────────────────────────────────────────

def figure_harness_artifact():
    """Paired bars: old (sweep, max_new_tokens=100) vs fixed-harness Δ on the
    same checkpoint family for two load-bearing claims."""
    cells = [
        # (label, old_delta, new_delta, source for caption)
        ("Phi-3.5\nL15-19 stab",  +7.3,  -3.5),
        ("Qwen-7B\nL24-27 vanilla", +11.6, -7.5),
    ]
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    x = np.arange(len(cells))
    w = 0.36

    old_vals = [c[1] for c in cells]
    new_vals = [c[2] for c in cells]
    ax.bar(x - w / 2, old_vals, w, color="#D62728", alpha=0.85,
           label="Old harness (sweep, max_new_tokens=100)")
    ax.bar(x + w / 2, new_vals, w, color="#1F77B4", alpha=0.85,
           label="Fixed harness v1 (max_new_tokens=512)")

    # Value labels above/below bars
    for xi, val in zip(x - w / 2, old_vals):
        ax.text(xi, val + (0.4 if val >= 0 else -0.7), f"{val:+.1f}",
                ha="center", fontsize=9, color="#A02020")
    for xi, val in zip(x + w / 2, new_vals):
        ax.text(xi, val + (0.4 if val >= 0 else -0.7), f"{val:+.1f}",
                ha="center", fontsize=9, color="#1A5490")

    ax.axhline(0.0, color="black", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in cells], fontsize=10)
    ax.set_ylabel(r"Mean perturbed $\Delta$ (pp)")
    ax.set_title("Eval-harness artifact: same checkpoint, two harnesses",
                 fontsize=11, loc="left")
    ax.set_ylim(-12, 14)
    remove_spines(ax)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
    ax.legend(loc="upper right", fontsize=9, frameon=True, edgecolor="black", fancybox=False)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "harness_artifact.pdf")
    plt.savefig(out)
    plt.savefig(out.replace(".pdf", ".png"), dpi=200)
    plt.close(fig)
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. LaTeX 48-row Qwen-14B per-layer table
# ─────────────────────────────────────────────────────────────────────────────

def dump_qwen14b_table():
    handoff = json.load(open(os.path.join(ROOT, "results/paper_figures_handoff.json")))
    per_layer = handoff["qwen14b_patching"]["per_layer_recovery_pct"]
    n = len(per_layer)

    rows_per_col = 16
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering\small")
    lines.append(r"\begin{tabular}{c c | c c | c c}")
    lines.append(r"\toprule")
    lines.append(r"Layer & Recovery & Layer & Recovery & Layer & Recovery \\")
    lines.append(r"\midrule")
    for i in range(rows_per_col):
        l1 = i
        l2 = i + rows_per_col
        l3 = i + 2 * rows_per_col
        cells = []
        for l in (l1, l2, l3):
            key = str(l)
            if l < n and key in per_layer:
                cells.append(f"L{l:02d}")
                cells.append(f"{per_layer[key]:.1f}\\%")
            else:
                cells.append("---")
                cells.append("---")
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Qwen2.5-14B per-layer activation-patching recovery on GSM8K"
                 r" (typos 5\%, $n=84$ clean-correct pairs). Identity-patch sanity 85.0\%"
                 r" clears the 80\% ceiling Qwen-7B did not.}")
    lines.append(r"\label{tab:qwen14b_patching_per_layer}")
    lines.append(r"\end{table}")

    out = os.path.join(ROOT, "results", "tab_qwen14b_patching_per_layer.tex")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved: {out}")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    print("=== 1. Qwen-14B per-layer patching figure ===")
    figure_qwen14b_patching()
    print("\n=== 2. Per-model layer-sweep seed panels ===")
    figure_seed_panels()
    print("\n=== 3. Harness artifact comparison ===")
    figure_harness_artifact()
    print("\n=== 4. LaTeX 48-row patching table ===")
    dump_qwen14b_table()


if __name__ == "__main__":
    main()
