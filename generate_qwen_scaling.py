"""
generate_qwen_scaling.py - Late/early LRD ratio vs model scale, two families.

Reads results/scale_table.json (produced by generate_scale_table.py with the
asymmetric-thirds convention that reproduces experiment_results.md §1 exactly).
Plots one point per (family, size). Two trace lines: Qwen 1.5B/7B/14B and
Llama 1B/8B. The Qwen 7B point — previously blank — and both Llama points are
filled in here.

Output: figures/qwen_scaling_lrd.pdf (+ .png)
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from figure_style import apply_style, remove_spines

apply_style()

FAMILY_STYLE = {
    "Qwen":  {"color": "#D62728", "marker": "o", "label": "Qwen2.5"},
    "Llama": {"color": "#1F77B4", "marker": "s", "label": "Llama-3"},
}


def main():
    path = os.path.join(ROOT, "results", "scale_table.json")
    if not os.path.exists(path):
        raise SystemExit(
            f"Missing {path}; run generate_scale_table.py first to produce it."
        )
    data = json.load(open(path))

    by_family = {}
    for label, payload in data.items():
        fam = payload["family"]
        by_family.setdefault(fam, []).append(
            (label, payload["params"], payload["mean_ratio"])
        )

    fig, ax = plt.subplots(figsize=(6.4, 4.2))

    for fam, points in by_family.items():
        points.sort(key=lambda r: r[1])
        xs = [p[1] for p in points]
        ys = [p[2] for p in points]
        st = FAMILY_STYLE[fam]
        ax.plot(xs, ys, color=st["color"], linewidth=2.0, zorder=2,
                label=st["label"])
        ax.scatter(xs, ys, s=90, color=st["color"], marker=st["marker"],
                   zorder=3, edgecolor="black", linewidth=0.8)

        # Always label each point with its short name and ratio.
        # Most points get the label to the right; Llama-3.2-1B gets it above
        # to avoid sitting on top of the line / collisions with other annotations.
        for x, y, lbl in zip(xs, ys, [p[0] for p in points]):
            size_tag = lbl.split("-")[-1]  # "1.5B", "7B", etc.
            if lbl == "Llama-3.2-1B":
                ax.annotate(f"{size_tag}\n({y:.2f})", xy=(x, y),
                            xytext=(0, 10), textcoords="offset points",
                            fontsize=9, va="bottom", ha="center")
            else:
                ax.annotate(f"{size_tag}\n({y:.2f})", xy=(x, y),
                            xytext=(8, -2), textcoords="offset points",
                            fontsize=9, va="top", ha="left")

    ax.axhline(1.0, color="black", linewidth=0.9, linestyle="--", alpha=0.5,
               zorder=1)

    ax.set_xscale("log")
    ax.set_xlim(0.8e9, 2.4e10)
    ax.set_ylim(0.5, 6.5)
    ax.set_xlabel("Model parameters (log scale)")
    ax.set_ylabel(r"Mean late/early LRD ratio")
    ax.set_title("Late-accumulation intensifies with scale (Qwen and Llama)",
                 fontsize=12, loc="left")
    remove_spines(ax)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
    ax.legend(loc="upper left", frameon=True, fancybox=False, edgecolor="black")

    plt.tight_layout()

    out = os.path.join(ROOT, "figures", "qwen_scaling_lrd.pdf")
    plt.savefig(out)
    print(f"Saved: {out}")
    out_png = os.path.join(ROOT, "figures", "qwen_scaling_lrd.png")
    plt.savefig(out_png, dpi=200)
    print(f"Saved: {out_png}")

    print("\n--- Rendered points (must match results/scale_table.json) ---")
    for fam, points in by_family.items():
        for label, params, ratio in sorted(points, key=lambda r: r[1]):
            print(f'  {label:<14}  params={params:.2e}  ratio={ratio:.3f}')


if __name__ == "__main__":
    main()
