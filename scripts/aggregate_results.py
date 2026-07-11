#!/usr/bin/env python3
"""
aggregate_results.py — Read completed run JSONs and print mean ± std tables.

Usage:
    python scripts/aggregate_results.py task1   # window-length ablation
    python scripts/aggregate_results.py task2   # all-layer LoRA
    python scripts/aggregate_results.py task3   # MMLU sweep
    python scripts/aggregate_results.py all     # all tasks
"""

import json
import os
import sys
import math
from typing import Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V2_DIR = os.path.join(ROOT, "results", "fixed_harness", "v2")
V1_DIR = os.path.join(ROOT, "results", "fixed_harness", "v1", "layer_sweep")


def _load(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _mean_std(vals):
    if not vals:
        return None, None
    n = len(vals)
    m = sum(vals) / n
    if n == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in vals) / (n - 1)
    return m, math.sqrt(var)


def _cell(mean, std) -> str:
    if mean is None:
        return "—"
    return f"{mean:+.1f} ± {std:.1f}"


def _latex_cell(mean, std) -> str:
    if mean is None:
        return r"\textit{pend.}"
    return f"${mean:+.1f} \\pm {std:.1f}$"


# ─────────────────────────────────────────────────────────────────────────────
# Task 1: Window-length ablation
# ─────────────────────────────────────────────────────────────────────────────

def task1():
    print("\n" + "=" * 70)
    print("TASK 1: Window-Length Ablation (Phi-3.5-mini-instruct)")
    print("Values = mean perturbed Δ (pp) ± std across 3 seeds")
    print("=" * 70)

    positions = [
        ("mid",  "L10-14", "L11-13", "L09-15", "center L12"),
        ("late", "L27-31", "L28-30", "L25-31", "center L29, late clamped"),
    ]
    seeds = [42, 43, 44]

    rows = []
    for pos_name, w5_tag, w3_tag, w7_tag, note in positions:
        # Width 3 (new, v2)
        v3 = [_load(os.path.join(V2_DIR, "layer_sweep", f"phi35_{w3_tag}_seed{s}.json"))
              for s in seeds]
        v3_deltas = [d["mean_perturbed_delta"] for d in v3 if d and d.get("mean_perturbed_delta") is not None]
        m3, s3 = _mean_std(v3_deltas)
        done3 = len(v3_deltas)

        # Width 5 (existing, v1) — note: evaluated at max_new_tokens=512
        v5 = [_load(os.path.join(V1_DIR, f"phi35_{w5_tag}_seed{s}.json")) for s in seeds]
        v5_deltas = [d["mean_perturbed_delta"] for d in v5 if d and d.get("mean_perturbed_delta") is not None]
        m5, s5 = _mean_std(v5_deltas)
        done5 = len(v5_deltas)

        # Width 7 (new, v2)
        v7 = [_load(os.path.join(V2_DIR, "layer_sweep", f"phi35_{w7_tag}_seed{s}.json"))
              for s in seeds]
        v7_deltas = [d["mean_perturbed_delta"] for d in v7 if d and d.get("mean_perturbed_delta") is not None]
        m7, s7 = _mean_std(v7_deltas)
        done7 = len(v7_deltas)

        rows.append((pos_name, note, w3_tag, done3, m3, s3, w5_tag, done5, m5, s5,
                     w7_tag, done7, m7, s7))

    print(f"\n{'Position':<8} {'Width 3':>15} {'Width 5*':>15} {'Width 7':>15}")
    print("-" * 60)
    for (pos, note, w3_tag, d3, m3, s3, w5_tag, d5, m5, s5, w7_tag, d7, m7, s7) in rows:
        c3 = _cell(m3, s3) + (f" ({d3}/3)" if d3 < 3 else "")
        c5 = _cell(m5, s5) + (f" ({d5}/3)" if d5 < 3 else "") + "*"
        c7 = _cell(m7, s7) + (f" ({d7}/3)" if d7 < 3 else "")
        print(f"{pos:<8} {c3:>15} {c5:>16} {c7:>15}")
    print("* Width-5 from v1 (max_new_tokens=512); new runs use 768.")

    print("\n--- LaTeX snippet (booktabs style) ---")
    print(r"\begin{tabular}{lrrr}")
    print(r"\toprule")
    print(r"Window position & Width 3 & Width 5 & Width 7 \\")
    print(r"\midrule")
    for (pos, note, w3_tag, d3, m3, s3, w5_tag, d5, m5, s5, w7_tag, d7, m7, s7) in rows:
        lc3 = _latex_cell(m3, s3)
        lc5 = _latex_cell(m5, s5)
        lc7 = _latex_cell(m7, s7)
        pos_label = "Mid (L10-14 center)" if pos == "mid" else "Late (L27-31 center)"
        print(f"{pos_label} & {lc3} & {lc5} & {lc7} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")

    # Summary for paper
    new_tags = ["L11-13", "L28-30", "L09-15", "L25-31"]
    n_done = sum(
        1 for tag in new_tags for s in seeds
        if _load(os.path.join(V2_DIR, "layer_sweep", f"phi35_{tag}_seed{s}.json"))
    )
    n_total = len(new_tags) * len(seeds)
    if n_done < n_total:
        print(f"\n[Status] {n_done}/{n_total} new cells complete (width-3 and width-7).")
    else:
        print("\n[Status] All new cells complete. Ready for paper.")


# ─────────────────────────────────────────────────────────────────────────────
# Task 2: All-layer LoRA
# ─────────────────────────────────────────────────────────────────────────────

def task2():
    print("\n" + "=" * 70)
    print("TASK 2: All-Layer LoRA Baseline (3 models × 3 seeds)")
    print("Values = mean perturbed Δ (pp) ± std")
    print("=" * 70)

    models = [
        ("phi35",      "Phi-3.5-mini"),
        ("qwen2.5_7b", "Qwen2.5-7B"),
        ("llama3_8b",  "Llama-3-8B"),
    ]
    seeds = [42, 43, 44]

    print(f"\n{'Model':<15} {'Mean Pert. Δ':>15} {'Clean Δ':>10} {'Seeds done':>12}")
    print("-" * 60)

    latex_rows = []
    for slug, label in models:
        deltas, clean_deltas = [], []
        for s in seeds:
            d = _load(os.path.join(V2_DIR, "all_layer", f"{slug}_all_seed{s}.json"))
            if d and d.get("mean_perturbed_delta") is not None:
                deltas.append(d["mean_perturbed_delta"])
            # Clean delta = acc_with_adapter - acc_no_adapter on clean condition
            if d and d.get("results"):
                cb = d["results"].get("clean_baseline", {})
                if cb.get("delta") is not None:
                    clean_deltas.append(cb["delta"])
        m, s = _mean_std(deltas)
        mc, sc = _mean_std(clean_deltas)
        c = _cell(m, s)
        cc = _cell(mc, sc)
        print(f"{label:<15} {c:>15} {cc:>10} {len(deltas)}/3")
        latex_rows.append((label, m, s, mc, sc))

    print("\n--- LaTeX snippet ---")
    print(r"\begin{tabular}{lrr}")
    print(r"\toprule")
    print(r"Model & Mean perturbed $\Delta$ & Clean $\Delta$ \\")
    print(r"\midrule")
    for label, m, s, mc, sc in latex_rows:
        print(f"{label} & {_latex_cell(m, s)} & {_latex_cell(mc, sc)} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


# ─────────────────────────────────────────────────────────────────────────────
# Task 3: MMLU sweep
# ─────────────────────────────────────────────────────────────────────────────

def task3():
    print("\n" + "=" * 70)
    print("TASK 3: MMLU Layer Sweep")
    print("Values = mean perturbed Δ (pp) ± std across 3 seeds")
    print("=" * 70)

    model_configs = [
        ("phi35", "Phi-3.5-mini",
         ["L00-04", "L05-09", "L10-14", "L15-19", "L20-24", "L27-31"]),
        ("qwen2.5_7b", "Qwen2.5-7B",
         ["L00-04", "L05-09", "L08-11", "L15-19", "L20-23", "L24-27"]),
    ]
    seeds = [42, 43, 44]

    for slug, label, windows in model_configs:
        print(f"\n{label}:")
        print(f"  {'Window':<10} {'Mean Pert. Δ':>15} {'Seeds':>6}")
        print("  " + "-" * 35)
        latex_rows = []
        for win in windows:
            deltas = []
            for s in seeds:
                d = _load(os.path.join(V2_DIR, "mmlu", f"{slug}_{win}_seed{s}.json"))
                if d and d.get("mean_perturbed_delta") is not None:
                    deltas.append(d["mean_perturbed_delta"])
            m, st = _mean_std(deltas)
            c = _cell(m, st)
            done = len(deltas)
            print(f"  {win:<10} {c:>15} {done}/3")
            latex_rows.append((win, m, st))

        print(f"\n  --- LaTeX ({label}) ---")
        print(r"  \begin{tabular}{lr}")
        print(r"  \toprule")
        print(r"  Window & Mean perturbed $\Delta$ (MMLU) \\")
        print(r"  \midrule")
        for win, m, st in latex_rows:
            print(f"  {win} & {_latex_cell(m, st)} \\\\")
        print(r"  \bottomrule")
        print(r"  \end{tabular}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    if target in ("task1", "all"):
        task1()
    if target in ("task2", "all"):
        task2()
    if target in ("task3", "all"):
        task3()
