"""
generate_scale_table.py - Unified late/early LRD ratio table across Qwen and
Llama scale points.

Convention: the same one used to produce experiment_results.md §1.
  early = mean LRD across hidden states [0 : n // 3]            (floor)
  late  = mean LRD across hidden states [n - ceil(n/3) : n]     (ceil)
  ratio = late / early
  ratio is computed per perturbation on the mean-across-records LRD profile;
  the headline scale-point number is the mean across perturbations.

This reproduces the §1 Qwen values 1.59 / 2.90 / 4.10 exactly. Applied
uniformly to Llama-3.2-1B and Llama-3-8B for an apples-to-apples cross-
family comparison.

Outputs:
  results/scale_table.json   — full data + per-perturbation breakdown
  results/scale_table.tex    — LaTeX-ready table
  prints the five-point values needed for figures/qwen_scaling_lrd.pdf and
  any companion Llama scaling figure.
"""

import json
import math
import os
import sys
from collections import OrderedDict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

POINTS = OrderedDict([
    ("Qwen2.5-1.5B", {
        "family":  "Qwen",
        "params":  1.5e9,
        "raw":     "Qwen2.5/scale_experiments/lrd_results_1.5B/raw_gsm8k.json",
    }),
    ("Qwen2.5-7B", {
        "family":  "Qwen",
        "params":  7.6e9,
        "raw":     "Qwen2.5/lrd_results/qwen_gsm8k/raw_gsm8k.json",
    }),
    ("Qwen2.5-14B", {
        "family":  "Qwen",
        "params":  1.4e10,
        "raw":     "Qwen2.5/scale_experiments/lrd_results_14B/raw_gsm8k.json",
    }),
    ("Llama-3.2-1B", {
        "family":  "Llama",
        "params":  1.24e9,
        "raw":     "Llama/lrd_results_3.2_1B/raw_gsm8k.json",
    }),
    ("Llama-3-8B", {
        "family":  "Llama",
        "params":  8.03e9,
        "raw":     "Llama/lrd_results/raw_gsm8k.json",
    }),
])

PERTS = ["Typos_5%", "OCR_5%", "Whitespace_10%", "Case_10%", "Homophones_20%", "Speech_10%"]


def ratio(lrd_profile: np.ndarray) -> tuple:
    """Asymmetric-thirds late/early ratio. Returns (early_mean, late_mean, ratio)."""
    n = len(lrd_profile)
    front = n // 3
    back  = math.ceil(n / 3)
    early = float(lrd_profile[:front].mean())
    late  = float(lrd_profile[-back:].mean())
    return early, late, late / early


def main():
    out = OrderedDict()
    print(f'{"scale":<14}{"n_hs":>5}{"n_rec":>7}    ' +
          ' '.join(f'{p[:8]:>9}' for p in PERTS) + '   mean')
    for label, meta in POINTS.items():
        path = os.path.join(ROOT, meta["raw"])
        if not os.path.exists(path):
            print(f'  [SKIP] {label}: {path} not found')
            continue
        d = json.load(open(path))
        n_rec = len(d[PERTS[0]])

        per_pert = OrderedDict()
        ratios = []
        for p in PERTS:
            prof = np.mean([r["lrd_profile"] for r in d[p]], axis=0)
            early, late, r = ratio(prof)
            per_pert[p] = {"early": early, "late": late, "ratio": r}
            ratios.append(r)
        mean_ratio = float(np.mean(ratios))

        n_hs = len(prof)
        front = n_hs // 3
        back  = math.ceil(n_hs / 3)
        out[label] = {
            "family":  meta["family"],
            "params":  meta["params"],
            "n_hidden_states": n_hs,
            "n_records": n_rec,
            "split":   {"early_slice": [0, front], "late_slice": [n_hs - back, n_hs]},
            "per_perturbation": per_pert,
            "mean_ratio": mean_ratio,
        }

        row = ' '.join(f'{per_pert[p]["ratio"]:>9.2f}' for p in PERTS)
        print(f'  {label:<13}{n_hs:>4}  {n_rec:>5}    {row}   {mean_ratio:.3f}')

    out_path = os.path.join(ROOT, "results", "scale_table.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=2)
    print(f'\nSaved: {out_path}')

    # LaTeX dump.
    lines = []
    lines.append(r"\begin{tabular}{lrrrrrrrr}")
    lines.append(r"\toprule")
    pert_headers = " & ".join(
        r"\textbf{" + p.replace("_", r"\_") + "}" for p in PERTS
    )
    lines.append(r"\textbf{Model} & \textbf{Params} & " + pert_headers +
                 r" & \textbf{mean} \\")
    lines.append(r"\midrule")
    for label, payload in out.items():
        cells = [payload["per_perturbation"][p]["ratio"] for p in PERTS]
        params = payload["params"]
        params_str = f"{params/1e9:.1f}B" if params < 1e10 else f"{params/1e9:.0f}B"
        cell_str = " & ".join(f"{c:.2f}" for c in cells)
        lines.append(f"{label} & {params_str} & {cell_str} & "
                     f"\\textbf{{{payload['mean_ratio']:.2f}}} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    tex_path = os.path.join(ROOT, "results", "scale_table.tex")
    open(tex_path, "w").write("\n".join(lines) + "\n")
    print(f'Saved: {tex_path}')

    # Figure 2 data points.
    print("\n--- Five-point figure inputs (mean ratio, all from same convention) ---")
    for label, payload in out.items():
        print(f'  {label:<14}  params={payload["params"]:.2e}  '
              f'mean_ratio={payload["mean_ratio"]:.3f}')


if __name__ == "__main__":
    main()
