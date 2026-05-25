"""
expA_cohens_d.py - Per-Layer Cohen's d Profiles

For each model x perturbation condition, computes the per-layer Cohen's d
between LRD values of examples that succeed vs fail under perturbation.

Uses existing lrd_profile + is_correct data from raw_gsm8k.json - no forward
passes required.

Output:
  expA_perlayer_cohens_d.json
  figures/cohens_d_phi35.pdf
  figures/cohens_d_llama3.pdf
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Apply publication styling
from figure_style import apply_style, remove_spines, add_legend_below
apply_style()

ROOT = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

MODEL_DATA = {
    "Phi3.5":  os.path.join(ROOT, "Phi3.5/lrd_results/phi_hardening2/raw_gsm8k.json"),
    "Llama3":  os.path.join(ROOT, "Llama/lrd_results/raw_gsm8k.json"),
    "Qwen2.5": os.path.join(ROOT, "Qwen2.5/lrd_results/qwen_gsm8k/raw_gsm8k.json"),
    "Gemma2":  os.path.join(ROOT, "Gemma2/lrd_results/gemma2_9b_gsm8k/raw_gsm8k.json"),
}

PATCHING_WINDOW = {
    "Phi3.5":  (0, 8),
    "Llama3":  (0, 8),
    "Mistral": (0, 8),
    "Qwen2.5": (0, 8),
    "Gemma2":  (0, 8),
}
LORA_WINDOW = {
    "Phi3.5":  (15, 19),
    "Llama3":  (20, 24),
    "Mistral": (20, 24),
    "Qwen2.5": (24, 27),
    "Gemma2":  (24, 29),
}


def cohens_d(a: list, b: list) -> float:
    """Cohen's d = (mean_b - mean_a) / pooled_std (b=fail, a=succeed)."""
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1 = float(np.var(a, ddof=1))
    var2 = float(np.var(b, ddof=1))
    pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled < 1e-10:
        return 0.0
    return float((np.mean(b) - np.mean(a)) / pooled)


def compute_cohens_d_profile(records: list) -> dict:
    """
    Split records by is_correct, then compute per-layer Cohen's d.

    Returns dict with keys: 'd_profile', 'n_succeed', 'n_fail', 'n_layers',
    'peak_d', 'peak_layer'.  Returns None if either group is empty.
    """
    succeed = [r for r in records if r["is_correct"]]
    fail    = [r for r in records if not r["is_correct"]]
    if not succeed or not fail:
        return None

    n_layers  = len(records[0]["lrd_profile"])
    d_profile = []
    for l in range(n_layers):
        s_vals = [r["lrd_profile"][l] for r in succeed]
        f_vals = [r["lrd_profile"][l] for r in fail]
        d_profile.append(cohens_d(s_vals, f_vals))

    d_arr = np.array(d_profile)
    return {
        "d_profile":  d_arr.tolist(),
        "n_succeed":  len(succeed),
        "n_fail":     len(fail),
        "n_layers":   n_layers,
        "peak_d":     float(d_arr.max()),
        "peak_layer": int(d_arr.argmax()),
    }


def main():
    out_json = {}
    colors   = plt.cm.tab10(np.linspace(0, 1, 10))

    for model_name, data_path in MODEL_DATA.items():
        if not os.path.exists(data_path):
            print(f"[SKIP] {model_name}: {data_path} not found")
            continue

        print(f"\n=== {model_name} ===")
        with open(data_path) as f:
            raw = json.load(f)

        model_out = {}
        fig, ax   = plt.subplots(figsize=(10, 4.2))  # reduced height ~30%

        for idx, (cond, records) in enumerate(raw.items()):
            res = compute_cohens_d_profile(records)
            if res is None:
                print(f"  [SKIP] {cond}: no succeed/fail split")
                continue

            print(f"  {cond}: n_succeed={res['n_succeed']}, n_fail={res['n_fail']}, "
                  f"peak_d={res['peak_d']:.3f} at layer {res['peak_layer']}")
            model_out[cond] = res

            # Plot with thicker lines
            ax.plot(res["d_profile"], label=cond, color=colors[idx % len(colors)],
                    linewidth=2.0)

        # Overlay windows with reference lines (thinner)
        pw = PATCHING_WINDOW.get(model_name)
        lw = LORA_WINDOW.get(model_name)
        if pw:
            ax.axvspan(pw[0], pw[1], alpha=0.10, color="red",
                       label=f"Patching window ({pw[0]}-{pw[1]})")
        if lw:
            ax.axvspan(lw[0], lw[1], alpha=0.10, color="blue",
                       label=f"LoRA window ({lw[0]}-{lw[1]})")

        ax.axhline(0, color="black", linewidth=1.0, linestyle="--")
        ax.set_xlabel("Layer")
        # Use proper LaTeX rendering for Cohen's d
        ax.set_ylabel(r"Cohen's $d$", fontsize=13)
        ax.set_title(f"Per-Layer Cohen's $d$ Profile - {model_name} / GSM8K")

        # Remove top/right spines
        remove_spines(ax)

        # Legend below plot in 2 rows x 4 columns
        add_legend_below(ax, ncol=4, y_offset=-0.18)

        plt.tight_layout()

        # Save to figures/ directory with correct naming
        model_key = model_name.lower().replace(".", "")
        fig_path = os.path.join(FIG_DIR, f"cohens_d_{model_key}.pdf")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  -> figure: {fig_path}")

        out_json[model_name] = model_out

    out_path = os.path.join(ROOT, "expA_perlayer_cohens_d.json")
    with open(out_path, "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
