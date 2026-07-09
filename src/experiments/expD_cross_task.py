# === lib path shim (added by repo reorg) ===
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir, 'lib'))
# === end lib path shim ===

"""
expD_cross_task.py - Cross-Task LRD Stability

For each model, computes full per-layer LRD profiles on GSM8K, MMLU, and BBH,
then measures how well the profile shape transfers across tasks.

Uses existing multi-task raw_*.json files - no forward passes needed.

Data sources:
  Phi3.5 - models/phi3.5/lrd_{gsm,mmlu,bbh}/raw_{dataset}.json
  Mistral - models/mistral/lrd_results/mistral_gsm8k/raw_{dataset}.json
  Llama3  - only GSM8K available; included with a caveat note

Procedure:
  1. Compute mean LRD profile (all layers) per model x task x perturbation type
  2. For each model, compute Spearman rho between GSM8K LRD profile and MMLU/BBH
     (for each perturbation type shared across tasks)
  3. Assess whether the profile *shape* (rank order) is stable across tasks,
     even if the *magnitude* scales

Output:
  expD_cross_task_lrd.json
  figures/cross_task_phi35.pdf
  figures/cross_task_mistral.pdf
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# Apply publication styling
from figure_style import apply_style, remove_spines, add_correlation_legend
apply_style()

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG_DIR = os.path.join(ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Data paths: model_name -> {task_label -> (raw_json_path, dataset_key)}
MODEL_TASKS = {
    "Phi3.5": {
        "GSM8K": (os.path.join(ROOT, "models/phi3.5/lrd_gsm/raw_gsm8k.json"),  "gsm8k"),
        "MMLU":  (os.path.join(ROOT, "models/phi3.5/lrd_mmlu/raw_mmlu.json"),  "mmlu"),
        "BBH":   (os.path.join(ROOT, "models/phi3.5/lrd_bbh/raw_bbh.json"),    "bbh"),
    },
    "Mistral": {
        "GSM8K": (os.path.join(ROOT, "models/mistral/lrd_results/mistral_gsm8k/raw_gsm8k.json"), "gsm8k"),
        "MMLU":  (os.path.join(ROOT, "models/mistral/lrd_results/mistral_gsm8k/raw_mmlu.json"),  "mmlu"),
        "BBH":   (os.path.join(ROOT, "models/mistral/lrd_results/mistral_gsm8k/raw_bbh.json"),   "bbh"),
    },
}

FOCUS_CONDS = ["Typos_5%", "Whitespace_10%"]


def mean_lrd_by_cond(raw: dict) -> dict:
    """Return {condition: mean_lrd_profile_array} from a raw JSON dict."""
    out = {}
    for cond, records in raw.items():
        profiles = np.array([r["lrd_profile"] for r in records])
        out[cond] = profiles.mean(axis=0)
    return out


def spearman_rho(a: np.ndarray, b: np.ndarray) -> dict:
    n = min(len(a), len(b))
    rho, p = spearmanr(a[:n], b[:n])
    return {"rho": round(float(rho), 4), "p": round(float(p), 4)}


def process_model(model_name: str, tasks: dict) -> dict:
    print(f"\n=== {model_name} ===")

    task_profiles = {}
    for task_label, (path, _ds) in tasks.items():
        if not os.path.exists(path):
            print(f"  [SKIP] {task_label}: {path} not found")
            continue
        with open(path) as f:
            raw = json.load(f)
        task_profiles[task_label] = mean_lrd_by_cond(raw)
        n_each = {k: len(v) for k, v in raw.items()}
        print(f"  Loaded {task_label}: {n_each}")

    if len(task_profiles) < 2:
        print("  [SKIP] Need at least 2 tasks - skipping model")
        return {}

    # Determine shared conditions across all available tasks
    all_conds = [set(tp.keys()) for tp in task_profiles.values()]
    shared    = set.intersection(*all_conds)
    focus     = [c for c in FOCUS_CONDS if c in shared]
    if not focus:
        focus = sorted(shared)[:2]
    print(f"  Focus conditions: {focus}")

    # Figure: overlay LRD curves for all tasks, per focus condition
    # Width increased by 50% total from original
    n_focus = len(focus)
    fig, axes = plt.subplots(1, n_focus, figsize=(10.94, 3.6), squeeze=False)
    task_colors = {"GSM8K": "tab:blue", "MMLU": "tab:orange", "BBH": "tab:green"}

    corr_results = {}
    handles, labels_legend = [], []

    for col, cond in enumerate(focus):
        ax = axes[0][col]
        profiles_for_cond = {}

        for task_label, cond_profiles in task_profiles.items():
            if cond not in cond_profiles:
                continue
            p = cond_profiles[cond]
            line, = ax.plot(p, label=task_label, color=task_colors.get(task_label, "gray"),
                           linewidth=2.5)
            profiles_for_cond[task_label] = p

            # Collect handles for shared legend (only from first panel)
            if col == 0:
                handles.append(line)
                labels_legend.append(task_label)

        # Only left panel gets y-label with proper font size
        if col == 0:
            ax.set_ylabel("Mean LRD", fontsize=13)
        ax.set_title(f"{model_name} - {cond}", fontsize=13)
        ax.tick_params(axis='both', labelsize=11)
        remove_spines(ax)
        ax.grid(True, axis='y', alpha=0.3, linewidth=0.5)

        # Spearman rho: GSM8K as reference vs others (upper-right per user request)
        if "GSM8K" in profiles_for_cond:
            ref = profiles_for_cond["GSM8K"]
            cond_corrs = {}
            for other, p in profiles_for_cond.items():
                if other == "GSM8K":
                    continue
                r = spearman_rho(ref, p)
                cond_corrs[f"GSM8K vs {other}"] = r
                print(f"    {cond} - GSM8K vs {other}: rho={r['rho']:+.3f}, p={r['p']:.4f}")

            add_correlation_legend(ax, cond_corrs, loc='upper right', fontsize=10)
            corr_results[cond] = cond_corrs

    # Layout: more room at top for title, tighter bottom
    fig.subplots_adjust(bottom=0.14, top=0.75, wspace=0.20)

    # Shared x-label closer to plot
    fig.supxlabel('Layer', fontsize=13, y=0.06)

    # Legend right below x-label
    fig.legend(handles, labels_legend,
               loc='upper center',
               bbox_to_anchor=(0.5, 0.01),
               ncol=3,
               frameon=True,
               fancybox=False,
               edgecolor='black',
               fontsize=12)

    plt.suptitle(f"Cross-Task LRD Stability - {model_name}", fontsize=14, y=0.95)

    model_key = model_name.lower().replace(".", "")
    fig_path = os.path.join(FIG_DIR, f"cross_task_{model_key}.pdf")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> figure: {fig_path}")

    # Serialise profiles
    serialised = {}
    for task_label, cond_profiles in task_profiles.items():
        serialised[task_label] = {c: p.tolist() for c, p in cond_profiles.items()}

    return {
        "profiles":     serialised,
        "focus_conds":  focus,
        "correlations": corr_results,
    }


def main():
    out = {}
    for model_name, tasks in MODEL_TASKS.items():
        result = process_model(model_name, tasks)
        if result:
            out[model_name] = result

    out_path = os.path.join(ROOT, "results", "expD_cross_task_lrd.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
