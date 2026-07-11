"""
Workstream C: Generate all publication figures for LRD paper.

C1. LRD Heatmaps (Phi-3.5, Llama-3)
C2. Activation Patching Recovery Curves (dual panel: Phi-3.5 + Llama-3)
C3. Layer Sweep Inverted-U Curve
C4. 2x2 Bar Chart (original rates) - money figure
C5. Per-Condition Taxonomy Split Figure
C6. Effect Size Ranking (from A2)
C7. NEW: Original-rate vs Matched-rate comparison (revised story)
C8. Interaction Forest Plot
"""

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

# Add root to path for figure_style import
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from figure_style import apply_style, remove_spines, COLORS as STYLE_COLORS
apply_style()

OUTPUT_DIR = os.path.join(ROOT, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Color palette (colorblind-friendly, grayscale-distinguishable)
C_DIRECTIONAL = "#2166AC"  # blue
C_UNIFORM     = "#B2182B"  # red
C_COSINE      = "#2166AC"
C_MSE         = "#B2182B"
C_CLEAN       = "#4DAF4A"  # green

PERTURBATION_ORDER = ["typos", "ocr", "speech", "homophones", "whitespace", "case"]
DIRECTIONAL_SET = {"typos", "ocr", "speech", "homophones"}
UNIFORM_SET     = {"whitespace", "case"}


def method_from_cond(cond_name):
    """Extract method name from condition key like 'typos_5pct'."""
    return cond_name.rsplit("_", 1)[0]


# ============================================================================
# C1: LRD Heatmaps
# ============================================================================
def figure_c1():
    print("Generating C1: LRD Heatmaps...")
    # Model configs with regime descriptions for titles
    models = {
        "Phi-3.5": {
            "path": os.path.join(ROOT, "models/phi3.5/lrd_gsm/raw_gsm8k.json"),
            "title": "Phi-3.5 (spike-and-suppress)",
        },
        "Llama-3": {
            "path": os.path.join(ROOT, "models/llama/lrd_results/raw_gsm8k.json"),
            "title": "Llama-3 (late-accumulation)",
        },
    }
    available = {k: v for k, v in models.items() if os.path.exists(v["path"])}
    n_models = len(available)
    if n_models == 0:
        print("  No LRD data found, skipping.")
        return

    # First pass: load all data and compute global vmin/vmax for shared colorbar
    all_heatmaps = {}
    all_conditions = {}
    global_vmin, global_vmax = float('inf'), float('-inf')

    for model_name, config in available.items():
        with open(config["path"]) as f:
            raw = json.load(f)

        conditions = sorted(raw.keys())
        if not raw[conditions[0]]:
            print(f"  {model_name}: empty data, skipping.")
            continue

        n_layers = len(raw[conditions[0]][0]["lrd_profile"])
        heatmap = np.zeros((len(conditions), n_layers))

        for i, cond in enumerate(conditions):
            profiles = [ex["lrd_profile"] for ex in raw[cond]]
            heatmap[i] = np.mean(profiles, axis=0)

        all_heatmaps[model_name] = heatmap
        all_conditions[model_name] = conditions
        global_vmin = min(global_vmin, heatmap.min())
        global_vmax = max(global_vmax, heatmap.max())

    # Create figure with shared colorbar
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models + 0.8, 4.5), squeeze=False)

    ims = []
    for idx, (model_name, config) in enumerate(available.items()):
        ax = axes[0][idx]
        heatmap = all_heatmaps[model_name]
        conditions = all_conditions[model_name]

        # Use shared vmin/vmax and aspect='auto' for square-ish cells
        im = ax.imshow(heatmap, aspect="auto", cmap="YlOrRd", interpolation="nearest",
                       vmin=global_vmin, vmax=global_vmax)
        ims.append(im)

        ax.set_xlabel("Layer")
        ax.set_yticks(range(len(conditions)))
        # Left panel: tick labels only (no y-axis label), serves as shared labels
        # Right panel: no y-axis label, no tick labels
        if idx == 0:
            ax.set_yticklabels([c.replace("_", " ") for c in conditions])
        else:
            ax.set_ylabel('')
            ax.tick_params(labelleft=False)
        ax.set_title(config["title"], fontsize=13)

    # Single shared colorbar to the right of all panels with proper spacing
    fig.subplots_adjust(right=0.88, wspace=0.3)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.025, 0.7])
    cbar = fig.colorbar(ims[-1], cax=cbar_ax, label="Mean LRD")
    cbar.ax.tick_params(labelsize=11)

    path = os.path.join(OUTPUT_DIR, "lrd_heatmaps.pdf")
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================================
# C2: Activation Patching Recovery Curves (1x2 panel: Phi-3.5 + Llama-3)
# ============================================================================
def figure_c2():
    print("Generating C2: Activation Patching Recovery Curves (dual panel)...")

    # Both models for headline patching evidence
    patching_files = {
        "Phi-3.5 (typos)": os.path.join(ROOT, "models/phi3.5/lrd_results/phi_patching_v3/patching_gsm8k_typos.json"),
        "Llama-3 (typos)": os.path.join(ROOT, "models/llama/lrd_results/patching_gsm8k_typos.json"),
    }
    available = {k: v for k, v in patching_files.items() if os.path.exists(v)}
    if not available:
        print("  No patching data found, skipping.")
        return

    n_panels = len(available)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 4.5), squeeze=False)

    for idx, (label, path) in enumerate(available.items()):
        ax = axes[0][idx]
        with open(path) as f:
            data = json.load(f)

        if "layer_recovery" in data:
            lr = data["layer_recovery"]
            layers = sorted(lr.keys(), key=int)
            recovery_rates = [np.mean(lr[l]) for l in layers]
            layer_nums = [int(l) for l in layers]

            # Bootstrap error bands
            ci_lo, ci_hi = [], []
            rng = np.random.RandomState(42)
            for l in layers:
                vals = np.array(lr[l])
                boots = [vals[rng.randint(0, len(vals), len(vals))].mean() for _ in range(2000)]
                ci_lo.append(np.percentile(boots, 2.5))
                ci_hi.append(np.percentile(boots, 97.5))

            ax.plot(layer_nums, recovery_rates, "o-", color=C_DIRECTIONAL,
                    markersize=5, linewidth=2.5)
            ax.fill_between(layer_nums, ci_lo, ci_hi, alpha=0.2, color=C_DIRECTIONAL)

            # Sanity lines (thinner, reference)
            if "sanity" in data:
                ax.axhline(data["sanity"]["identity_mean"], color="green", linestyle="--",
                          linewidth=1.5, label=f"Identity ceiling ({data['sanity']['identity_mean']:.0%})")
                ax.axhline(data["sanity"]["random_mean"], color="gray", linestyle=":",
                          linewidth=1.5, label=f"Random floor ({data['sanity']['random_mean']:.0%})")

        # Only left panel gets y-label
        if idx == 0:
            ax.set_ylabel("Recovery rate", fontsize=13)
        ax.set_title(label, fontsize=13)
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(axis='both', labelsize=11)
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.3)
        remove_spines(ax)

    # Shared x-label beneath both panels
    fig.text(0.5, 0.02, "Layer patched", ha='center', fontsize=13)
    fig.subplots_adjust(bottom=0.15)

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    path = os.path.join(OUTPUT_DIR, "patching_recovery.pdf")
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================================
# C3: Layer Sweep Inverted-U Curve
# ============================================================================
def figure_c3():
    print("Generating C3: Layer Sweep Curve...")
    windows = [
        ("0-4",   os.path.join(ROOT, "stabilizer_weights/p2_sweep_L00_04/eval_results.json")),
        ("5-9",   os.path.join(ROOT, "stabilizer_weights/p2_sweep_L05_09/eval_results.json")),
        ("10-14", os.path.join(ROOT, "stabilizer_weights/p2_sweep_L10_14/eval_results.json")),
        ("15-19", os.path.join(ROOT, "stabilizer_weights/p2_sweep_L15_19/eval_results.json")),
        ("20-24", os.path.join(ROOT, "stabilizer_weights/p2_sweep_L20_24/eval_results.json")),
        ("27-31", os.path.join(ROOT, "stabilizer_weights/p2_sweep_L27_31/eval_results.json")),
    ]

    labels, perturbed_deltas, clean_deltas = [], [], []
    for label, path in windows:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            r = json.load(f)
        labels.append(label)
        pert = [r[k]["delta"] for k in r if k != "clean_baseline"]
        perturbed_deltas.append(np.mean(pert))
        clean_deltas.append(r["clean_baseline"]["delta"])

    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = np.arange(len(labels))
    width = 0.35

    # Grouped bars with clearer spacing
    bars1 = ax.bar(x - width/2, perturbed_deltas, width, color=C_COSINE,
                   label="Avg perturbed delta", edgecolor="black", linewidth=0.8)
    bars2 = ax.bar(x + width/2, clean_deltas, width, color=C_CLEAN,
                   label="Clean delta", edgecolor="black", linewidth=0.8)

    # Shade optimal window L15-19 only (not other windows)
    optimal_idx = labels.index("15-19") if "15-19" in labels else None
    if optimal_idx is not None:
        ax.axvspan(optimal_idx - 0.5, optimal_idx + 0.5, alpha=0.15, color='gold',
                   label='Optimal window (L15-19)', zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in labels], fontsize=12)
    ax.set_xlabel("LoRA layer window", fontsize=14)
    ax.set_ylabel("Accuracy delta (%)", fontsize=14)
    ax.set_title("Phi-3.5: Robustness Gain by LoRA Placement", fontsize=15)
    ax.tick_params(axis='both', labelsize=11)

    # Horizontal zero line
    ax.axhline(0, color="black", linewidth=1.2, linestyle="-")

    ax.legend(loc='upper right', fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    remove_spines(ax)

    plt.tight_layout()
    # NOTE: intentionally NOT figures/layer_sweep.pdf (Fig. 6) — this function
    # reads the old broken-harness p2_sweep_* eval_results.json (no longer on
    # disk) and would silently clobber the correct fixed-harness figure if it
    # shared that filename. See generate_layer_sweep.py for the real Fig. 6.
    path = os.path.join(OUTPUT_DIR, "legacy_p2_layer_sweep.pdf")
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================================
# C4: 2x2 Bar Chart (original rates) - money figure
# ============================================================================
def figure_c4():
    print("Generating C4: 2x2 Bar Chart (original rates)...")
    configs = {
        "Cosine\nstab=24": os.path.join(ROOT, "results/bootstrap/cosv2_L15_stab24_high/paired_bootstrap_results.json"),
        "MSE\nstab=24": os.path.join(ROOT, "results/bootstrap/mse_L15_stab24_high/paired_bootstrap_results.json"),
    }

    stab30_configs = {
        "Cosine\nstab=30": os.path.join(ROOT, "stabilizer_weights/cosv2_L15_stab30_high/eval_results.json"),
        "MSE\nstab=30": os.path.join(ROOT, "stabilizer_weights/mse_L15_stab30_high/eval_results.json"),
    }

    fig, ax = plt.subplots(figsize=(8, 5.5))
    x_labels = []
    perturbed_vals, perturbed_los, perturbed_his = [], [], []
    clean_vals, clean_los, clean_his = [], [], []

    for label, path in configs.items():
        if not os.path.exists(path):
            continue
        with open(path) as f:
            r = json.load(f)
        x_labels.append(label)

        pool = r.get("_pooled_perturbed", {})
        perturbed_vals.append(pool.get("delta_pct", 0))
        perturbed_los.append(pool.get("ci_lo_pct", 0))
        perturbed_his.append(pool.get("ci_hi_pct", 0))

        cl = r.get("clean_baseline", {})
        clean_vals.append(cl.get("delta_pct", 0))
        clean_los.append(cl.get("ci_lo_pct", 0))
        clean_his.append(cl.get("ci_hi_pct", 0))

    for label, path in stab30_configs.items():
        if not os.path.exists(path):
            continue
        with open(path) as f:
            r = json.load(f)
        x_labels.append(label)
        pert = [r[k]["delta"] for k in r if k != "clean_baseline"]
        perturbed_vals.append(np.mean(pert))
        perturbed_los.append(np.mean(pert))
        perturbed_his.append(np.mean(pert))
        clean_vals.append(r["clean_baseline"]["delta"])
        clean_los.append(r["clean_baseline"]["delta"])
        clean_his.append(r["clean_baseline"]["delta"])

    x = np.arange(len(x_labels))
    w = 0.35

    perturbed_err = [[v - lo for v, lo in zip(perturbed_vals, perturbed_los)],
                     [hi - v for v, hi in zip(perturbed_vals, perturbed_his)]]
    clean_err = [[v - lo for v, lo in zip(clean_vals, clean_los)],
                 [hi - v for v, hi in zip(clean_vals, clean_his)]]

    ax.bar(x - w/2, perturbed_vals, w, color=C_COSINE, label="Avg perturbed delta",
           edgecolor="black", linewidth=0.8, yerr=perturbed_err, capsize=4)
    ax.bar(x + w/2, clean_vals, w, color=C_CLEAN, label="Clean delta",
           edgecolor="black", linewidth=0.8, yerr=clean_err, capsize=4)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Accuracy delta (%)")
    ax.set_title(r"Phi-3.5: Intervention Results (L15-19, $\lambda_{stab}$=3.0)")
    ax.axhline(0, color="black", linewidth=1.0)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    remove_spines(ax)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "c4_2x2_original.pdf")
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================================
# C5: Per-Condition Taxonomy Split
# ============================================================================
def figure_c5():
    print("Generating C5: Per-Condition Taxonomy Split...")
    cosine_path = os.path.join(ROOT, "results/bootstrap/cosv2_L15_stab24_high/paired_bootstrap_results.json")
    mse_path = os.path.join(ROOT, "results/bootstrap/mse_L15_stab24_high/paired_bootstrap_results.json")

    if not os.path.exists(cosine_path) or not os.path.exists(mse_path):
        print("  Missing bootstrap data, skipping.")
        return

    with open(cosine_path) as f:
        cos = json.load(f)
    with open(mse_path) as f:
        mse = json.load(f)

    ordered_conds = []
    for m in PERTURBATION_ORDER:
        for k in cos:
            if k.startswith(m + "_") and not k.startswith("_") and k != "clean_baseline":
                ordered_conds.append(k)
                break

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(ordered_conds))
    w = 0.35

    cos_vals = [cos[c]["delta_pct"] for c in ordered_conds]
    mse_vals = [mse[c]["delta_pct"] for c in ordered_conds]
    cos_errs = [[cos[c]["delta_pct"] - cos[c]["ci_lo_pct"] for c in ordered_conds],
                [cos[c]["ci_hi_pct"] - cos[c]["delta_pct"] for c in ordered_conds]]
    mse_errs = [[mse[c]["delta_pct"] - mse[c]["ci_lo_pct"] for c in ordered_conds],
                [mse[c]["ci_hi_pct"] - mse[c]["delta_pct"] for c in ordered_conds]]

    ax.bar(x - w/2, cos_vals, w, color=C_COSINE, label="Cosine (LRD)",
           edgecolor="black", linewidth=0.8, yerr=cos_errs, capsize=4)
    ax.bar(x + w/2, mse_vals, w, color=C_MSE, label="MSE",
           edgecolor="black", linewidth=0.8, yerr=mse_errs, capsize=4)

    sep_x = len([c for c in ordered_conds if method_from_cond(c) in DIRECTIONAL_SET]) - 0.5
    ax.axvline(sep_x, color="gray", linewidth=1.5, linestyle="--")

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in ordered_conds])
    ax.set_ylabel("Accuracy delta (%)")
    ax.set_title(r"Per-Condition: Cosine vs MSE (Original Rates, L15-19, stab=24, $\lambda$=3.0)")
    ax.axhline(0, color="black", linewidth=1.0)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    remove_spines(ax)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "c5_taxonomy_split_original.pdf")
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================================
# C6: Effect Size Ranking (uses A2 output)
# ============================================================================
def figure_c6():
    print("Generating C6: Effect Size Ranking...")
    a2_path = os.path.join(ROOT, "results/workstream_a/whitespace_analysis/whitespace_analysis_results.json")
    if not os.path.exists(a2_path):
        print("  A2 results not found, skipping.")
        return

    with open(a2_path) as f:
        a2 = json.load(f)

    ranking = a2["ranking"]

    fig, ax = plt.subplots(figsize=(8, 5))
    names = [r[0] for r in ranking]
    ds = [r[1] for r in ranking]
    colors = [C_DIRECTIONAL if r[2] == "directional" else C_UNIFORM for r in ranking]

    per_cond = a2["per_condition"]
    ci_errors = [[per_cond[n]["cohens_d_final"] - per_cond[n]["ci_lo"] for n in names],
                 [per_cond[n]["ci_hi"] - per_cond[n]["cohens_d_final"] for n in names]]

    y_pos = range(len(names))
    ax.barh(y_pos, ds, color=colors, edgecolor="black", linewidth=0.8,
            xerr=ci_errors, capsize=4, error_kw={"linewidth": 1.5})
    ax.set_yticks(y_pos)
    ax.set_yticklabels([n.replace("_", " ") for n in names])
    ax.set_xlabel("Cohen's $d$ (LRD-failure association)")
    ax.set_title("Effect Size: LRD Predicts Task Failure")
    ax.axvline(0, color="black", linewidth=1.0)

    legend_elements = [Patch(facecolor=C_DIRECTIONAL, edgecolor="black", label="Directional"),
                      Patch(facecolor=C_UNIFORM, edgecolor="black", label="Uniform")]
    ax.legend(handles=legend_elements, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    remove_spines(ax)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "c6_effect_size_ranking.pdf")
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================================
# C7: Original vs Matched-Rate Comparison
# ============================================================================
def figure_c7():
    print("Generating C7: Original vs Matched-Rate Comparison...")
    data_configs = {
        "original": {
            "cosine": os.path.join(ROOT, "results/bootstrap/cosv2_L15_stab24_high/paired_bootstrap_results.json"),
            "mse": os.path.join(ROOT, "results/bootstrap/mse_L15_stab24_high/paired_bootstrap_results.json"),
        },
        "matched": {
            "cosine": os.path.join(ROOT, "results/bootstrap/phi_matched_cosine_stab24/paired_bootstrap_results.json"),
            "mse": os.path.join(ROOT, "results/bootstrap/phi_matched_mse_stab24/paired_bootstrap_results.json"),
        },
    }

    for rate_type, paths in data_configs.items():
        for metric, p in paths.items():
            if not os.path.exists(p):
                print(f"  Missing {rate_type}/{metric}: {p}")
                return

    all_data = {}
    for rate_type, paths in data_configs.items():
        all_data[rate_type] = {}
        for metric, p in paths.items():
            with open(p) as f:
                all_data[rate_type][metric] = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    for ax, rate_type, title_suffix in [
        (ax1, "original", "Original Rates"),
        (ax2, "matched", "Matched Rate (all 10%)"),
    ]:
        cos_data = all_data[rate_type]["cosine"]
        mse_data = all_data[rate_type]["mse"]

        conds = []
        for m in PERTURBATION_ORDER:
            for k in cos_data:
                if k.startswith(m + "_") and not k.startswith("_") and k != "clean_baseline":
                    conds.append(k)
                    break

        x = np.arange(len(conds))
        w = 0.35

        cos_vals = [cos_data[c]["delta_pct"] for c in conds]
        mse_vals = [mse_data[c]["delta_pct"] for c in conds]
        cos_errs = [[cos_data[c]["delta_pct"] - cos_data[c]["ci_lo_pct"] for c in conds],
                    [cos_data[c]["ci_hi_pct"] - cos_data[c]["delta_pct"] for c in conds]]
        mse_errs = [[mse_data[c]["delta_pct"] - mse_data[c]["ci_lo_pct"] for c in conds],
                    [mse_data[c]["ci_hi_pct"] - mse_data[c]["delta_pct"] for c in conds]]

        ax.bar(x - w/2, cos_vals, w, color=C_COSINE, label="Cosine (LRD)",
               edgecolor="black", linewidth=0.8, yerr=cos_errs, capsize=4)
        ax.bar(x + w/2, mse_vals, w, color=C_MSE, label="MSE",
               edgecolor="black", linewidth=0.8, yerr=mse_errs, capsize=4)

        n_dir = len([c for c in conds if method_from_cond(c) in DIRECTIONAL_SET])
        sep_x = n_dir - 0.5
        ax.axvline(sep_x, color="gray", linewidth=1.5, linestyle="--")

        pretty_names = []
        for c in conds:
            m = method_from_cond(c)
            rate = c.split("_")[-1]
            pretty_names.append(f"{m}\n({rate})")

        ax.set_xticks(x)
        ax.set_xticklabels(pretty_names)
        ax.set_title(title_suffix)
        ax.axhline(0, color="black", linewidth=1.0)
        ax.grid(axis="y", alpha=0.3)
        remove_spines(ax)

        if ax == ax1:
            ax.set_ylabel("Accuracy delta (%)")

    ax1.legend(loc="upper right")
    fig.suptitle(r"Cosine vs MSE $L_{stab}$: Effect of Rate Matching (Phi-3.5, L15-19, stab=24, $\lambda$=3.0)")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "c7_original_vs_matched_rate.pdf")
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================================
# C5b: Matched-rate taxonomy split
# ============================================================================
def figure_c5b():
    print("Generating C5b: Taxonomy Split (Matched Rate)...")
    cosine_path = os.path.join(ROOT, "results/bootstrap/phi_matched_cosine_stab24/paired_bootstrap_results.json")
    mse_path = os.path.join(ROOT, "results/bootstrap/phi_matched_mse_stab24/paired_bootstrap_results.json")

    if not os.path.exists(cosine_path) or not os.path.exists(mse_path):
        print("  Missing matched-rate bootstrap data, skipping.")
        return

    with open(cosine_path) as f:
        cos = json.load(f)
    with open(mse_path) as f:
        mse = json.load(f)

    ordered_conds = []
    for m in PERTURBATION_ORDER:
        for k in cos:
            if k.startswith(m + "_") and not k.startswith("_") and k != "clean_baseline":
                ordered_conds.append(k)
                break

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(ordered_conds))
    w = 0.35

    cos_vals = [cos[c]["delta_pct"] for c in ordered_conds]
    mse_vals = [mse[c]["delta_pct"] for c in ordered_conds]
    cos_errs = [[cos[c]["delta_pct"] - cos[c]["ci_lo_pct"] for c in ordered_conds],
                [cos[c]["ci_hi_pct"] - cos[c]["delta_pct"] for c in ordered_conds]]
    mse_errs = [[mse[c]["delta_pct"] - mse[c]["ci_lo_pct"] for c in ordered_conds],
                [mse[c]["ci_hi_pct"] - mse[c]["delta_pct"] for c in ordered_conds]]

    ax.bar(x - w/2, cos_vals, w, color=C_COSINE, label="Cosine (LRD)",
           edgecolor="black", linewidth=0.8, yerr=cos_errs, capsize=4)
    ax.bar(x + w/2, mse_vals, w, color=C_MSE, label="MSE",
           edgecolor="black", linewidth=0.8, yerr=mse_errs, capsize=4)

    n_dir = len([c for c in ordered_conds if method_from_cond(c) in DIRECTIONAL_SET])
    sep_x = n_dir - 0.5
    ax.axvline(sep_x, color="gray", linewidth=1.5, linestyle="--")

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in ordered_conds])
    ax.set_ylabel("Accuracy delta (%)")
    ax.set_title(r"Per-Condition: Cosine vs MSE (Matched Rate 10%, L15-19, stab=24, $\lambda$=3.0)")
    ax.axhline(0, color="black", linewidth=1.0)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    remove_spines(ax)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "c5b_taxonomy_split_matched.pdf")
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================================
# C8: Interaction Forest Plot
# ============================================================================
def figure_c8():
    """Forest plot showing cos-MSE difference per condition, grouped by taxonomy."""
    print("Generating C8: Interaction Forest Plot...")

    data_configs = {
        "Original rates": {
            "cosine": os.path.join(ROOT, "results/bootstrap/cosv2_L15_stab24_high/paired_bootstrap_results.json"),
            "mse": os.path.join(ROOT, "results/bootstrap/mse_L15_stab24_high/paired_bootstrap_results.json"),
        },
        "Matched (10%)": {
            "cosine": os.path.join(ROOT, "results/bootstrap/phi_matched_cosine_stab24/paired_bootstrap_results.json"),
            "mse": os.path.join(ROOT, "results/bootstrap/phi_matched_mse_stab24/paired_bootstrap_results.json"),
        },
    }

    for rate_label, paths in data_configs.items():
        for p in paths.values():
            if not os.path.exists(p):
                print(f"  Missing: {p}")
                return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)

    for ax, (rate_label, paths) in zip([ax1, ax2], data_configs.items()):
        with open(paths["cosine"]) as f:
            cos = json.load(f)
        with open(paths["mse"]) as f:
            mse = json.load(f)

        conds_dir, conds_uni = [], []
        for m in PERTURBATION_ORDER:
            for k in cos:
                if k.startswith(m + "_") and not k.startswith("_") and k != "clean_baseline":
                    if m in DIRECTIONAL_SET:
                        conds_dir.append(k)
                    else:
                        conds_uni.append(k)
                    break

        all_conds = conds_dir + conds_uni
        diffs, ci_los, ci_his, colors = [], [], [], []

        rng = np.random.RandomState(42)
        for c in all_conds:
            cos_s = np.array(cos[c]["per_sample_with_adapter"])
            mse_s = np.array(mse[c]["per_sample_with_adapter"])
            imp = cos_s - mse_s
            point = imp.mean() * 100

            boots = [imp[rng.randint(0, len(imp), len(imp))].mean() * 100 for _ in range(10000)]
            lo = np.percentile(boots, 2.5)
            hi = np.percentile(boots, 97.5)

            diffs.append(point)
            ci_los.append(point - lo)
            ci_his.append(hi - point)
            colors.append(C_DIRECTIONAL if method_from_cond(c) in DIRECTIONAL_SET else C_UNIFORM)

        y = np.arange(len(all_conds))
        ax.errorbar(diffs, y, xerr=[ci_los, ci_his], fmt="none", color="gray",
                    capsize=4, linewidth=1.5)
        for i, (d, c) in enumerate(zip(diffs, colors)):
            ax.plot(d, i, "o", color=c, markersize=10, zorder=5)

        ax.axvline(0, color="black", linewidth=1.0, linestyle="--")
        ax.set_yticks(y)
        ax.set_yticklabels([c.replace("_", " ") for c in all_conds])
        ax.set_title(rate_label)
        ax.grid(axis="x", alpha=0.3)
        remove_spines(ax)

        ax.axhline(len(conds_dir) - 0.5, color="gray", linewidth=0.8, linestyle=":")

    legend_elements = [Patch(facecolor=C_DIRECTIONAL, edgecolor="black", label="Directional"),
                      Patch(facecolor=C_UNIFORM, edgecolor="black", label="Uniform")]
    ax1.legend(handles=legend_elements, loc="lower left")

    fig.suptitle("Cosine vs MSE Advantage by Perturbation Type and Rate Condition")
    fig.subplots_adjust(bottom=0.12)
    fig.supxlabel(r'Cosine $-$ MSE advantage (%)', fontsize=12, y=0.02)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    path = os.path.join(OUTPUT_DIR, "interaction_forest.pdf")
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    figure_c1()
    figure_c2()
    figure_c3()
    figure_c4()
    figure_c5()
    figure_c5b()
    figure_c6()
    figure_c7()
    figure_c8()
    print(f"\nAll figures saved to: {OUTPUT_DIR}/")
