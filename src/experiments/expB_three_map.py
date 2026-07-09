# === lib path shim (added by repo reorg) ===
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir, 'lib'))
# === end lib path shim ===

"""
expB_three_map.py - Three-Map Overlay

Directly visualises and quantifies the dissociation between:
  1. LRD magnitude         - mean per-layer cosine distance across all conditions
  2. Patching recovery     - fraction of failed examples recovered by clean patch at L
  3. LoRA effectiveness    - average perturbed-accuracy delta for window centred on L

For each model, normalises all three signals to [0, 1] and overlays them on
shared axes. Computes pairwise Spearman correlations and the "slack" composite
score (1 - norm_LRD) x (1 - norm_patching) correlated against LoRA delta.

Data sources (all pre-existing):
  LRD     - models/phi3.5/lrd_results/phi_hardening2/raw_gsm8k.json
             models/llama/lrd_results/raw_gsm8k.json
             models/mistral/lrd_results/mistral_gsm8k/raw_gsm8k.json
  Patching - models/phi3.5/lrd_results/phi_hardening2/patching_gsm8k_typos.json
              models/llama/lrd_results/patching_gsm8k_typos.json
              models/mistral/lrd_results/mistral_patching_nocache_v2/patching_gsm8k_ocr.json
  LoRA     - stabilizer_weights/{model}_sweep_L{window}/eval_results.json
             (Phi3.5 uses p2_sweep_L* directories)

Output:
  expB_three_map_overlay.json
  figures/three_map_phi35.pdf
  figures/three_map_llama3.pdf
  figures/three_map_mistral.pdf
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.interpolate import interp1d

# Apply publication styling
from figure_style import (apply_style, remove_spines, COLORS, LINE_STYLES,
                          add_correlation_legend)
apply_style()

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SW   = os.path.join(ROOT, "stabilizer_weights")
FIG_DIR = os.path.join(ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

MODELS = {
    "Phi3.5": {
        "lrd_path":     "models/phi3.5/lrd_results/phi_hardening2/raw_gsm8k.json",
        "patch_path":   "models/phi3.5/lrd_results/phi_patching_v3/patching_gsm8k_typos.json",
        "patch_note":   "typos",
        "sweep_dirs":   [
            ("p2_sweep_L00_04", (0, 4)),
            ("p2_sweep_L05_09", (5, 9)),
            ("p2_sweep_L10_14", (10, 14)),
            ("p2_sweep_L15_19", (15, 19)),
            ("p2_sweep_L20_24", (20, 24)),
            ("p2_sweep_L27_31", (27, 31)),
        ],
    },
    "Llama3": {
        "lrd_path":     "models/llama/lrd_results/raw_gsm8k.json",
        "patch_path":   "models/llama/lrd_results/patching_gsm8k_typos.json",
        "patch_note":   "typos",
        "sweep_dirs":   [
            ("llama_sweep_L00_04", (0, 4)),
            ("llama_sweep_L05_09", (5, 9)),
            ("llama_sweep_L10_14", (10, 14)),
            ("llama_sweep_L15_19", (15, 19)),
            ("llama_sweep_L20_24", (20, 24)),
            ("llama_sweep_L27_31", (27, 31)),
        ],
    },
    "Mistral": {
        "lrd_path":     "models/mistral/lrd_results/mistral_gsm8k/raw_gsm8k.json",
        "patch_path":   "models/mistral/lrd_results/mistral_patching_nocache_v2/patching_gsm8k_ocr.json",
        "patch_note":   "OCR (only available)",
        "sweep_dirs":   [
            ("mistral_sweep_L00_04", (0, 4)),
            ("mistral_sweep_L05_09", (5, 9)),
            ("mistral_sweep_L10_14", (10, 14)),
            ("mistral_sweep_L15_19", (15, 19)),
            ("mistral_sweep_L20_24", (20, 24)),
            ("mistral_sweep_L27_31", (27, 31)),
        ],
    },
}


def normalize_01(arr: np.ndarray) -> np.ndarray:
    lo  = np.nanmin(arr)
    hi  = np.nanmax(arr)
    if hi - lo < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def mean_lrd_per_layer(raw: dict) -> np.ndarray:
    """Average lrd_profile across all conditions and examples."""
    all_profiles = []
    for records in raw.values():
        for r in records:
            all_profiles.append(r["lrd_profile"])
    arr = np.array(all_profiles)
    return arr.mean(axis=0)


def patching_recovery_per_layer(patch: dict) -> np.ndarray:
    """Mean recovery fraction at each layer (layer_recovery dict)."""
    lr = patch["layer_recovery"]
    n_layers = max(int(k) for k in lr) + 1
    out = np.zeros(n_layers)
    for k, vals in lr.items():
        out[int(k)] = float(np.mean(vals))
    return out


def lora_delta_per_layer(sweep_dirs: list, n_layers: int) -> np.ndarray:
    """
    For each sweep window, compute mean delta accuracy across all perturbation
    conditions, then interpolate to a per-layer curve.

    Window centre = (lo + hi) / 2, rounded to int.
    Interpolation: linear between window centres, clamped at edges.
    """
    centres, deltas = [], []
    for sweep_name, (lo, hi) in sweep_dirs:
        ep = os.path.join(SW, sweep_name, "eval_results.json")
        if not os.path.exists(ep):
            print(f"  [WARN] Missing sweep: {ep}")
            continue
        with open(ep) as f:
            ev = json.load(f)
        vals = [v["delta"] for k, v in ev.items()
                if k != "clean_baseline" and "delta" in v]
        if not vals:
            continue
        centres.append((lo + hi) / 2.0)
        deltas.append(float(np.mean(vals)))

    if len(centres) < 2:
        print("  [WARN] Fewer than 2 sweep windows - returning zeros for LoRA")
        return np.zeros(n_layers)

    centres = np.array(centres)
    deltas  = np.array(deltas)
    layer_x = np.arange(n_layers, dtype=float)

    interp = interp1d(centres, deltas, kind="linear",
                      bounds_error=False, fill_value=(deltas[0], deltas[-1]))
    return interp(layer_x)


def spearman(a: np.ndarray, b: np.ndarray) -> dict:
    rho, pval = spearmanr(a, b)
    return {"rho": round(float(rho), 4), "p": round(float(pval), 4)}


def process_model(model_name: str, cfg: dict) -> dict:
    print(f"\n=== {model_name} ===")

    lrd_path   = os.path.join(ROOT, cfg["lrd_path"])
    patch_path = os.path.join(ROOT, cfg["patch_path"])

    if not os.path.exists(lrd_path):
        print(f"  [SKIP] LRD data not found: {lrd_path}")
        return {}
    if not os.path.exists(patch_path):
        print(f"  [SKIP] Patching data not found: {patch_path}")
        return {}

    with open(lrd_path) as f:
        raw = json.load(f)
    with open(patch_path) as f:
        patch = json.load(f)

    # Signal 1: LRD mean per layer
    lrd_mean   = mean_lrd_per_layer(raw)
    n_layers   = len(lrd_mean)

    # Signal 2: patching recovery per layer
    recovery   = patching_recovery_per_layer(patch)
    if len(recovery) < n_layers:
        pad      = np.full(n_layers - len(recovery), np.nan)
        recovery = np.concatenate([recovery, pad])
    recovery = recovery[:n_layers]

    # Signal 3: LoRA delta interpolated per layer
    lora_delta = lora_delta_per_layer(cfg["sweep_dirs"], n_layers)

    # Normalise to [0, 1]
    norm_lrd   = normalize_01(lrd_mean)
    norm_patch = normalize_01(recovery)
    norm_lora  = normalize_01(lora_delta)

    # Correlations (on layers where recovery is valid)
    valid = ~np.isnan(recovery)
    corr = {
        "LRD vs Patch": spearman(lrd_mean[valid], recovery[valid]),
        "LRD vs LoRA":  spearman(lrd_mean[valid], lora_delta[valid]),
        "Patch vs LoRA": spearman(recovery[valid], lora_delta[valid]),
    }
    print(f"  Spearman correlations:")
    for k, v in corr.items():
        print(f"    {k}: rho={v['rho']:+.3f}, p={v['p']:.4f}")

    # Composite slack score
    slack = (1 - norm_lrd[valid]) * (1 - norm_patch[valid])
    slack_vs_lora = spearman(slack, norm_lora[valid])
    print(f"  Slack x LoRA: rho={slack_vs_lora['rho']:+.3f}, "
          f"p={slack_vs_lora['p']:.4f}")

    # Figure with distinct line styles for B&W/colorblind accessibility
    layers = np.arange(n_layers)
    fig, ax = plt.subplots(figsize=(11, 5.5))

    # Plot with distinct colors AND line styles
    ax.plot(layers, norm_lrd,
            label="LRD (norm)",
            color=COLORS['lrd'],
            linestyle='-',       # solid
            linewidth=2.5)

    ax.plot(layers[valid], norm_patch[valid],
            label=f"Patching recovery (norm, {cfg['patch_note']})",
            color=COLORS['patching'],
            linestyle='--',      # dashed
            linewidth=2.5)

    ax.plot(layers, norm_lora,
            label="LoRA delta acc (norm, interpolated)",
            color=COLORS['lora'],
            linestyle=':',       # dotted
            linewidth=2.5)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Normalized score (per-layer)")
    ax.set_title(f"Three-Map Overlay - {model_name} / GSM8K")

    # Set y-axis minimum slightly below 0 to make near-zero lines visible
    ax.set_ylim(-0.05, 1.05)

    remove_spines(ax)

    # Light y-axis grid
    ax.grid(True, axis='y', alpha=0.3, linewidth=0.5)

    # Legend with bumped font size
    ax.legend(loc='lower left', framealpha=0.9, fontsize=12)

    # Correlation values in upper-right (per user request)
    add_correlation_legend(ax, corr, loc='upper right', fontsize=12)

    plt.tight_layout()
    model_key = model_name.lower().replace(".", "")
    fig_path = os.path.join(FIG_DIR, f"three_map_{model_key}.pdf")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> figure: {fig_path}")

    return {
        "n_layers":    n_layers,
        "lrd_mean":    lrd_mean.tolist(),
        "recovery":    [None if np.isnan(x) else float(x) for x in recovery],
        "lora_delta":  lora_delta.tolist(),
        "norm_lrd":    norm_lrd.tolist(),
        "norm_patch":  [None if np.isnan(x) else float(x) for x in norm_patch],
        "norm_lora":   norm_lora.tolist(),
        "correlations": corr,
        "slack_vs_lora": slack_vs_lora,
        "patch_note":  cfg["patch_note"],
    }


def main():
    out = {}
    for model_name, cfg in MODELS.items():
        result = process_model(model_name, cfg)
        if result:
            out[model_name] = result

    out_path = os.path.join(ROOT, "results", "expB_three_map_overlay.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
