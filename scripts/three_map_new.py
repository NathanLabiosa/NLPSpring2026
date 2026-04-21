

import json
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.interpolate import interp1d

ROOT = os.path.dirname(os.path.abspath(__file__))
SW   = os.path.join(ROOT, "stabilizer_weights")

#Model configurations

MODELS = {
    "Gemma2_9B": {
        "lrd_path":   "Gemma2/lrd_results/gemma2_9b_gsm8k/raw_gsm8k.json",
        "patch_path": "Gemma2/lrd_results/gemma2_9b_patching/patching_gsm8k_typos.json",
        "patch_note": "typos",
        "sweep_dirs": [
            ("gemma2_9b_sweep_L00_05", (0,  5)),
            ("gemma2_9b_sweep_L06_11", (6,  11)),
            ("gemma2_9b_sweep_L12_17", (12, 17)),
            ("gemma2_9b_sweep_L18_23", (18, 23)),
            ("gemma2_9b_sweep_L24_29", (24, 29)),
            ("gemma2_9b_sweep_L30_35", (30, 35)),
            ("gemma2_9b_sweep_L36_41", (36, 41)),
        ],
    },
    "Qwen25_7B": {
        "lrd_path":   "Qwen2.5/lrd_results/qwen_gsm8k/raw_gsm8k.json",
        "patch_path": "Qwen2.5/lrd_results/qwen_patching/patching_gsm8k_typos.json",
        "patch_note": "typos",
        "sweep_dirs": [
            ("qwen_sweep_L00_03", (0,  3)),
            ("qwen_sweep_L04_07", (4,  7)),
            ("qwen_sweep_L08_11", (8,  11)),
            ("qwen_sweep_L12_15", (12, 15)),
            ("qwen_sweep_L16_19", (16, 19)),
            ("qwen_sweep_L20_23", (20, 23)),
            ("qwen_sweep_L24_27", (24, 27)),
        ],
    },
    "TinyLlama_1B": {
        "lrd_path":   "TinyLlama/lrd_results/tinyllama_gsm8k/raw_gsm8k.json",
        "patch_path": "TinyLlama/lrd_results/tinyllama_patching/patching_gsm8k_typos.json",
        "patch_note": "typos",
        "sweep_dirs": [
            ("tinyllama_sweep_L00_03", (0,  3)),
            ("tinyllama_sweep_L04_07", (4,  7)),
            ("tinyllama_sweep_L08_11", (8,  11)),
            ("tinyllama_sweep_L12_15", (12, 15)),
            ("tinyllama_sweep_L16_19", (16, 19)),
        ],
    },
}


# Helpers

def normalize_01(arr):
    lo, hi = np.nanmin(arr), np.nanmax(arr)
    if hi - lo < 1e-12:  # avoid div-by-zero for constant signals
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def mean_lrd_per_layer(raw):
    all_profiles = [r["lrd_profile"] for recs in raw.values() for r in recs]
    return np.array(all_profiles).mean(axis=0)


def patching_recovery_per_layer(patch):
    lr = patch["layer_recovery"]
    n_layers = max(int(k) for k in lr) + 1
    out = np.zeros(n_layers)
    for k, vals in lr.items():
        out[int(k)] = float(np.mean(vals))
    return out


def lora_delta_per_layer(sweep_dirs, n_layers):
    centres, deltas = [], []
    for sweep_name, (lo, hi) in sweep_dirs:
        ep = os.path.join(SW, sweep_name, "eval_results.json")
        if not os.path.exists(ep):
            print(f"  [WARN] Missing sweep: {ep}")
            continue
        ev = json.load(open(ep))
        vals = [v["delta"] for k, v in ev.items()
                if k != "clean_baseline" and "delta" in v]
        if vals:
            centres.append((lo + hi) / 2.0)  # window midpoint
            deltas.append(float(np.mean(vals)))

    if len(centres) < 2:  # need at least 2 points for interpolation
        return np.zeros(n_layers)

    centres = np.array(centres)
    deltas = np.array(deltas)
    # linear interpolation between window midpoints to get per-layer estimates
    interp = interp1d(centres, deltas, kind="linear",
                      bounds_error=False, fill_value=(deltas[0], deltas[-1]))
    return interp(np.arange(n_layers, dtype=float))


# Block Bootstrap

def block_bootstrap_spearman(x, y, block_size=5, n_boot=10000, seed=42):
    """Block bootstrap to account for layer-wise autocorrelation."""
    rng = np.random.default_rng(seed)
    n = len(x)
    rho_point, _ = spearmanr(x, y)

    max_start = n - block_size
    n_blocks = math.ceil(n / block_size)

    boot_rhos = np.empty(n_boot)
    for i in range(n_boot):
        # sample random block starts with replacement
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        idx = []
        for s in starts:
            idx.extend(range(s, s + block_size))
        idx = np.array(idx[:n])  # trim to original length
        rho_b, _ = spearmanr(x[idx], y[idx])
        boot_rhos[i] = rho_b

    ci_low, ci_high = np.percentile(boot_rhos, [2.5, 97.5])
    frac_neg = float((boot_rhos < 0).mean())
    p_two = 2.0 * min(frac_neg, 1.0 - frac_neg)  # two-tailed p-value

    return {
        "rho": round(float(rho_point), 4),
        "ci_low": round(float(ci_low), 4),
        "ci_high": round(float(ci_high), 4),
        "p_two": round(float(p_two), 4),
        "block_size": block_size,
        "n_boot": n_boot,
    }


# Autocorrelation

def compute_autocorrelation(signal, max_lag=8):
    """Compute ACF, decorrelation lag (1/e threshold), VIF, and effective N."""
    n = len(signal)
    s = signal - signal.mean()
    var = float((s ** 2).mean())
    if var < 1e-12:  # constant signal edge case
        return {"acf": [1.0], "decorrelation_lag": 0, "vif": 1.0, "effective_n": float(n)}

    acf = [1.0]
    for k in range(1, min(max_lag + 1, n)):
        acf.append(float((s[:n - k] * s[k:]).mean()) / var)

    # Decorrelation lag: first lag where |ACF| < 1/e
    decor = max_lag
    for k in range(1, len(acf)):
        if abs(acf[k]) < 1.0 / np.e:  # 1/e ≈ 0.368
            decor = k
            break

    # VIF = 1 + 2*sum(positive ACF lags)
    trunc = 0
    for k in range(1, len(acf)):
        if acf[k] > 0:
            trunc = k
        else:
            break
    vif = max(1.0, 1.0 + 2.0 * sum(acf[1:trunc + 1]))
    eff_n = n / vif  # effective degrees of freedom

    return {
        "acf": [round(v, 4) for v in acf],
        "decorrelation_lag": decor,
        "vif": round(vif, 3),
        "effective_n": round(eff_n, 1),
    }


# Main

def process_model(model_name, cfg):
    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"{'='*60}")

    lrd_path = os.path.join(ROOT, cfg["lrd_path"])
    patch_path = os.path.join(ROOT, cfg["patch_path"])

    if not os.path.exists(lrd_path):
        print(f"  [SKIP] LRD data not found: {lrd_path}")
        return None
    if not os.path.exists(patch_path):
        print(f"  [SKIP] Patching data not found: {patch_path}")
        return None

    raw = json.load(open(lrd_path))
    patch = json.load(open(patch_path))

    lrd_mean = mean_lrd_per_layer(raw)
    n_layers = len(lrd_mean)

    recovery = patching_recovery_per_layer(patch)
    if len(recovery) < n_layers:
        recovery = np.concatenate([recovery, np.full(n_layers - len(recovery), np.nan)])
    recovery = recovery[:n_layers]

    lora_delta = lora_delta_per_layer(cfg["sweep_dirs"], n_layers)

    norm_lrd = normalize_01(lrd_mean)
    norm_patch = normalize_01(recovery)
    norm_lora = normalize_01(lora_delta)

    valid = ~np.isnan(recovery)

    #Block bootstrap correlations
    print("\n  Block Bootstrap Correlations (b=5, 10k resamples):")
    # compare all three signals pairwise
    pairs = {
        "LRD_vs_LoRA":      (lrd_mean[valid], lora_delta[valid]),
        "Patch_vs_LoRA":     (recovery[valid], lora_delta[valid]),
        "LRD_vs_Patching":   (lrd_mean[valid], recovery[valid]),
    }

    corr_results = {}
    for pair_name, (x, y) in pairs.items():
        result = block_bootstrap_spearman(x, y)  # block size=5, 10k resamples
        corr_results[pair_name] = result
        sig = "*" if result["p_two"] < 0.05 else ""
        print(f"    {pair_name}: rho={result['rho']:+.3f}  "
              f"CI=[{result['ci_low']:+.3f}, {result['ci_high']:+.3f}]  "
              f"p={result['p_two']:.4f}{sig}")

    # Autocorrelation
    # TODO: might want to export ACF plots per model for supplemental figures
    print("\n  Autocorrelation Structure:")
    autocorr = {}
    for sig_name, arr in [("LRD", norm_lrd), ("Patching", norm_patch[valid]),
                           ("LoRA", norm_lora)]:
        ac = compute_autocorrelation(arr)  # decorrelation lag, VIF, effective N
        autocorr[sig_name] = ac
        print(f"    {sig_name}: decor_lag={ac['decorrelation_lag']}  "
              f"VIF={ac['vif']:.2f}  eff_N={ac['effective_n']:.1f}")

    # Three-map figure 
    layers = np.arange(n_layers)
    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(layers, norm_lrd, label="LRD (norm)", color="tab:red", linewidth=2.0)
    ax.plot(layers[valid], norm_patch[valid],
            label=f"Patching recovery (norm, {cfg['patch_note']})",
            color="tab:orange", linewidth=2.0)
    # LoRA is interpolated from window midpoints, so use dashed line
    ax.plot(layers, norm_lora, label="LoRA delta acc (norm, interpolated)",
            color="tab:blue", linewidth=2.0, linestyle="--")

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Normalised signal [0, 1]", fontsize=11)
    ax.set_title(f"Three-Map Overlay - {model_name} / GSM8K", fontsize=13)
    ax.legend(fontsize=9)

    txt = "\n".join([f"{k}: rho={v['rho']:+.3f} (p={v['p_two']:.3f})"
                     for k, v in corr_results.items()])
    ax.text(0.02, 0.97, txt, transform=ax.transAxes, fontsize=8,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    fig_path = os.path.join(ROOT, f"{model_name}_three_map.pdf")
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"\n  Figure: {fig_path}")

    # Save results
    three_map = {
        "n_layers": n_layers,
        "correlations": corr_results,
        "lrd_mean": lrd_mean.tolist(),
        "recovery": [None if np.isnan(x) else float(x) for x in recovery],
        "lora_delta": lora_delta.tolist(),
    }
    tm_path = os.path.join(ROOT, f"{model_name}_three_map.json")
    json.dump(three_map, open(tm_path, "w"), indent=2)
    print(f"  Three-map: {tm_path}")

    ac_path = os.path.join(ROOT, f"{model_name}_autocorrelation.json")
    json.dump(autocorr, open(ac_path, "w"), indent=2)
    print(f"  Autocorrelation: {ac_path}")

    return three_map


def check_predictions():
    """Compare C3/C4 predictions to actual sweep results for each model."""
    # This function runs post-hoc after all sweeps are complete
    print(f"\n{'='*60}")
    print("  Checking C3/C4 Predictions vs Actual")
    print(f"{'='*60}")

    pred_dir = os.path.join(ROOT, "predictions")
    if not os.path.exists(pred_dir):
        print("  No predictions directory found.")
        return

    for model_key in ["Gemma2", "Qwen25", "TinyLlama"]:
        pred_files = sorted([f for f in os.listdir(pred_dir)
                             if f.startswith(model_key + "_") and f.endswith(".json")])
        if not pred_files:
            print(f"\n  {model_key}: No prediction file found.")
            continue

        latest = os.path.join(pred_dir, pred_files[-1])  # use most recent prediction
        print(f"\n  {model_key}: Loading {pred_files[-1]}")
        record = json.load(open(latest))

        if "prediction" not in record:
            print(f"    No prediction data in file.")
            continue

        pred = record["prediction"]
        print(f"    Predicted best: {pred['predicted_best']}")

        # Load actual sweep results
        cfg_key = model_key
        if cfg_key not in MODELS:
            # Try mapping for filename consistency
            mapping = {"Gemma2": "Gemma2_9B", "Qwen25": "Qwen25_7B", "TinyLlama": "TinyLlama_1B"}
            cfg_key = mapping.get(model_key)
        if cfg_key and cfg_key in MODELS:
            cfg = MODELS[cfg_key]
            actual_deltas = {}
            for sweep_name, (lo, hi) in cfg["sweep_dirs"]:
                ep = os.path.join(SW, sweep_name, "eval_results.json")
                if os.path.exists(ep):
                    ev = json.load(open(ep))
                    # average delta across all perturbation types
                    vals = [v["delta"] for k, v in ev.items()
                            if k != "clean_baseline" and "delta" in v]
                    if vals:
                        actual_deltas[sweep_name] = round(float(np.mean(vals)), 3)

            if actual_deltas:
                actual_ranked = sorted(actual_deltas, key=lambda k: -actual_deltas[k])
                actual_best = actual_ranked[0]
                hit = (pred["predicted_best"] == actual_best)
                rank = actual_ranked.index(pred["predicted_best"]) + 1 if pred["predicted_best"] in actual_ranked else None
                print(f"    Actual best: {actual_best} (delta={actual_deltas[actual_best]:+.1f}%)")
                print(f"    Predicted window ranked: #{rank}")
                print(f"    Result: {'EXACT HIT' if hit else ('NEAR-HIT' if rank and rank <= 2 else 'MISS')}")
            else:
                print(f"    No sweep results available yet.")


def main():
    results = {}
    for model_name, cfg in MODELS.items():
        result = process_model(model_name, cfg)
        if result:
            results[model_name] = result

    if results:
        combined = os.path.join(ROOT, "three_map_new_overlay.json")
        json.dump(results, open(combined, "w"), indent=2)
        print(f"\nCombined results: {combined}")

    check_predictions()

    print("\nDone.")


if __name__ == "__main__":
    main()
