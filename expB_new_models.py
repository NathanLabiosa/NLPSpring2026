"""
expB_new_models.py — Three-Map Overlay for TinyLlama, Gemma-2-2B, Qwen2.5-7B.

Same analysis as expB_three_map.py but configured for the three new models.
Run this AFTER all three pipeline stages complete for each model:
  1. lrd_results/      (from lrd_run.py)
  2. lrd_results/      (patching from patch_run.py)
  3. stabilizer_weights/{model}_sweep_L*/  (from submit_sweep.slurm)

Also computes block-bootstrap 95% CIs (block_b=5) for each correlation,
and appends results to bootstrap_block_results.json.

Output:
    expB_three_map_TinyLlama.pdf
    expB_three_map_Gemma2.pdf
    expB_three_map_Qwen25.pdf
    expB_new_models_overlay.json      (same format as expB_three_map_overlay.json)
"""

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


def compute_windows(n_layers: int, n_windows: int = 6):
    """Divide n_layers into n_windows equal non-overlapping windows."""
    base  = n_layers // n_windows
    extra = n_layers % n_windows
    wins  = []
    start = 0
    for i in range(n_windows):
        size = base + (1 if i < extra else 0)
        wins.append((start, start + size - 1))
        start += size
    return wins


NEW_MODELS = {
    "TinyLlama": {
        "lrd_path":   "TinyLlama/lrd_results/tinyllama_gsm8k/raw_gsm8k.json",
        "patch_path": "TinyLlama/lrd_results/tinyllama_patching/patching_gsm8k_typos.json",
        "patch_note": "typos",
        "n_layers":   22,
        "sweep_prefix": "tinyllama_sweep",
        "sweep_windows": [  # (dir_suffix, (lo, hi)) — must match submit_sweep.slurm
            ("L00_03", (0,  3)),
            ("L04_07", (4,  7)),
            ("L08_11", (8,  11)),
            ("L12_15", (12, 15)),
            ("L16_18", (16, 18)),
            ("L19_21", (19, 21)),
        ],
    },
    "Gemma2": {
        "lrd_path":   "Gemma2/lrd_results/gemma2_gsm8k/raw_gsm8k.json",
        "patch_path": "Gemma2/lrd_results/gemma2_patching/patching_gsm8k_typos.json",
        "patch_note": "typos",
        "n_layers":   26,
        "sweep_prefix": "gemma2_sweep",
        "sweep_windows": [
            ("L00_04", (0,  4)),
            ("L05_09", (5,  9)),
            ("L10_13", (10, 13)),
            ("L14_17", (14, 17)),
            ("L18_21", (18, 21)),
            ("L22_25", (22, 25)),
        ],
    },
    "Qwen25": {
        "lrd_path":   "Qwen2.5/lrd_results/qwen_gsm8k/raw_gsm8k.json",
        "patch_path": "Qwen2.5/lrd_results/qwen_patching/patching_gsm8k_typos.json",
        "patch_note": "typos",
        "n_layers":   28,
        "sweep_prefix": "qwen_sweep",
        "sweep_windows": [
            ("L00_04", (0,  4)),
            ("L05_09", (5,  9)),
            ("L10_14", (10, 14)),
            ("L15_19", (15, 19)),
            ("L20_23", (20, 23)),
            ("L24_27", (24, 27)),
        ],
    },
}


def normalize_01(arr: np.ndarray) -> np.ndarray:
    lo, hi = np.nanmin(arr), np.nanmax(arr)
    if hi - lo < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def mean_lrd_per_layer(raw: dict) -> np.ndarray:
    all_p = [r["lrd_profile"] for recs in raw.values() for r in recs]
    return np.array(all_p).mean(0)


def patching_recovery_per_layer(patch: dict, n_layers: int) -> np.ndarray:
    lr  = patch["layer_recovery"]
    out = np.full(n_layers, np.nan)
    for k, vals in lr.items():
        ki = int(k)
        if ki < n_layers:
            out[ki] = float(np.mean(vals))
    return out


def lora_delta_per_layer(sweep_prefix: str, sweep_windows: list,
                         n_layers: int) -> np.ndarray:
    centres, deltas = [], []
    for suffix, (lo, hi) in sweep_windows:
        ep = os.path.join(SW, f"{sweep_prefix}_{suffix}", "eval_results.json")
        if not os.path.exists(ep):
            print(f"  [WARN] Missing: {ep}")
            continue
        ev   = json.load(open(ep))
        vals = [v["delta"] for k, v in ev.items()
                if k != "clean_baseline" and "delta" in v]
        if not vals:
            continue
        centres.append((lo + hi) / 2.0)
        deltas.append(float(np.mean(vals)))

    if len(centres) < 2:
        print("  [WARN] < 2 sweep windows found — returning zeros")
        return np.zeros(n_layers)

    interp = interp1d(centres, deltas, kind="linear",
                      bounds_error=False,
                      fill_value=(deltas[0], deltas[-1]))
    return interp(np.arange(n_layers, dtype=float))


def block_bootstrap_spearman(x, y, block_b=5, n_boot=10_000,
                             rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(x)
    rho_pt, _ = spearmanr(x, y)
    max_start  = n - block_b
    n_blocks   = math.ceil(n / block_b)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        idx    = np.concatenate([np.arange(s, s + block_b) for s in starts])[:n]
        boots[i], _ = spearmanr(x[idx], y[idx])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    frac_neg = float((boots < 0).mean())
    p_two    = 2.0 * min(frac_neg, 1.0 - frac_neg)
    return float(rho_pt), float(lo), float(hi), float(p_two)


def process_model(model_name: str, cfg: dict, rng) -> dict:
    print(f"\n=== {model_name} ===")
    lrd_path   = os.path.join(ROOT, cfg["lrd_path"])
    patch_path = os.path.join(ROOT, cfg["patch_path"])

    if not os.path.exists(lrd_path):
        print(f"  [SKIP] LRD data missing: {lrd_path}")
        return {}
    if not os.path.exists(patch_path):
        print(f"  [SKIP] Patching data missing: {patch_path}")
        return {}

    raw   = json.load(open(lrd_path))
    patch = json.load(open(patch_path))
    n_L   = cfg["n_layers"]

    lrd_mean = mean_lrd_per_layer(raw)
    # If model has different layer count, warn
    if len(lrd_mean) != n_L:
        print(f"  [WARN] Expected {n_L} layers, got {len(lrd_mean)} — using actual")
        n_L = len(lrd_mean)

    recovery   = patching_recovery_per_layer(patch, n_L)
    lora_delta = lora_delta_per_layer(cfg["sweep_prefix"],
                                      cfg["sweep_windows"], n_L)

    # Normalise
    norm_lrd   = normalize_01(lrd_mean)
    norm_patch = normalize_01(recovery)
    norm_lora  = normalize_01(lora_delta)

    # ── Correlations with block bootstrap CIs ────────────────────────────────
    valid = ~np.isnan(recovery)
    pairs = [
        ("LRD_vs_Patching", lrd_mean[valid],   recovery[valid]),
        ("LRD_vs_LoRA",     lrd_mean[valid],   lora_delta[valid]),
        ("Patch_vs_LoRA",   recovery[valid],   lora_delta[valid]),
    ]

    corr = {}
    for pair_name, xa, ya in pairs:
        rho, lo, hi, pval = block_bootstrap_spearman(xa, ya, rng=rng)
        flag = "***" if pval < 0.001 else ("**" if pval < 0.01 else
               ("*" if pval < 0.05 else " "))
        corr[pair_name] = {"rho": round(rho, 4), "ci_low": round(lo, 4),
                           "ci_high": round(hi, 4), "p_block": round(pval, 4),
                           "n": int(valid.sum())}
        print(f"  {pair_name}: ρ={rho:+.3f} [{lo:+.3f},{hi:+.3f}] "
              f"p={pval:.4f}{flag}")

    # ── Identity recovery check ───────────────────────────────────────────────
    id_rec = patch.get("identity_recovery", {})
    ceiling = float(np.mean(list(id_rec.values()))) * 100 if id_rec else None
    if ceiling is not None:
        flag = "PASSES" if ceiling >= 80 else "EXPLORATORY (ceiling < 80%)"
        print(f"  Identity recovery ceiling: {ceiling:.1f}%  → {flag}")

    # ── Figure ───────────────────────────────────────────────────────────────
    layers = np.arange(n_L)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(layers, norm_lrd,   label="LRD (norm)",          color="tab:red",    lw=2)
    ax.plot(layers[valid], norm_patch[valid],
            label=f"Patching recovery ({cfg['patch_note']})", color="tab:orange", lw=2)
    ax.plot(layers, norm_lora,  label="LoRA Δ acc (norm)",   color="tab:blue",   lw=2, ls="--")
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Normalised signal [0, 1]", fontsize=11)
    ax.set_title(f"Three-Map Overlay — {model_name} / GSM8K", fontsize=13)
    ax.legend(fontsize=9)

    corr_txt = "\n".join([f"{k}: $\\rho$={v['rho']:+.3f} [{v['ci_low']:+.3f},{v['ci_high']:+.3f}]"
                          for k, v in corr.items()])
    ax.text(0.02, 0.97, corr_txt, transform=ax.transAxes, fontsize=7.5,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    plt.tight_layout()
    fig_path = os.path.join(ROOT, f"expB_three_map_{model_name}.pdf")
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"  → figure: {fig_path}")

    return {
        "n_layers":   n_L,
        "lrd_mean":   lrd_mean.tolist(),
        "norm_lrd":   norm_lrd.tolist(),
        "norm_patch": norm_patch.tolist(),
        "norm_lora":  norm_lora.tolist(),
        "lora_delta": lora_delta.tolist(),
        "correlations": corr,
        "identity_recovery_ceiling": ceiling,
    }


def main():
    rng  = np.random.default_rng(42)
    out  = {}
    skipped = []

    for model_name, cfg in NEW_MODELS.items():
        result = process_model(model_name, cfg, rng)
        if result:
            out[model_name] = result
        else:
            skipped.append(model_name)

    if skipped:
        print(f"\n[Skipped — data not yet available]: {skipped}")
        print("Re-run after SLURM jobs complete.")

    if out:
        out_path = os.path.join(ROOT, "expB_new_models_overlay.json")
        json.dump(out, open(out_path, "w"), indent=2)
        print(f"\nSaved: {out_path}")

        # ── Regime classification ─────────────────────────────────────────────
        print("\n=== Regime Classification ===")
        for model, data in out.items():
            lrd = np.array(data["lrd_mean"])
            n   = len(lrd)
            q   = max(1, n // 4)
            early = lrd[:q].mean()
            late  = lrd[-q:].mean()
            max_l = int(lrd.argmax())
            ratio = late / early if early > 1e-8 else 1.0

            if max_l < n // 2 and ratio < 0.7:
                regime = "spike-and-suppress"
            else:
                regime = "late-accumulation"

            corrs = data["correlations"]
            lrd_lora_rho = corrs["LRD_vs_LoRA"]["rho"]
            sign = "NEGATIVE (spike-and-suppress sign)" if lrd_lora_rho < 0 else \
                   "POSITIVE (late-accumulation sign)"

            print(f"  {model:<12}: regime={regime}  LRD×LoRA ρ={lrd_lora_rho:+.3f} → {sign}")
    else:
        print("\nNo models processed. Run SLURM jobs first.")


if __name__ == "__main__":
    main()
