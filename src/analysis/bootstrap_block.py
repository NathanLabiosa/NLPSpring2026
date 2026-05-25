"""
bootstrap_block.py — Block Bootstrap for layer-wise Spearman correlations (W3).

Replaces the i.i.d. bootstrap with a block bootstrap that respects spatial
autocorrelation across the 32 transformer layers.

Method:
  - Block size b (default 5; sensitivity sweep over 2–8 also reported)
  - n_blocks = ceil(n / b) blocks drawn with replacement per resample
  - Concatenate blocks, trim to exactly n=32 layers
  - 10,000 resamples; CI = [2.5th, 97.5th] percentile
  - p-value: 2 * min(frac_below_zero, frac_above_zero) — two-sided

Effective N via variance inflation factor (VIF):
  VIF(signal) = 1 + 2 * sum_{k=1}^{K*} ACF(k)
    K* = max lag where ACF stays above 0 (Bartlett's rule)
  effective_N = n / VIF
  Reported per signal pair as min(eff_N_x, eff_N_y).

Input:  expB_three_map_overlay.json
Output: bootstrap_block_results.json       (CIs, p-values, effective N)
        bootstrap_block_sensitivity.pdf    (block-size 2–8 sensitivity plot)
"""

import json
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

RNG_SEED    = 42
N_BOOT      = 10_000
DEFAULT_B   = 5
DATA_FILE   = "results/expB_three_map_overlay.json"
OUT_FILE    = "results/bootstrap_block_results.json"
SENS_PDF    = "bootstrap_block_sensitivity.pdf"

PAIRS = [
    ("LRD_vs_Patching", "norm_lrd",   "norm_patch"),
    ("LRD_vs_LoRA",     "norm_lrd",   "norm_lora"),
    ("Patch_vs_LoRA",   "norm_patch", "norm_lora"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Block bootstrap core
# ─────────────────────────────────────────────────────────────────────────────

def block_bootstrap_spearman(x: np.ndarray, y: np.ndarray,
                             block_size: int, n_boot: int,
                             rng: np.random.Generator):
    """
    Block bootstrap Spearman ρ with contiguous blocks of `block_size`.
    Returns (rho_point, ci_low, ci_high, p_two_sided).
    """
    n = len(x)
    rho_point, _ = spearmanr(x, y)

    n_blocks_needed = math.ceil(n / block_size)
    # Valid start positions for a block of size `block_size`
    max_start = n - block_size  # inclusive

    boot_rhos = np.empty(n_boot)
    for i in range(n_boot):
        # Draw n_blocks_needed block starts with replacement
        starts = rng.integers(0, max_start + 1, size=n_blocks_needed)
        # Build resample index list
        idx = []
        for s in starts:
            idx.extend(range(s, s + block_size))
        idx = np.array(idx[:n])  # trim to exactly n
        rho_b, _ = spearmanr(x[idx], y[idx])
        boot_rhos[i] = rho_b

    ci_low, ci_high = np.percentile(boot_rhos, [2.5, 97.5])
    frac_neg = float((boot_rhos < 0).mean())
    p_two = 2.0 * min(frac_neg, 1.0 - frac_neg)

    return float(rho_point), float(ci_low), float(ci_high), float(p_two), boot_rhos


# ─────────────────────────────────────────────────────────────────────────────
# Effective N via VIF
# ─────────────────────────────────────────────────────────────────────────────

def effective_n(signal: np.ndarray, max_lag: int = 15) -> dict:
    """
    Compute variance inflation factor from ACF and return effective N.

    VIF = 1 + 2 * sum_{k=1}^{K*} ACF(k)
    where K* is the last positive lag (Bartlett truncation).
    effective_N = len(signal) / VIF
    """
    n = len(signal)
    s = signal - signal.mean()
    var = float((s ** 2).mean())
    if var < 1e-12:
        return {"vif": 1.0, "eff_n": float(n), "acf": [1.0], "trunc_lag": 0}

    acf_vals = [1.0]
    for k in range(1, min(max_lag + 1, n)):
        c = float((s[:n - k] * s[k:]).mean()) / var
        acf_vals.append(c)

    # Truncate at first non-positive lag (Bartlett's rule)
    trunc = 0
    for k in range(1, len(acf_vals)):
        if acf_vals[k] > 0:
            trunc = k
        else:
            break

    vif = 1.0 + 2.0 * sum(acf_vals[1: trunc + 1])
    vif = max(vif, 1.0)
    eff = n / vif

    return {
        "vif":       round(vif, 3),
        "eff_n":     round(eff, 1),
        "acf":       [round(v, 4) for v in acf_vals],
        "trunc_lag": trunc,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    rng  = np.random.default_rng(RNG_SEED)
    data = json.load(open(DATA_FILE))

    results = {}
    header  = f"{'Model':<10} {'Pair':<20} {'ρ':>7}  {'95% CI (block-b5)':>22}  {'p':>7}  {'eff_N':>6}"
    print(header)
    print("-" * len(header))

    for model, entry in data.items():
        results[model] = {"pairs": {}, "effective_n": {}}

        # Effective N for each signal
        for sig_key in ("norm_lrd", "norm_patch", "norm_lora"):
            arr = np.array(entry[sig_key], dtype=float)
            mask = ~np.isnan(arr)
            if mask.sum() < 4:
                continue
            results[model]["effective_n"][sig_key] = effective_n(arr[mask])

        for pair_name, key_x, key_y in PAIRS:
            x = np.array(entry[key_x], dtype=float)
            y = np.array(entry[key_y], dtype=float)
            mask = ~(np.isnan(x) | np.isnan(y))
            x, y = x[mask], y[mask]
            n_valid = int(mask.sum())

            rho, lo, hi, pval, _ = block_bootstrap_spearman(
                x, y, block_size=DEFAULT_B, n_boot=N_BOOT, rng=rng
            )

            # Effective N for this pair: min of the two signals
            en_x = results[model]["effective_n"].get(key_x, {}).get("eff_n", n_valid)
            en_y = results[model]["effective_n"].get(key_y, {}).get("eff_n", n_valid)
            en_pair = min(en_x, en_y)

            results[model]["pairs"][pair_name] = {
                "rho":      rho,
                "ci_low":   lo,
                "ci_high":  hi,
                "p_two":    pval,
                "n":        n_valid,
                "block_b":  DEFAULT_B,
                "eff_n":    round(en_pair, 1),
            }

            sig_flag = "***" if pval < 0.001 else ("**" if pval < 0.01 else
                       ("*" if pval < 0.05 else "  "))
            print(
                f"{model:<10} {pair_name:<20} {rho:+.3f}  "
                f"[{lo:+.3f}, {hi:+.3f}]  "
                f"p={pval:.4f}{sig_flag}  eff_N≈{en_pair:.0f}"
            )
        print()

    json.dump(results, open(OUT_FILE, "w"), indent=2)
    print(f"Saved: {OUT_FILE}")

    # ── Sensitivity: block size 2..8 ──────────────────────────────────────────
    print("\nRunning block-size sensitivity (b=2..8) ...")
    sensitivity = {}
    block_sizes = list(range(2, 9))
    rng2 = np.random.default_rng(RNG_SEED + 1)

    for model, entry in data.items():
        sensitivity[model] = {}
        for pair_name, key_x, key_y in PAIRS:
            x = np.array(entry[key_x], dtype=float)
            y = np.array(entry[key_y], dtype=float)
            mask = ~(np.isnan(x) | np.isnan(y))
            x, y = x[mask], y[mask]

            sens_rows = []
            for b in block_sizes:
                rho, lo, hi, pval, _ = block_bootstrap_spearman(
                    x, y, block_size=b, n_boot=N_BOOT, rng=rng2
                )
                sens_rows.append({"b": b, "rho": rho, "ci_low": lo,
                                  "ci_high": hi, "p": pval})
            sensitivity[model][pair_name] = sens_rows

    # Plot sensitivity
    n_pairs  = len(PAIRS)
    n_models = len(data)
    fig, axes = plt.subplots(n_models, n_pairs,
                             figsize=(4 * n_pairs, 3.5 * n_models),
                             sharey=False)
    if n_models == 1:
        axes = [axes]

    for row_i, model in enumerate(data):
        for col_j, (pair_name, _, _) in enumerate(PAIRS):
            ax    = axes[row_i][col_j]
            rows  = sensitivity[model][pair_name]
            bs    = [r["b"]       for r in rows]
            rhos  = [r["rho"]     for r in rows]
            los   = [r["ci_low"]  for r in rows]
            his   = [r["ci_high"] for r in rows]

            ax.plot(bs, rhos, "o-", color="steelblue", linewidth=2, label=r"$\rho$")
            ax.fill_between(bs, los, his, alpha=0.2, color="steelblue", label="95% CI")
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
            ax.axvline(DEFAULT_B, color="tomato", linewidth=1.0, linestyle=":",
                       label=f"default b={DEFAULT_B}")
            ax.set_xlabel("Block size b", fontsize=9)
            ax.set_ylabel(r"Spearman $\rho$", fontsize=9)
            ax.set_title(f"{model} / {pair_name}", fontsize=8)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Block bootstrap sensitivity — block size 2–8, n_boot={N_BOOT:,}",
        fontsize=11, y=1.01
    )
    plt.tight_layout()
    plt.savefig(SENS_PDF, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Sensitivity plot: {SENS_PDF}")

    # Save sensitivity JSON
    sens_file = "results/bootstrap_block_sensitivity.json"
    json.dump(sensitivity, open(sens_file, "w"), indent=2)
    print(f"Sensitivity data: {sens_file}")


if __name__ == "__main__":
    main()
