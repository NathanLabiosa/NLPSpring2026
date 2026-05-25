"""
Bootstrap 95% CIs for the three-map Spearman correlation matrix (Table 2).
Resamples the 32 layer-pairs with replacement, 10k iterations per pair.
Reads expB_three_map_overlay.json; writes bootstrap_correlations.json.
"""

import json
import numpy as np
from scipy.stats import spearmanr

RNG_SEED = 42
N_BOOT = 10_000
DATA_FILE = "expB_three_map_overlay.json"
OUT_FILE = "bootstrap_correlations.json"


def bootstrap_spearman(x, y, n_boot=N_BOOT, rng=None):
    """Return (rho_point, ci_low, ci_high) via percentile bootstrap."""
    rho_point, _ = spearmanr(x, y)
    n = len(x)
    boot_rhos = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_rhos[i], _ = spearmanr(x[idx], y[idx])
    ci_low, ci_high = np.percentile(boot_rhos, [2.5, 97.5])
    return float(rho_point), float(ci_low), float(ci_high)


def main():
    rng = np.random.default_rng(RNG_SEED)
    data = json.load(open(DATA_FILE))

    pairs = [
        ("LRD_vs_Patching", "norm_lrd",   "norm_patch"),
        ("LRD_vs_LoRA",     "norm_lrd",   "norm_lora"),
        ("Patch_vs_LoRA",   "norm_patch", "norm_lora"),
    ]

    results = {}
    header = f"{'Model':<10} {'Pair':<20} {'ρ':>7}  {'95% CI':>18}"
    print(header)
    print("-" * len(header))

    for model, entry in data.items():
        results[model] = {}
        for pair_name, key_x, key_y in pairs:
            x = np.array(entry[key_x], dtype=float)
            y = np.array(entry[key_y], dtype=float)
            # Drop positions where either is NaN (embedding layer in patch)
            mask = ~(np.isnan(x) | np.isnan(y))
            x, y = x[mask], y[mask]

            rho, lo, hi = bootstrap_spearman(x, y, rng=rng)
            results[model][pair_name] = {"rho": rho, "ci_low": lo, "ci_high": hi, "n": int(mask.sum())}
            print(f"{model:<10} {pair_name:<20} {rho:+.3f}  [{lo:+.3f}, {hi:+.3f}]")
        print()

    json.dump(results, open(OUT_FILE, "w"), indent=2)
    print(f"Saved: {OUT_FILE}")


if __name__ == "__main__":
    main()
