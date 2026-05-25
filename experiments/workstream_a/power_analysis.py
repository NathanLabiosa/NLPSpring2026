"""
A3: Power Analysis Update

Report what sample size would be needed to detect individually significant
cosine-vs-MSE differences for each perturbation type at 80% power.
"""

import json, os
import numpy as np

# ── Config ──────────────────────────────────────────────────────────────────
COSINE_PATH = "bootstrap_results/cosv2_L15_stab24_high/paired_bootstrap_results.json"
MSE_PATH    = "bootstrap_results/mse_L15_stab24_high/paired_bootstrap_results.json"
OUTPUT_DIR  = "experiments/workstream_a/power_analysis"

DIRECTIONAL = {"typos_5pct", "ocr_5pct", "speech_10pct", "homophones_20pct"}
UNIFORM     = {"whitespace_10pct", "case_10pct"}


def required_n_mcnemar(p_cosine, p_mse, power=0.80, alpha=0.05):
    """
    Approximate sample size for McNemar's test (paired proportions).

    Uses the formula: n = (z_alpha + z_beta)^2 / (p1 - p2)^2 * (p_discord)
    where p_discord = P(cosine correct, MSE wrong) + P(cosine wrong, MSE correct).

    For planning, we estimate p_discord from marginal differences.
    """
    from scipy.stats import norm
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)

    diff = abs(p_cosine - p_mse)
    if diff < 1e-6:
        return float("inf")

    # Estimate discordant proportion from marginal accuracies
    # Conservative: assume max possible discordance
    p_discord = abs(p_cosine - p_mse) + 2 * min(p_cosine, p_mse) * (1 - max(p_cosine, p_mse))
    p_discord = min(p_discord, 0.5)  # cap at 50%
    p_discord = max(p_discord, diff)  # at least the difference

    n = (z_alpha + z_beta) ** 2 * p_discord / diff ** 2
    return int(np.ceil(n))


def required_n_paired_proportion(effect_size, power=0.80, alpha=0.05):
    """
    Sample size for detecting a paired proportion difference.
    Uses normal approximation: n = (z_alpha + z_beta)^2 * 2*p*(1-p) / d^2
    where d is the expected difference and p is the average proportion.
    """
    from scipy.stats import norm
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)

    if abs(effect_size) < 1e-6:
        return float("inf")

    # For binary outcomes, variance of difference ≈ 2*p*(1-p) where p ≈ 0.7-0.8
    p_avg = 0.75  # reasonable for GSM8K accuracy range
    var_diff = 2 * p_avg * (1 - p_avg)

    n = (z_alpha + z_beta) ** 2 * var_diff / effect_size ** 2
    return int(np.ceil(n))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(COSINE_PATH) as f:
        cosine_data = json.load(f)
    with open(MSE_PATH) as f:
        mse_data = json.load(f)

    print(f"{'='*70}")
    print("A3: POWER ANALYSIS — SAMPLE SIZE FOR PER-CONDITION SIGNIFICANCE")
    print(f"{'='*70}")

    results = {}

    print(f"\n{'Condition':<22} {'Cos Δ':>7} {'MSE Δ':>7} {'Diff':>7} {'n needed':>10} {'Taxonomy':>12}")
    print("-" * 70)

    for cond in sorted(set(cosine_data.keys()) & set(mse_data.keys())):
        if cond.startswith("_") or cond == "clean_baseline":
            continue

        cos_delta = cosine_data[cond]["delta_pct"]
        mse_delta = mse_data[cond]["delta_pct"]
        diff = cos_delta - mse_delta
        taxonomy = "directional" if cond in DIRECTIONAL else "uniform"

        # Compute effect size from per-sample data
        cos_with = np.array(cosine_data[cond]["per_sample_with_adapter"])
        mse_with = np.array(mse_data[cond]["per_sample_with_adapter"])
        observed_effect = cos_with.mean() - mse_with.mean()

        n_needed = required_n_paired_proportion(observed_effect)
        if n_needed > 1e6:
            n_str = ">1M"
        else:
            n_str = f"{n_needed:,}"

        print(f"  {cond:<20} {cos_delta:>+6.1f}% {mse_delta:>+6.1f}% {diff:>+6.1f}% {n_str:>10} {taxonomy:>12}")

        results[cond] = {
            "cosine_delta": cos_delta,
            "mse_delta": mse_delta,
            "cosine_minus_mse": round(diff, 2),
            "observed_effect_size": round(observed_effect * 100, 3),
            "n_required_80pct_power": n_needed if n_needed < 1e6 else None,
            "taxonomy": taxonomy,
            "current_n": 500,
            "powered": n_needed <= 500,
        }

    # Summary
    powered = [c for c, r in results.items() if r["powered"]]
    underpowered = [c for c, r in results.items() if not r["powered"]]

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Currently powered (n=500 sufficient): {powered if powered else 'None'}")
    print(f"Underpowered:                          {underpowered if underpowered else 'None'}")
    print(f"\nInterpretation: Individual per-condition cosine-vs-MSE comparisons")
    if underpowered:
        print(f"require larger samples. The aggregate (pooled) test and the")
        print(f"interaction test (A1) are the appropriate statistical approach")
        print(f"for the current sample size.")

    output = {
        "per_condition": results,
        "powered_conditions": powered,
        "underpowered_conditions": underpowered,
        "test_params": {
            "alpha": 0.05,
            "power": 0.80,
            "current_n": 500,
        },
    }
    out_path = os.path.join(OUTPUT_DIR, "power_analysis_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
