"""
bootstrap_diagnostics.py — Bootstrap CIs for LRD diagnostics (Priority 3).

Loads existing raw LRD results from lrd_diagnostics.py output and computes:
  1. Bootstrap CIs on point-biserial r (Claim 3)
  2. Bootstrap CIs on recovery rates and cascade slope (Claim 2)
  3. Power analysis: given n examples, what effect size is detectable at 80% power?

If --patching_dir is provided, also bootstraps per-layer activation patching
recovery rates with error bands.

Usage:
    python bootstrap_diagnostics.py \
        --lrd_results_dir ./models/phi3.5/lrd_results \
        --dataset gsm8k \
        --n_bootstrap 10000 \
        --output ./results/bootstrap

    # With patching results:
    python bootstrap_diagnostics.py \
        --lrd_results_dir ./models/phi3.5/lrd_results \
        --patching_dir ./models/phi3.5/lrd_results \
        --dataset gsm8k \
        --n_bootstrap 10000 \
        --output ./results/bootstrap
"""

import os, sys, json, argparse
import numpy as np
from scipy.stats import pointbiserialr


def bootstrap_ci(data_fn, data, n_bootstrap=10000, ci=0.95, seed=42):
    """Generic bootstrap: data_fn(resampled_data) -> scalar statistic."""
    rng = np.random.RandomState(seed)
    n = len(data)
    stats = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        resampled = [data[i] for i in idx]
        try:
            val = data_fn(resampled)
            if np.isfinite(val):
                stats.append(val)
        except Exception:
            continue
    stats = np.array(stats)
    alpha = (1 - ci) / 2
    return {
        "mean": float(np.mean(stats)),
        "ci_lo": float(np.percentile(stats, alpha * 100)),
        "ci_hi": float(np.percentile(stats, (1 - alpha) * 100)),
        "n_valid": len(stats),
    }


def bootstrap_point_biserial(records, n_bootstrap=10000):
    """Bootstrap CI on point-biserial r between final_lrd and is_correct."""
    data = [(r["final_lrd"], int(r["is_correct"])) for r in records]

    def compute_r(pairs):
        lrd = np.array([p[0] for p in pairs])
        correct = np.array([p[1] for p in pairs])
        if correct.std() == 0 or lrd.std() == 0:
            return float("nan")
        r, _ = pointbiserialr(lrd, correct)
        return r

    return bootstrap_ci(compute_r, data, n_bootstrap)


def bootstrap_recovery_rate(records, n_bootstrap=10000):
    """Bootstrap CI on % of examples showing LRD recovery (late < 0.8 * early)."""
    recovery_flags = [int(r["recovered"]) for r in records]
    return bootstrap_ci(lambda d: np.mean(d), recovery_flags, n_bootstrap)


def bootstrap_cascade_slope(records, n_bootstrap=10000):
    """Bootstrap CI on mean cascade slope."""
    slopes = [r["cascade_slope"] for r in records]
    return bootstrap_ci(lambda d: np.mean(d), slopes, n_bootstrap)


def bootstrap_accuracy(records, n_bootstrap=10000):
    """Bootstrap CI on accuracy under perturbation."""
    correct = [int(r["is_correct"]) for r in records]
    return bootstrap_ci(lambda d: np.mean(d), correct, n_bootstrap)


def power_analysis(n, alpha=0.05, power=0.80):
    """
    Minimum detectable effect size (Cohen's d) at given n, alpha, power.
    Uses approximation: d = (z_alpha + z_power) / sqrt(n).

    Also computes the minimum detectable accuracy difference (for
    a two-proportion z-test at baseline accuracy ~70%).
    """
    from scipy.stats import norm
    z_alpha = norm.ppf(1 - alpha / 2)
    z_power = norm.ppf(power)
    d_min = (z_alpha + z_power) / np.sqrt(n)

    # For accuracy differences: min detectable Δp at baseline p0=0.70
    # SE = sqrt(p0*(1-p0)/n + p0*(1-p0)/n) = sqrt(2*p0*(1-p0)/n)
    p0 = 0.70
    se_two_sample = np.sqrt(2 * p0 * (1 - p0) / n)
    min_delta_p = (z_alpha + z_power) * se_two_sample

    return {
        "n": n,
        "alpha": alpha,
        "power": power,
        "min_detectable_d": float(d_min),
        "min_detectable_acc_delta": float(min_delta_p),
        "interpretation": f"With n={n}, can detect Cohen's d >= {d_min:.3f} at {power*100:.0f}% power",
        # Paper-ready sentence for limitations section:
        "limitations_text": (
            f"Given our diagnostic sample size of n={n} examples per condition, "
            f"a post-hoc power analysis indicates we can detect effect sizes of "
            f"Cohen's d >= {d_min:.2f} (approximately {min_delta_p*100:.1f} percentage "
            f"points in accuracy difference at a 70% baseline) with {power*100:.0f}% "
            f"power at alpha={alpha}. Effects below this threshold may exist but would "
            f"require larger samples to detect reliably."
        ),
    }


def bootstrap_patching_curves(patching_data, n_bootstrap=10000):
    """Bootstrap per-layer patching recovery with CIs.

    patching_data: dict with per-layer lists of recovery values
    Expected format: {"layer_0": [recovery_1, recovery_2, ...], ...}
    """
    results = {}
    for layer_name, values in sorted(patching_data.items()):
        ci = bootstrap_ci(lambda d: np.mean(d), values, n_bootstrap)
        results[layer_name] = ci
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lrd_results_dir", type=str, required=True,
                        help="Directory containing raw_<dataset>.json from lrd_diagnostics.py")
    parser.add_argument("--dataset", type=str, default="gsm8k")
    parser.add_argument("--patching_dir", type=str, default=None,
                        help="Directory with patching results (optional)")
    parser.add_argument("--n_bootstrap", type=int, default=10000)
    parser.add_argument("--output", type=str, default="./results/bootstrap")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load raw LRD results (handle NaN/Infinity from numpy serialization)
    raw_path = os.path.join(args.lrd_results_dir, f"raw_{args.dataset}.json")
    print(f"Loading LRD results from {raw_path}")
    with open(raw_path) as f:
        content = f.read().replace('NaN', 'null').replace('Infinity', '1e308')
        all_results = json.loads(content)

    bootstrap_output = {}

    for exp_name, records in all_results.items():
        if not records or len(records) < 10:
            print(f"  Skipping {exp_name}: only {len(records)} records")
            continue

        print(f"\n{'─'*55}\n{exp_name} (n={len(records)})\n{'─'*55}")

        r_ci = bootstrap_point_biserial(records, args.n_bootstrap)
        recovery_ci = bootstrap_recovery_rate(records, args.n_bootstrap)
        slope_ci = bootstrap_cascade_slope(records, args.n_bootstrap)
        acc_ci = bootstrap_accuracy(records, args.n_bootstrap)
        power = power_analysis(len(records))

        print(f"  Point-biserial r: {r_ci['mean']:.4f}  95% CI [{r_ci['ci_lo']:.4f}, {r_ci['ci_hi']:.4f}]")
        print(f"  Recovery rate:    {recovery_ci['mean']:.4f}  95% CI [{recovery_ci['ci_lo']:.4f}, {recovery_ci['ci_hi']:.4f}]")
        print(f"  Cascade slope:    {slope_ci['mean']:.6f}  95% CI [{slope_ci['ci_lo']:.6f}, {slope_ci['ci_hi']:.6f}]")
        print(f"  Accuracy:         {acc_ci['mean']:.4f}  95% CI [{acc_ci['ci_lo']:.4f}, {acc_ci['ci_hi']:.4f}]")
        print(f"  {power['interpretation']}")

        bootstrap_output[exp_name] = {
            "n": len(records),
            "point_biserial_r": r_ci,
            "recovery_rate": recovery_ci,
            "cascade_slope": slope_ci,
            "accuracy": acc_ci,
            "power_analysis": power,
        }

    # Bootstrap patching curves if available
    if args.patching_dir:
        import glob
        patching_files = glob.glob(os.path.join(args.patching_dir, f"patching_{args.dataset}*.json"))
        for patching_path in patching_files:
            print(f"\nBootstrapping activation patching curves from {patching_path}")
            try:
                with open(patching_path) as f:
                    content = f.read().replace('NaN', 'null').replace('Infinity', '1e308')
                    patching_data = json.loads(content)
                tag = os.path.basename(patching_path).replace(".json", "")
                bootstrap_output[tag] = bootstrap_patching_curves(
                    patching_data, args.n_bootstrap
                )
            except Exception as ex:
                print(f"  Failed to process {patching_path}: {ex}")
        if not patching_files:
            print(f"  No patching files found matching patching_{args.dataset}*.json")

    # Print paper-ready power analysis text (for limitations section)
    # Use the smallest sample size across conditions for the most conservative estimate
    sample_sizes = [v["n"] for v in bootstrap_output.values() if isinstance(v, dict) and "n" in v]
    if sample_sizes:
        min_n = min(sample_sizes)
        pa = power_analysis(min_n)
        print(f"\n{'='*55}")
        print("PAPER-READY TEXT (limitations section):")
        print(f"{'='*55}")
        print(pa["limitations_text"])
        print(f"{'='*55}")
        bootstrap_output["_power_analysis_summary"] = {
            "min_sample_size": min_n,
            "limitations_text": pa["limitations_text"],
            **{k: v for k, v in pa.items() if k != "limitations_text"},
        }

    # Save
    out_path = os.path.join(args.output, f"bootstrap_{args.dataset}.json")
    with open(out_path, "w") as f:
        json.dump(bootstrap_output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
