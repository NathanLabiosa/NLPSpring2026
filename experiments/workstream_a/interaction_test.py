"""
A1: Formal Interaction Test for Taxonomy × Geometry

Tests whether the cosine-vs-MSE advantage is significantly moderated by
taxonomy class (directional vs. uniform).

Runs on BOTH original-rate and matched-rate (10%) Phi data.
The contrast between the two p-values is the backbone of Section 6.3-6.4.
"""

import json, os
import numpy as np
from collections import defaultdict

# ── Config ──────────────────────────────────────────────────────────────────
DATASETS = {
    "original_rate": {
        "cosine": "bootstrap_results/cosv2_L15_stab24_high/paired_bootstrap_results.json",
        "mse":    "bootstrap_results/mse_L15_stab24_high/paired_bootstrap_results.json",
        "label":  "Original rates (typos 5%, OCR 5%, WS 10%, case 10%, speech 10%, homo 20%)",
    },
    "matched_rate": {
        "cosine": "bootstrap_results/phi_matched_cosine_stab24/paired_bootstrap_results.json",
        "mse":    "bootstrap_results/phi_matched_mse_stab24/paired_bootstrap_results.json",
        "label":  "Matched rate (all perturbations at 10%)",
    },
}

OUTPUT_DIR = "experiments/workstream_a/interaction_test"

# Taxonomy classification — by perturbation method name (ignoring rate suffix)
DIRECTIONAL_METHODS = {"typos", "ocr", "speech", "homophones"}
UNIFORM_METHODS     = {"whitespace", "case"}

N_BOOTSTRAP = 10000
SEED = 42


def classify_condition(cond_name):
    """Classify a condition key like 'typos_5pct' into directional/uniform."""
    method = cond_name.rsplit("_", 1)[0]  # strip rate suffix
    if method in DIRECTIONAL_METHODS:
        return "directional"
    elif method in UNIFORM_METHODS:
        return "uniform"
    return None


def run_interaction_test(cosine_path, mse_path, dataset_label, n_bootstrap=10000, seed=42):
    """Run both aggregate and per-example interaction tests."""
    with open(cosine_path) as f:
        cosine_data = json.load(f)
    with open(mse_path) as f:
        mse_data = json.load(f)

    # Get overlapping perturbed conditions
    all_conds = sorted([
        k for k in set(cosine_data.keys()) & set(mse_data.keys())
        if not k.startswith("_") and k != "clean_baseline"
    ])

    print(f"\n{'='*70}")
    print(f"INTERACTION TEST: {dataset_label}")
    print(f"{'='*70}")

    # ── 1. Aggregate delta analysis ─────────────────────────────────────
    print(f"\n--- Per-condition cosine − MSE deltas ---")
    diff_per_cond = {}
    for c in all_conds:
        cos_d = cosine_data[c]["delta_pct"]
        mse_d = mse_data[c]["delta_pct"]
        diff = cos_d - mse_d
        diff_per_cond[c] = diff
        tax = classify_condition(c)
        print(f"  {c:<22} [{tax[:3].upper()}]: cos={cos_d:+.1f}% mse={mse_d:+.1f}% → diff={diff:+.1f}%")

    dir_conds = [c for c in all_conds if classify_condition(c) == "directional"]
    uni_conds = [c for c in all_conds if classify_condition(c) == "uniform"]

    dir_mean = np.mean([diff_per_cond[c] for c in dir_conds])
    uni_mean = np.mean([diff_per_cond[c] for c in uni_conds])
    observed_interaction = dir_mean - uni_mean

    print(f"\n  Directional mean (cos−MSE): {dir_mean:+.2f}%")
    print(f"  Uniform mean (cos−MSE):     {uni_mean:+.2f}%")
    print(f"  Observed interaction:        {observed_interaction:+.2f}%")

    # Permutation test on aggregate deltas
    rng = np.random.RandomState(seed)
    n_dir = len(dir_conds)
    diffs_array = np.array([diff_per_cond[c] for c in all_conds])
    null_interactions = []
    for _ in range(n_bootstrap):
        perm = rng.permutation(len(all_conds))
        group_a = diffs_array[perm[:n_dir]].mean()
        group_b = diffs_array[perm[n_dir:]].mean()
        null_interactions.append(group_a - group_b)
    null_interactions = np.array(null_interactions)
    p_agg = float(np.mean(np.abs(null_interactions) >= np.abs(observed_interaction)))

    print(f"\n  Permutation test (aggregate): p = {p_agg:.4f}")

    # ── 2. Per-example paired analysis ──────────────────────────────────
    print(f"\n--- Per-example paired analysis ---")
    by_taxonomy = defaultdict(list)
    per_cond_improvements = {}

    for c in all_conds:
        tax = classify_condition(c)
        cos_samples = cosine_data[c]["per_sample_with_adapter"]
        mse_samples = mse_data[c]["per_sample_with_adapter"]
        improvements = [cs - ms for cs, ms in zip(cos_samples, mse_samples)]
        by_taxonomy[tax].extend(improvements)
        per_cond_improvements[c] = improvements

    paired_results = {}
    for tax in ["directional", "uniform"]:
        imp = np.array(by_taxonomy[tax])
        n = len(imp)
        mean_imp = imp.mean()
        se = imp.std() / np.sqrt(n)
        print(f"  {tax.upper()} (n={n}): mean cos−MSE = {mean_imp*100:+.2f}% ± {se*100:.2f}%")
        paired_results[tax] = {
            "n": n, "mean_pct": round(mean_imp * 100, 3), "se_pct": round(se * 100, 3),
        }

    # Bootstrap CI on the interaction (directional improvement - uniform improvement)
    dir_imp = np.array(by_taxonomy["directional"])
    uni_imp = np.array(by_taxonomy["uniform"])
    rng2 = np.random.RandomState(seed + 1)
    boot_diffs = []
    for _ in range(n_bootstrap):
        d_boot = dir_imp[rng2.randint(0, len(dir_imp), len(dir_imp))].mean()
        u_boot = uni_imp[rng2.randint(0, len(uni_imp), len(uni_imp))].mean()
        boot_diffs.append(d_boot - u_boot)
    boot_diffs = np.sort(boot_diffs)
    ci_lo = float(np.percentile(boot_diffs, 2.5))
    ci_hi = float(np.percentile(boot_diffs, 97.5))
    interaction_paired = dir_imp.mean() - uni_imp.mean()
    sig_paired = bool(ci_lo > 0 or ci_hi < 0)

    print(f"\n  INTERACTION (per-example bootstrap):")
    print(f"    Directional − Uniform improvement: {interaction_paired*100:+.2f}%")
    print(f"    95% CI: [{ci_lo*100:+.2f}%, {ci_hi*100:+.2f}%]")
    print(f"    {'SIGNIFICANT' if sig_paired else 'NOT significant'} (CI {'excludes' if sig_paired else 'includes'} zero)")

    # ── 3. Per-condition bootstrap CIs on cos-MSE difference ────────────
    print(f"\n--- Per-condition cosine−MSE bootstrap CIs ---")
    per_cond_ci = {}
    rng3 = np.random.RandomState(seed + 2)
    for c in all_conds:
        imp = np.array(per_cond_improvements[c])
        boot_means = []
        for _ in range(n_bootstrap):
            boot_means.append(imp[rng3.randint(0, len(imp), len(imp))].mean())
        boot_means = np.sort(boot_means)
        lo = float(np.percentile(boot_means, 2.5))
        hi = float(np.percentile(boot_means, 97.5))
        point = imp.mean()
        sig = bool(lo > 0 or hi < 0)
        tax = classify_condition(c)
        print(f"  {c:<22} [{tax[:3].upper()}]: {point*100:+.2f}%  CI [{lo*100:+.2f}%, {hi*100:+.2f}%]  {'*' if sig else ''}")
        per_cond_ci[c] = {
            "point_pct": round(point * 100, 2),
            "ci_lo_pct": round(lo * 100, 2),
            "ci_hi_pct": round(hi * 100, 2),
            "significant": sig,
            "taxonomy": tax,
        }

    return {
        "aggregate": {
            "directional_mean_diff": round(dir_mean, 2),
            "uniform_mean_diff": round(uni_mean, 2),
            "observed_interaction": round(observed_interaction, 2),
            "p_value_permutation": round(p_agg, 4),
            "significant_005": p_agg < 0.05,
        },
        "paired": {
            "directional": paired_results.get("directional"),
            "uniform": paired_results.get("uniform"),
            "interaction_pct": round(interaction_paired * 100, 3),
            "interaction_ci_lo_pct": round(ci_lo * 100, 3),
            "interaction_ci_hi_pct": round(ci_hi * 100, 3),
            "interaction_significant": sig_paired,
        },
        "per_condition_ci": per_cond_ci,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_results = {}

    for name, cfg in DATASETS.items():
        results = run_interaction_test(
            cfg["cosine"], cfg["mse"], cfg["label"], N_BOOTSTRAP, SEED
        )
        all_results[name] = results

    # ── Summary comparison ──────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY: INTERACTION TEST COMPARISON")
    print(f"{'='*70}")
    for name in ["original_rate", "matched_rate"]:
        r = all_results[name]
        agg = r["aggregate"]
        paired = r["paired"]
        print(f"\n{name.upper()}:")
        print(f"  Aggregate permutation p = {agg['p_value_permutation']:.4f}")
        print(f"  Per-example interaction: {paired['interaction_pct']:+.1f}%  "
              f"CI [{paired['interaction_ci_lo_pct']:+.1f}%, {paired['interaction_ci_hi_pct']:+.1f}%]  "
              f"{'SIG' if paired['interaction_significant'] else 'N.S.'}")

    out_path = os.path.join(OUTPUT_DIR, "interaction_test_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
