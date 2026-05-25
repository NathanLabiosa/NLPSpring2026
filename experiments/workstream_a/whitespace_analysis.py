"""
A2: Resolve the Whitespace Contradiction

Whitespace is classified as "uniform" (LRD does not predict failure), but
Section 4.3 lists it as having a significant LRD-failure association.

Computes Cohen's d for the LRD difference between correct and incorrect
examples for each perturbation type, producing a ranked bar chart.
"""

import json, os
import numpy as np

# ── Config ──────────────────────────────────────────────────────────────────
LRD_DATA_PATH = "Phi3.5/lrd_gsm/raw_gsm8k.json"
OUTPUT_DIR    = "experiments/workstream_a/whitespace_analysis"

# Map raw data keys to taxonomy
DIRECTIONAL_KEYS = {"Typos_5%", "OCR_5%", "Homophones_20%"}
# Note: speech may not be in the diagnostic data. Check and classify if present.
UNIFORM_KEYS     = {"Whitespace_10%", "Case_10%"}

N_BOOTSTRAP = 10000
SEED = 42


def cohens_d(group1, group2):
    """Compute Cohen's d (pooled SD)."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    m1, m2 = np.mean(group1), np.mean(group2)
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if pooled_sd == 0:
        return 0.0
    return (m1 - m2) / pooled_sd


def bootstrap_cohens_d(correct_lrd, incorrect_lrd, n_bootstrap=10000, seed=42):
    """Bootstrap CI on Cohen's d."""
    rng = np.random.RandomState(seed)
    boot_ds = []
    for _ in range(n_bootstrap):
        c_boot = correct_lrd[rng.randint(0, len(correct_lrd), len(correct_lrd))]
        i_boot = incorrect_lrd[rng.randint(0, len(incorrect_lrd), len(incorrect_lrd))]
        boot_ds.append(cohens_d(i_boot, c_boot))  # incorrect - correct direction
    boot_ds = np.sort(boot_ds)
    return np.percentile(boot_ds, 2.5), np.percentile(boot_ds, 97.5)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(LRD_DATA_PATH) as f:
        raw_data = json.load(f)

    print(f"{'='*70}")
    print("A2: WHITESPACE CONTRADICTION ANALYSIS")
    print(f"{'='*70}")
    print(f"\nAvailable conditions: {list(raw_data.keys())}")

    results = {}
    all_effect_sizes = []

    for cond_name, examples in raw_data.items():
        correct_lrd = [ex["final_lrd"] for ex in examples if ex["is_correct"]]
        incorrect_lrd = [ex["final_lrd"] for ex in examples if not ex["is_correct"]]

        n_correct = len(correct_lrd)
        n_incorrect = len(incorrect_lrd)

        if n_incorrect < 5:
            print(f"\n{cond_name}: Skipping (only {n_incorrect} incorrect examples)")
            continue

        correct_lrd = np.array(correct_lrd)
        incorrect_lrd = np.array(incorrect_lrd)

        d = cohens_d(incorrect_lrd, correct_lrd)  # positive = incorrect has higher LRD
        ci_lo, ci_hi = bootstrap_cohens_d(correct_lrd, incorrect_lrd, N_BOOTSTRAP, SEED)

        # Also compute mean-LRD version
        correct_mean_lrd = np.array([ex["mean_lrd"] for ex in examples if ex["is_correct"]])
        incorrect_mean_lrd = np.array([ex["mean_lrd"] for ex in examples if not ex["is_correct"]])
        d_mean = cohens_d(incorrect_mean_lrd, correct_mean_lrd)

        taxonomy = "directional" if cond_name in DIRECTIONAL_KEYS else "uniform"

        print(f"\n{cond_name} [{taxonomy.upper()}]:")
        print(f"  n_correct={n_correct}, n_incorrect={n_incorrect}")
        print(f"  Mean final LRD: correct={correct_lrd.mean():.4f}, incorrect={incorrect_lrd.mean():.4f}")
        print(f"  Cohen's d (final LRD): {d:.3f}  95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]")
        print(f"  Cohen's d (mean LRD):  {d_mean:.3f}")

        results[cond_name] = {
            "taxonomy": taxonomy,
            "n_correct": n_correct,
            "n_incorrect": n_incorrect,
            "mean_lrd_correct": round(float(correct_lrd.mean()), 5),
            "mean_lrd_incorrect": round(float(incorrect_lrd.mean()), 5),
            "cohens_d_final": round(d, 4),
            "cohens_d_mean": round(d_mean, 4),
            "ci_lo": round(ci_lo, 4),
            "ci_hi": round(ci_hi, 4),
            "significant": bool(ci_lo > 0),  # positive d = LRD predicts failure
        }
        all_effect_sizes.append((cond_name, d, taxonomy))

    # Sort by effect size
    all_effect_sizes.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'='*70}")
    print("RANKED EFFECT SIZES (Cohen's d: LRD predicts failure)")
    print(f"{'='*70}")
    for cond, d, tax in all_effect_sizes:
        bar = "█" * int(abs(d) * 20)
        print(f"  {cond:<20} [{tax[:3].upper()}]  d={d:+.3f}  {bar}")

    # Assess taxonomy boundary
    dir_ds = [d for _, d, t in all_effect_sizes if t == "directional"]
    uni_ds = [d for _, d, t in all_effect_sizes if t == "uniform"]

    print(f"\nDirectional mean d: {np.mean(dir_ds):.3f}")
    print(f"Uniform mean d:     {np.mean(uni_ds):.3f}")
    gap = min(dir_ds) - max(uni_ds) if dir_ds and uni_ds else 0
    print(f"Gap between groups: {gap:.3f}")
    if gap > 0:
        threshold = (min(dir_ds) + max(uni_ds)) / 2
        print(f"Clean separation. Suggested threshold: d = {threshold:.3f}")
        print("→ Classification: directional if Cohen's d > {:.3f}".format(threshold))
    else:
        print("Groups overlap — taxonomy is a continuum, not a binary split.")
        print("→ Consider reframing as 'strongly predictive' vs 'weakly predictive'")

    # Generate figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        names = [x[0] for x in all_effect_sizes]
        ds = [x[1] for x in all_effect_sizes]
        colors = ["#2196F3" if x[2] == "directional" else "#FF9800" for x in all_effect_sizes]

        ci_los = [results[n]["ci_lo"] for n in names]
        ci_his = [results[n]["ci_hi"] for n in names]
        errors = [[d - lo for d, lo in zip(ds, ci_los)],
                  [hi - d for d, hi in zip(ds, ci_his)]]

        y_pos = range(len(names))
        ax.barh(y_pos, ds, color=colors, edgecolor="black", linewidth=0.5,
                xerr=errors, capsize=3, error_kw={"linewidth": 1})
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=10)
        ax.set_xlabel("Cohen's d (LRD–failure association)", fontsize=11)
        ax.set_title("Effect Size: LRD Predicts Task Failure by Perturbation Type", fontsize=12)
        ax.axvline(x=0, color="black", linewidth=0.5)

        # Add taxonomy threshold if clean separation
        if gap > 0:
            threshold = (min(dir_ds) + max(uni_ds)) / 2
            ax.axvline(x=threshold, color="red", linestyle="--", linewidth=1, label=f"Threshold d={threshold:.2f}")
            ax.legend(fontsize=9)

        # Legend patches
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor="#2196F3", label="Directional"),
                          Patch(facecolor="#FF9800", label="Uniform")]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

        plt.tight_layout()
        fig_path = os.path.join(OUTPUT_DIR, "effect_size_ranking.pdf")
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        print(f"\nFigure saved: {fig_path}")
    except ImportError:
        print("\nmatplotlib not available — skipping figure generation")

    # Save results
    output = {
        "per_condition": results,
        "ranking": [(n, round(d, 4), t) for n, d, t in all_effect_sizes],
        "directional_mean_d": round(float(np.mean(dir_ds)), 4) if dir_ds else None,
        "uniform_mean_d": round(float(np.mean(uni_ds)), 4) if uni_ds else None,
        "gap": round(gap, 4),
    }
    out_path = os.path.join(OUTPUT_DIR, "whitespace_analysis_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
