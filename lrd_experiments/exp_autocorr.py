
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

DATA_FILE = "expB_three_map_overlay.json"
OUT_FILE  = "exp_autocorr_results.json"
OUT_PDF   = "exp_autocorr.pdf"

MAX_LAG = 8
ONE_OVER_E = 1.0 / np.e  # ≈ 0.368, threshold for decorrelation

SIGNAL_KEYS = [
    ("LRD", "norm_lrd"),
    ("Patching recovery","norm_patch"),
    ("LoRA effectiveness","norm_lora"),
]


def compute_acf(signal: np.ndarray, max_lag: int) -> np.ndarray:
    """Biased ACF estimator (consistent with standard statsmodels convention)."""
    n = len(signal)
    s = signal - signal.mean()
    var = float((s ** 2).mean())
    if var < 1e-12:
        return np.ones(max_lag + 1)  # constant signal has ACF=1 everywhere
    # print(f"acf: n={n}, var={var:.4f}")

    acf = np.empty(max_lag + 1)
    acf[0] = 1.0
    for k in range(1, max_lag + 1):
        if k >= n:
            acf[k] = 0.0
        else:
            acf[k] = float((s[:n - k] * s[k:]).mean()) / var
    return acf


def decorrelation_length(acf: np.ndarray) -> int:

    for k in range(1, len(acf)):
        if abs(acf[k]) < ONE_OVER_E:
            return k
    return len(acf) - 1  # still correlated at max_lag


def vif_from_acf(acf: np.ndarray, trunc_lag: int) -> float:
    if trunc_lag < 1:
        return 1.0
    return max(1.0, 1.0 + 2.0 * sum(acf[1:trunc_lag + 1]))


def main():
    data = json.load(open(DATA_FILE))
    results = {}
    # print(f"loaded {len(data)} models")

    n_models = len(data)
    n_signals = len(SIGNAL_KEYS)

    # create subplots: rows=models, cols=signals
    fig, axes = plt.subplots(
        n_models, n_signals,
        figsize=(4.5 * n_signals, 3.5 * n_models),
        sharey=False,
    )
    if n_models == 1:
        axes = [axes]  # ensure axes is 2D

    lags = np.arange(MAX_LAG + 1)

    for row_i, (model, entry) in enumerate(data.items()):
        results[model] = {}
        # print(f"processing {model}")
        for col_j, (sig_label, sig_key) in enumerate(SIGNAL_KEYS):
            arr = np.array(entry[sig_key], dtype=float)
            mask = ~np.isnan(arr)
            arr = arr[mask]
            n = len(arr)

            acf = compute_acf(arr, MAX_LAG)
            decor = decorrelation_length(acf)
            vif = vif_from_acf(acf, trunc_lag=decor)
            eff_n = round(n / vif, 1)
            # print(f"{sig_label}: decor={decor}, vif={vif:.2f}")

            results[model][sig_label] = {
                "n": n,
                "acf_lag0_to_8": [round(float(v), 4) for v in acf],
                "decorrelation_lag": int(decor),
                "vif": round(float(vif), 3),
                "effective_n": eff_n,
            }

            print(
                f"{model:<10}  {sig_label:<22}  "
                f"decor_lag={decor}  VIF={vif:.2f}  eff_N≈{eff_n}"
            )

            # plot ACF bar chart
            ax = axes[row_i][col_j]
            ax.bar(lags, acf, color="steelblue", alpha=0.7, label="ACF")
            ax.axhline( ONE_OVER_E, color="tomato", linestyle="--",
                        linewidth=1.2, label=f"1/e ≈ {ONE_OVER_E:.2f}")
            ax.axhline(-ONE_OVER_E, color="tomato", linestyle="--",
                        linewidth=1.2)
            ax.axhline(0, color="black", linewidth=0.7)
            if decor <= MAX_LAG:
                ax.axvline(decor, color="purple", linestyle=":",
                           linewidth=1.5, label=f"decor lag={decor}")

            # Bartlett confidence bands
            ci = 1.96 / np.sqrt(n)
            ax.fill_between(lags, -ci, ci, alpha=0.12, color="gray",
                            label="±1.96/√n CI")

            ax.set_xticks(lags)
            ax.set_xlabel("Lag (layers)", fontsize=9)
            ax.set_ylabel("ACF", fontsize=9)
            ax.set_ylim(-1.1, 1.1)
            ax.set_title(
                f"{model} / {sig_label}\n"
                f"VIF={vif:.2f}  eff_N≈{eff_n}  (n={n})",
                fontsize=8,
            )
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        print()  # blank line between models

    fig.suptitle(
        "Layer-wise Autocorrelation — LRD, Patching Recovery, LoRA Effectiveness\n"
        "Tomato dashes = 1/e threshold for decorrelation length",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(OUT_PDF, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {OUT_PDF}")

    json.dump(results, open(OUT_FILE, "w"), indent=2)
    print(f"Results saved: {OUT_FILE}")

    # summary table
    print("\n=== Effective N Summary ===")
    print(f"{'Model':<10} {'Signal':<24} {'n':>4} {'decor_lag':>10} {'VIF':>6} {'eff_N':>7}")
    print("-" * 65)
    for model, sigs in results.items():
        for sig_label, info in sigs.items():
            print(
                f"{model:<10} {sig_label:<24} "
                f"{info['n']:>4} {info['decorrelation_lag']:>10} "
                f"{info['vif']:>6.2f} {info['effective_n']:>7.1f}"
            )
        print()


if __name__ == "__main__":
    main()
