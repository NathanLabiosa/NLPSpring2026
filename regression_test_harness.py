"""
regression_test_harness.py — Harness drift detector.

Re-runs eval_fixed_harness.py on the Qwen stabilizer checkpoint
(qwen25_7b_L24_27_stab/lora_final) and asserts the mean perturbed Δ is
within ±1pp of the recorded value (+0.17 pp). Run before every batch of
jobs to catch silent harness changes.

Recorded reference (results/fixed_harness/v1/three_checkpoint_comparison/
fixed_harness_qwen_stab.json):
  clean_baseline acc_no_adapter:  89.0%
  mean perturbed Δ:               +0.17 pp

Tolerance: ±1.0 pp on mean perturbed Δ, ±1.5 pp on clean baseline.

Usage:
    sbatch submit_regression_test.slurm
    # OR (interactive, on a GPU node):
    python regression_test_harness.py
"""

import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.abspath(__file__))

REFERENCE_FILE = os.path.join(
    ROOT, "results", "fixed_harness", "v1",
    "three_checkpoint_comparison", "fixed_harness_qwen_stab.json",
)

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
CHECKPOINT = os.path.join(ROOT, "stabilizer_weights", "qwen25_7b_L24_27_stab", "lora_final")

MEAN_DELTA_TOLERANCE = 1.0   # pp
BASELINE_TOLERANCE   = 1.5   # pp
N_SAMPLES = 500
SEED = 42


def load_reference():
    with open(REFERENCE_FILE) as f:
        ref = json.load(f)
    res = ref["results"]
    perturbed = [v["delta"] for k, v in res.items() if k != "clean_baseline"]
    mean_d = sum(perturbed) / len(perturbed)
    clean = res["clean_baseline"]["acc_no_adapter"]
    return clean, mean_d


def run_eval():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        out_path = tmp.name
    cmd = [
        sys.executable,
        os.path.join(ROOT, "eval_fixed_harness.py"),
        "--base_model", BASE_MODEL,
        "--checkpoint", CHECKPOINT,
        "--n_samples", str(N_SAMPLES),
        "--seed", str(SEED),
        "--output_file", out_path,
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)
    with open(out_path) as f:
        out = json.load(f)
    os.unlink(out_path)
    res = out["results"]
    perturbed = [v["delta"] for k, v in res.items() if k != "clean_baseline"]
    mean_d = sum(perturbed) / len(perturbed)
    clean = res["clean_baseline"]["acc_no_adapter"]
    return clean, mean_d


def main():
    ref_clean, ref_mean = load_reference()
    print(f"Reference: clean={ref_clean:.2f}%  mean perturbed Δ={ref_mean:+.2f} pp")
    print(f"Tolerance: clean ±{BASELINE_TOLERANCE} pp,  mean Δ ±{MEAN_DELTA_TOLERANCE} pp")

    cur_clean, cur_mean = run_eval()
    print(f"\nCurrent:   clean={cur_clean:.2f}%  mean perturbed Δ={cur_mean:+.2f} pp")
    print(f"Drift:     clean Δ={cur_clean - ref_clean:+.2f} pp,  "
          f"mean Δ shift={cur_mean - ref_mean:+.2f} pp")

    failures = []
    if abs(cur_clean - ref_clean) > BASELINE_TOLERANCE:
        failures.append(f"clean baseline drifted {cur_clean - ref_clean:+.2f} pp "
                        f"(tol ±{BASELINE_TOLERANCE})")
    if abs(cur_mean - ref_mean) > MEAN_DELTA_TOLERANCE:
        failures.append(f"mean perturbed Δ drifted {cur_mean - ref_mean:+.2f} pp "
                        f"(tol ±{MEAN_DELTA_TOLERANCE})")

    if failures:
        print("\nREGRESSION FAILED:")
        for msg in failures:
            print(f"  - {msg}")
        sys.exit(1)

    print("\nREGRESSION PASSED.")


if __name__ == "__main__":
    main()
