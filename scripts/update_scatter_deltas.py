#!/usr/bin/env python3
"""
update_scatter_deltas.py — Merge GSM8K eval results into disruption JSON caches.

After running:
  1. run_scatter_eval.sh (phi35, qwen) and/or run_scatter_train.sh (llama, mistral)
  2. run_scatter_disruption.sh for each model

Run this script to write the mean acc_delta per window into
results/expF_clean_disruption_{model}.json (the acc_delta field
that expF uses for the scatter y-axis).

Then regenerate figures:
  LEROBOT_PY=/home/nathanlabiosa/miniforge3/envs/lerobot_smol/bin/python
  CUDA_VISIBLE_DEVICES="" $LEROBOT_PY src/experiments/expF_clean_disruption.py --model phi35 --reuse_cache
  ... (repeat for qwen, llama, mistral)
"""

import json
import math
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCATTER_DIR = os.path.join(ROOT, "results", "fixed_harness", "v2", "scatter")
RESULTS_DIR = os.path.join(ROOT, "results")

MODEL_MAP = {
    "phi35":   ("phi35",          "phase2_scatter_phi35"),
    "qwen":    ("qwen2.5_7b",     "phase2_scatter_qwen2.5_7b"),
    "llama3":  ("llama3_8b",      "phase2_scatter_llama3_8b"),
    "mistral": ("mistral_7b_v03", "phase2_scatter_mistral_7b_v03"),
}


def mean_std(vals):
    if not vals:
        return None, None
    n = len(vals)
    m = sum(vals) / n
    if n == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in vals) / (n - 1)
    return m, math.sqrt(var)


def load_scatter_eval(slug):
    """Return {window_tag: [delta_seed42, delta_seed43, delta_seed44]} from scatter/ dir."""
    pat = re.compile(rf"^{re.escape(slug)}_(.+?)_seed(\d+)\.json$")
    by_window = {}
    for fname in sorted(os.listdir(SCATTER_DIR)):
        m = pat.match(fname)
        if not m:
            continue
        win_tag, seed = m.group(1), int(m.group(2))
        fpath = os.path.join(SCATTER_DIR, fname)
        try:
            d = json.load(open(fpath))
            delta = d.get("mean_perturbed_delta")
        except Exception as e:
            print(f"  [WARN] Could not read {fname}: {e}")
            continue
        if delta is None:
            continue
        by_window.setdefault(win_tag, []).append((seed, delta))
    return by_window


def update_model(model_key):
    slug, ckpt_prefix = MODEL_MAP[model_key]
    cache_path = os.path.join(RESULTS_DIR, f"expF_clean_disruption_{model_key}.json")

    if not os.path.exists(cache_path):
        print(f"[{model_key}] No disruption cache at {cache_path} — run disruption first.")
        return

    cache = json.load(open(cache_path))

    eval_data = load_scatter_eval(slug)
    if not eval_data:
        print(f"[{model_key}] No scatter eval results in {SCATTER_DIR} — run eval first.")
        return

    updated = 0
    for ckpt_name, entry in cache.items():
        if ckpt_name == "_n_samples":
            continue
        # Extract window tag from checkpoint name: phase2_mmlu_phi35_L10-14_seed42 → L10-14
        m = re.search(r"_(L\d+-\d+)_seed\d+$", ckpt_name)
        if not m:
            print(f"  [WARN] Cannot parse window from key: {ckpt_name}")
            continue
        win_tag = m.group(1)

        if win_tag not in eval_data:
            print(f"  [WARN] No eval data for window {win_tag} (key: {ckpt_name})")
            continue

        deltas = [d for _, d in sorted(eval_data[win_tag])]
        mean, std = mean_std(deltas)
        if mean is None:
            continue

        old_val = entry.get("acc_delta")
        entry["acc_delta"] = round(mean, 4)
        seeds_str = "/".join(str(s) for s, _ in sorted(eval_data[win_tag]))
        print(f"  {ckpt_name}: acc_delta {old_val} → {mean:+.2f} ± {std:.2f} "
              f"(seeds {seeds_str}, n={len(deltas)})")
        updated += 1

    json.dump(cache, open(cache_path, "w"), indent=2)
    print(f"[{model_key}] Updated {updated} windows → {cache_path}")


def main():
    if not os.path.isdir(SCATTER_DIR):
        print(f"Scatter eval dir not found: {SCATTER_DIR}")
        print("Run run_scatter_eval.sh / run_scatter_train.sh first.")
        return

    for model_key in MODEL_MAP:
        print(f"\n=== {model_key} ===")
        update_model(model_key)

    print("\nDone. Regenerate scatter figures with:")
    print("  LEROBOT_PY=/home/nathanlabiosa/miniforge3/envs/lerobot_smol/bin/python")
    print("  for m in phi35 qwen llama3 mistral; do")
    print("    CUDA_VISIBLE_DEVICES='' $LEROBOT_PY src/experiments/expF_clean_disruption.py \\")
    print("      --model $m --reuse_cache")
    print("  done")


if __name__ == "__main__":
    main()
