# === lib path shim (added by repo reorg) ===
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir, 'lib'))
# === end lib path shim ===

"""
predict_optimal_window.py — C3/C4 Predictive Validation (Experiment 4).

For each new model:
  1. Load LRD diagnostic data → classify regime (spike-and-suppress vs late-accumulation)
  2. Compute C3 (weight effective rank) and C4 (gradient norm) per layer from frozen model
  3. Based on regime + C3/C4 values, PREDICT which 5-layer window should be optimal:
       spike-and-suppress: lowest C3 + highest C4 → most "plastic" layer under stress
       late-accumulation:  highest C3 + lowest C4 → most capacity with least saturation
  4. Save timestamped prediction to results/predictions/{model}_{timestamp}.json
  5. If sweep results already exist, check prediction vs. actual and report hit/miss

Usage:
    # Step A (before sweep): generates prediction
    python predict_optimal_window.py --model TinyLlama --compute_c3c4

    # Step B (after sweep): checks prediction against sweep results
    python predict_optimal_window.py --model TinyLlama --check_prediction

    # Both steps at once (if sweep data already available)
    python predict_optimal_window.py --model TinyLlama --compute_c3c4 --check_prediction
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SW   = os.path.join(ROOT, "stabilizer_weights")

# ─────────────────────────────────────────────────────────────────────────────
# Model configurations for new models
# ─────────────────────────────────────────────────────────────────────────────

MODEL_CFGS = {
    "TinyLlama": {
        "model_id":    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "n_layers":    22,
        "lrd_path":    "Tinymodels/llama/lrd_results/tinyllama_gsm8k/raw_gsm8k.json",
        "q_weight_key": "model.layers.{l}.self_attn.q_proj",
        "q_fused":     False,
        "sweep_dirs": [
            ("tinyllama_sweep_L00_03", (0,  3)),
            ("tinyllama_sweep_L04_07", (4,  7)),
            ("tinyllama_sweep_L08_11", (8,  11)),
            ("tinyllama_sweep_L12_15", (12, 15)),
            ("tinyllama_sweep_L16_19", (16, 19)),
        ],
    },
    "Gemma2": {
        "model_id":    "google/gemma-2-9b",
        "n_layers":    42,
        "lrd_path":    "models/gemma2/lrd_results/gemma2_9b_gsm8k/raw_gsm8k.json",
        "q_weight_key": "model.layers.{l}.self_attn.q_proj",
        "q_fused":     False,
        "sweep_dirs": [
            ("gemma2_9b_sweep_L00_05", (0,  5)),
            ("gemma2_9b_sweep_L06_11", (6,  11)),
            ("gemma2_9b_sweep_L12_17", (12, 17)),
            ("gemma2_9b_sweep_L18_23", (18, 23)),
            ("gemma2_9b_sweep_L24_29", (24, 29)),
            ("gemma2_9b_sweep_L30_35", (30, 35)),
            ("gemma2_9b_sweep_L36_41", (36, 41)),
        ],
    },
    "Qwen25": {
        "model_id":    "Qwen/Qwen2.5-7B-Instruct",
        "n_layers":    28,
        "lrd_path":    "models/qwen2.5/lrd_results/qwen_gsm8k/raw_gsm8k.json",
        "q_weight_key": "model.layers.{l}.self_attn.q_proj",
        "q_fused":     False,
        "sweep_dirs": [
            ("qwen_sweep_L00_03", (0,  3)),
            ("qwen_sweep_L04_07", (4,  7)),
            ("qwen_sweep_L08_11", (8,  11)),
            ("qwen_sweep_L12_15", (12, 15)),
            ("qwen_sweep_L16_19", (16, 19)),
            ("qwen_sweep_L20_23", (20, 23)),
            ("qwen_sweep_L24_27", (24, 27)),
        ],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Regime classification from LRD data
# ─────────────────────────────────────────────────────────────────────────────

def classify_regime(lrd_path: str) -> dict:
    """
    Classify LRD profile as spike-and-suppress or late-accumulation.

    spike-and-suppress: LRD peaks in first half, recovers in second half.
    late-accumulation:  LRD grows monotonically (or peaks in second half).
    """
    if not os.path.exists(lrd_path):
        raise FileNotFoundError(f"LRD data not found: {lrd_path}")

    raw = json.load(open(lrd_path))
    all_profiles = [r["lrd_profile"] for recs in raw.values() for r in recs]
    mean_lrd_full = np.array(all_profiles).mean(0)

    # Drop the final layer: it's the output/logits head, which spikes for every
    # model regardless of regime and dominates argmax. Classify on the hidden
    # states only (layers 0..n-2).
    mean_lrd = mean_lrd_full[:-1]

    n  = len(mean_lrd)
    q  = max(1, n // 4)
    early   = float(mean_lrd[:q].mean())
    late    = float(mean_lrd[-q:].mean())
    max_l   = int(mean_lrd.argmax())
    ratio   = late / early if early > 1e-8 else 1.0

    if max_l < n // 2 and ratio < 0.7:
        regime = "spike-and-suppress"
        description = (f"LRD peaks at layer {max_l} (first half, output layer excluded), "
                       f"recovery ratio={ratio:.2f} (< 0.70)")
    else:
        regime = "late-accumulation"
        description = (f"LRD peaks at layer {max_l} (output layer excluded), "
                       f"late/early ratio={ratio:.2f} (≥ 0.70 or peak in second half)")

    return {
        "regime":      regime,
        "description": description,
        "mean_lrd":    mean_lrd.tolist(),
        "max_layer":   max_l,
        "late_early_ratio": round(ratio, 3),
    }


# ─────────────────────────────────────────────────────────────────────────────
# C3 (weight effective rank) and C4 (gradient norm) computation
# ─────────────────────────────────────────────────────────────────────────────

def effective_rank(W: torch.Tensor) -> float:
    """exp(entropy(σ / Σσ)) for the weight matrix W."""
    s = torch.linalg.svdvals(W.float())
    s = s[s > 1e-8]
    if len(s) == 0:
        return 1.0
    p   = s / s.sum()
    ent = -(p * torch.log(p + 1e-12)).sum()
    return float(torch.exp(ent).item())


def compute_c3_c4(model_id: str, cfg: dict, n_samples: int = 50,
                  device: str = "cuda") -> dict:
    """
    C3: weight effective rank per layer (q_proj/v_proj), static.
    C4: gradient norm ||∂L_CE / ∂W||_F for q_proj/v_proj on clean GSM8K.
    """
    print(f"\nLoading {model_id} for C3/C4 computation...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use bfloat16 for better numerical stability (avoids NaN in loss/gradients)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    # C4 needs a backward pass on a frozen 9B+ model on a V100 32GB — without
    # gradient checkpointing this OOMs. Checkpointing requires use_cache=False.
    # Qwen2.5-7B is smaller (7B), try without gradient checkpointing first
    is_qwen = "Qwen" in model_id
    if hasattr(model, "gradient_checkpointing_enable") and not is_qwen:
        model.gradient_checkpointing_enable()
    if hasattr(model, "config"):
        model.config.use_cache = False

    # Use eval mode to avoid NaN from dropout, but gradients still flow
    model.eval()

    n_layers = cfg["n_layers"]
    q_key    = cfg["q_weight_key"]

    # ── C3: weight effective rank (static, no forward pass needed) ────────────
    c3_per_layer = {}
    for l in range(n_layers):
        key = q_key.format(l=l)
        try:
            W = dict(model.named_parameters())[key + ".weight"]
            if cfg["q_fused"]:
                # Fused QKV — take only the Q portion (first 1/3 rows)
                d = W.shape[0] // 3
                W = W[:d]
            c3_per_layer[l] = effective_rank(W)
        except KeyError:
            # Try v_proj fallback
            vkey = key.replace("q_proj", "v_proj")
            try:
                W = dict(model.named_parameters())[vkey + ".weight"]
                c3_per_layer[l] = effective_rank(W)
            except KeyError:
                print(f"  [WARN] Layer {l}: neither {key} nor v_proj found — C3=NaN")
                c3_per_layer[l] = float("nan")

    print(f"  C3 (effective rank) computed for {len(c3_per_layer)} layers.")

    # ── C4: gradient norm on clean GSM8K ─────────────────────────────────────
    print(f"  Computing C4 (gradient norms) on {n_samples} clean examples...")
    ds    = load_dataset("openai/gsm8k", "main", split="train")
    items = list(ds)[:n_samples]

    sys.path.insert(0, os.path.join(ROOT, "Phi3.5"))
    from train_lrd_stabilizer import GSM8K_SYSTEM_PROMPT, format_question_prompt
    from data_loader import GSM8K_FEW_SHOT

    # Base models (e.g. Gemma-2-9b) don't have a chat template — fall back
    # to few-shot prompting, matching the LRD diagnostic pipeline.
    is_base_model = tokenizer.chat_template is None
    if is_base_model:
        print("  [info] Tokenizer has no chat_template — using few-shot prompting.")

    # Freeze all params except q_proj / v_proj to avoid allocating gradient
    # buffers for the full 9B model (otherwise OOM on a 32GB V100).
    n_trainable = 0
    for name, param in model.named_parameters():
        if ("q_proj" in name or "v_proj" in name) and name.endswith(".weight"):
            param.requires_grad = True
            n_trainable += 1
        else:
            param.requires_grad = False
    print(f"  [info] {n_trainable} params with requires_grad=True (q/v_proj weights only).")

    # Gradient checkpointing needs the embedding output to require grad so the
    # graph connects from frozen embeddings to the first checkpointed layer.
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    grad_accum = {l: [] for l in range(n_layers)}

    print(f"  [DEBUG] Processing {len(items)} items, n_layers={n_layers}, q_key template='{q_key}'")

    for item in items:
        if is_base_model:
            prompt = GSM8K_FEW_SHOT.format(question=item["question"])
        else:
            prompt = format_question_prompt(tokenizer, item["question"])
        # Tighter truncation on base-model few-shot prompts to keep activation
        # memory manageable during the backward pass.
        enc = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=1024 if is_base_model else 512).to(device)
        # Shift labels for CE
        out    = model(**enc, labels=enc["input_ids"])
        loss   = out.loss

        # Check for NaN loss before backward
        if math.isnan(loss.item()):
            if item == items[0]:
                print(f"  [WARN] First example has NaN loss, skipping")
            continue

        loss.backward()

        # Build param_dict once per example
        param_dict = dict(model.named_parameters())

        # Check first example only
        if item == items[0]:
            first_key = q_key.format(l=0) + ".weight"
            if first_key in param_dict:
                W0 = param_dict[first_key]
                print(f"  [DEBUG] First example: {first_key} exists, requires_grad={W0.requires_grad}, grad is None={W0.grad is None}, loss={loss.item():.6f}")
                if W0.grad is not None:
                    grad_fp32 = W0.grad.detach().to(torch.float32)
                    gnorm = grad_fp32.norm('fro').item()
                    print(f"  [DEBUG] First example grad norm (fp32): {gnorm}, is_nan={math.isnan(gnorm)}")
                    print(f"  [DEBUG] First example grad min/max: {W0.grad.min().item():.6e} / {W0.grad.max().item():.6e}")
            else:
                print(f"  [DEBUG] First example: {first_key} NOT FOUND in model parameters")
                # Print first few q_proj keys
                q_keys = [k for k in param_dict.keys() if 'q_proj' in k and 'weight' in k][:3]
                print(f"  [DEBUG] Sample q_proj keys: {q_keys}")
        for l in range(n_layers):
            key = q_key.format(l=l)
            try:
                W   = param_dict[key + ".weight"]
                if W.grad is not None:
                    # Convert to float32 before computing norm to avoid float16 overflow/underflow
                    grad_fp32 = W.grad.detach().to(torch.float32)
                    gnorm = float(grad_fp32.norm("fro").item())
                    if not math.isnan(gnorm):
                        grad_accum[l].append(gnorm)
                elif l == 0:  # Debug: print why grad is None for first layer only
                    print(f"  [DEBUG] Layer {l} ({key}.weight): grad is None, requires_grad={W.requires_grad}")
            except KeyError:
                if l == 0:  # Debug: print key error for first layer only
                    print(f"  [DEBUG] Layer {l}: KeyError for {key}.weight")

        model.zero_grad()

    c4_per_layer = {
        l: float(np.mean(gnorms)) if gnorms else float("nan")
        for l, gnorms in grad_accum.items()
    }
    print(f"  C4 computed for {sum(1 for v in c4_per_layer.values() if not math.isnan(v))} layers.")

    # Unload model to free GPU memory
    del model
    torch.cuda.empty_cache()

    return {
        "c3": {int(k): round(v, 4) for k, v in c3_per_layer.items()},
        "c4": {int(k): round(v, 6) for k, v in c4_per_layer.items()},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Prediction logic
# ─────────────────────────────────────────────────────────────────────────────

def predict_optimal_window(regime: str, c3_layers: dict, c4_layers: dict,
                           sweep_windows: list) -> dict:
    """
    For each sweep window, compute a composite score from C3 and C4.

    spike-and-suppress: best window = lowest C3 × highest C4
        → most plastic layers under perturbation stress
        → score = (1 - c3_norm) * c4_norm  (higher = more favourable)

    late-accumulation: best window = highest C3 × lowest C4
        → most capacity with least saturation
        → score = c3_norm * (1 - c4_norm)  (higher = more favourable)

    Returns a ranked list of windows with their predicted scores.
    """
    # Map each window to its mean C3 and C4
    window_data = []
    for dir_suffix, (lo, hi) in sweep_windows:
        layers    = list(range(lo, hi + 1))
        c3_vals   = [c3_layers.get(l, float("nan")) for l in layers]
        c4_vals   = [c4_layers.get(l, float("nan")) for l in layers]
        c3_mean   = float(np.nanmean(c3_vals))
        c4_mean   = float(np.nanmean(c4_vals))
        window_data.append({
            "dir_suffix": dir_suffix,
            "lo": lo, "hi": hi,
            "c3_mean": c3_mean,
            "c4_mean": c4_mean,
        })

    # Normalise c3 and c4 to [0, 1] across windows
    c3_arr = np.array([w["c3_mean"] for w in window_data])
    c4_arr = np.array([w["c4_mean"] for w in window_data])

    def norm01(arr):
        lo_v, hi_v = np.nanmin(arr), np.nanmax(arr)
        if hi_v - lo_v < 1e-8:
            return np.zeros_like(arr)
        return (arr - lo_v) / (hi_v - lo_v)

    c3_norm = norm01(c3_arr)
    c4_norm = norm01(c4_arr)

    for i, w in enumerate(window_data):
        if regime == "spike-and-suppress":
            score = (1 - c3_norm[i]) * c4_norm[i]
        else:
            score = c3_norm[i] * (1 - c4_norm[i])
        w["c3_norm"] = round(float(c3_norm[i]), 4)
        w["c4_norm"] = round(float(c4_norm[i]), 4)
        w["score"]   = round(float(score), 4)

    # Rank by score descending
    ranked = sorted(window_data, key=lambda w: -w["score"])
    for rank, w in enumerate(ranked, 1):
        w["predicted_rank"] = rank

    return {
        "regime": regime,
        "rule":   ("lowest C3, highest C4 (spike-and-suppress)"
                   if regime == "spike-and-suppress"
                   else "highest C3, lowest C4 (late-accumulation)"),
        "windows_ranked": ranked,
        "predicted_best": ranked[0]["dir_suffix"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Check prediction against sweep results
# ─────────────────────────────────────────────────────────────────────────────

def check_prediction(prediction: dict, sweep_windows: list) -> dict:
    """Load sweep eval_results.json for each window, compare to prediction."""
    print("\n=== Checking Prediction vs Sweep Results ===")

    window_deltas = {}
    for dir_suffix, (lo, hi) in sweep_windows:
        ep = os.path.join(SW, dir_suffix, "eval_results.json")
        if not os.path.exists(ep):
            print(f"  [MISSING] {ep}")
            continue
        ev   = json.load(open(ep))
        vals = [v["delta"] for k, v in ev.items()
                if k != "clean_baseline" and "delta" in v]
        if vals:
            window_deltas[dir_suffix] = round(float(np.mean(vals)), 3)

    if not window_deltas:
        print("  No sweep results available yet.")
        return {"status": "no_sweep_data"}

    # Rank by actual delta descending
    actual_ranked = sorted(window_deltas, key=lambda k: -window_deltas[k])
    actual_best   = actual_ranked[0]

    pred_best     = prediction["predicted_best"]
    pred_rank_of_actual = actual_ranked.index(actual_best) + 1
    actual_rank_of_pred = (actual_ranked.index(pred_best) + 1
                           if pred_best in actual_ranked else None)

    print(f"  Predicted best:  {pred_best}")
    print(f"  Actual best:     {actual_best} (Δ={window_deltas[actual_best]:+.2f}%)")
    if actual_rank_of_pred is not None:
        print(f"  Predicted window ranked #{actual_rank_of_pred} out of {len(actual_ranked)}")

    hit  = (pred_best == actual_best)
    near = (actual_rank_of_pred is not None and actual_rank_of_pred <= 2)

    print(f"  Outcome: {'EXACT HIT' if hit else ('NEAR-HIT (top-2)' if near else 'MISS')}")

    # Spearman correlation between predicted and actual rankings
    predicted_ranks = {w["dir_suffix"]: w["predicted_rank"]
                       for w in prediction["windows_ranked"]
                       if w["dir_suffix"] in window_deltas}
    actual_ranks    = {k: i + 1 for i, k in enumerate(actual_ranked)
                       if k in predicted_ranks}

    common = list(predicted_ranks.keys() & actual_ranks.keys())
    if len(common) >= 3:
        from scipy.stats import spearmanr
        pred_r   = [predicted_ranks[k] for k in common]
        actual_r = [actual_ranks[k]    for k in common]
        rho, pval = spearmanr(pred_r, actual_r)
        print(f"  Rank correlation (predicted vs actual): ρ={rho:+.3f}, p={pval:.4f}")
    else:
        rho, pval = float("nan"), float("nan")
        print("  Too few common windows for rank correlation.")

    return {
        "status":              "checked",
        "predicted_best":      pred_best,
        "actual_best":         actual_best,
        "actual_rank_of_pred": actual_rank_of_pred,
        "hit":                 hit,
        "near_hit":            near,
        "window_deltas":       window_deltas,
        "actual_ranking":      actual_ranked,
        "rank_spearman_rho":   round(float(rho), 4),
        "rank_spearman_p":     round(float(pval), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",             type=str, required=True,
                   choices=list(MODEL_CFGS.keys()))
    p.add_argument("--compute_c3c4",      action="store_true",
                   help="Compute C3/C4 from frozen model (needs GPU)")
    p.add_argument("--check_prediction",  action="store_true",
                   help="Check existing prediction against sweep results")
    p.add_argument("--n_samples",         type=int, default=50,
                   help="GSM8K examples for C4 gradient computation")
    args = p.parse_args()

    cfg       = MODEL_CFGS[args.model]
    pred_dir  = os.path.join(ROOT, "results/predictions")
    os.makedirs(pred_dir, exist_ok=True)
    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    pred_path = os.path.join(pred_dir, f"{args.model}_{ts}.json")

    # ── Step A: classify regime + compute C3/C4 + write prediction ────────────
    if args.compute_c3c4:
        print(f"\n=== Step A: Regime Classification + C3/C4 for {args.model} ===")

        # 1. Classify regime
        print("\n[1] Classifying regime from LRD diagnostic...")
        regime_info = classify_regime(os.path.join(ROOT, cfg["lrd_path"]))
        print(f"  → regime: {regime_info['regime']}")
        print(f"  → {regime_info['description']}")

        # 2. Compute C3/C4
        print("\n[2] Computing C3 (weight rank) and C4 (gradient norm)...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        metrics = compute_c3_4(cfg["model_id"], cfg, n_samples=args.n_samples,
                                device=device)

        # 3. Generate prediction
        print("\n[3] Generating prediction...")
        prediction = predict_optimal_window(
            regime_info["regime"],
            metrics["c3"],
            metrics["c4"],
            cfg["sweep_dirs"],
        )

        print(f"\n  Predicted optimal window: {prediction['predicted_best']}")
        print(f"  Rule applied: {prediction['rule']}")
        print("\n  Window rankings:")
        for w in prediction["windows_ranked"]:
            print(f"    #{w['predicted_rank']}  {w['dir_suffix']:<30} "
                  f"score={w['score']:.3f}  C3_norm={w['c3_norm']:.3f}  "
                  f"C4_norm={w['c4_norm']:.3f}")

        # 4. Save prediction
        record = {
            "model":          args.model,
            "model_id":       cfg["model_id"],
            "timestamp":      ts,
            "regime_info":    regime_info,
            "c3_per_layer":   metrics["c3"],
            "c4_per_layer":   metrics["c4"],
            "prediction":     prediction,
            "sweep_checked":  False,
            "check_results":  None,
        }
        json.dump(record, open(pred_path, "w"), indent=2)
        print(f"\nPrediction saved: {pred_path}")
        print("IMPORTANT: submit the sweep job AFTER this file is written.")

    # ── Step B: check prediction against sweep results ─────────────────────────
    if args.check_prediction:
        print(f"\n=== Step B: Checking Prediction for {args.model} ===")

        # Find most recent prediction file
        pred_files = sorted([f for f in os.listdir(pred_dir)
                             if f.startswith(args.model + "_") and f.endswith(".json")])
        if not pred_files:
            print("No prediction file found. Run --compute_c3c4 first.")
            return

        latest = os.path.join(pred_dir, pred_files[-1])
        print(f"Loading prediction from: {latest}")
        record = json.load(open(latest))

        check_results = check_prediction(
            record["prediction"],
            cfg["sweep_dirs"],
        )

        record["sweep_checked"] = True
        record["check_results"] = check_results
        json.dump(record, open(latest, "w"), indent=2)
        print(f"\nUpdated: {latest}")


# Alias for internal call with correct function name
def compute_c3_4(model_id, cfg, n_samples, device):
    return compute_c3_c4(model_id, cfg, n_samples, device)


if __name__ == "__main__":
    main()
