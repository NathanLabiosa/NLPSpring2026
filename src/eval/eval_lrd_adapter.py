# === lib path shim (added by repo reorg) ===
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir, 'lib'))
# === end lib path shim ===

"""
eval_lrd_adapter.py — Layer-wise LRD with and without the v17 LoRA adapter.

For each perturbation condition in EVAL_CONDITIONS, computes per-layer mean
cosine distance between:

  (a) clean-input hidden states              ← reference
  (b) perturbed-input hidden states, NO adapter   (baseline LRD)
  (c) perturbed-input hidden states, WITH adapter (adapted LRD)

If the adapter reduces LRD specifically at the causal layers (0–4 for Phi-3.5),
this closes the mechanistic narrative:
  diagnostics identified the problem
  → adapter placed at causal window
  → adapter measurably reduces the divergence it was designed to fix
  → task performance improves

Usage:
    python eval_lrd_adapter.py \\
        --checkpoint_dir ./stabilizer_weights/lora_v17/lora_final \\
        --n_samples 100 \\
        --output ./stabilizer_weights/lora_v17/lrd_adapter_analysis.json
"""

import os, sys, json, random, argparse
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_lrd_stabilizer import (
    EVAL_CONDITIONS, format_question_prompt, _import_perturbation_engine,
)

PerturbationEngine = _import_perturbation_engine()


def mean_cosine_dist(h1: torch.Tensor, h2: torch.Tensor,
                     mask: torch.Tensor) -> float:
    """
    Mean cosine distance between h1 and h2 over non-padding positions.
    h1, h2: [B, T, D]   mask: [B, T]  (1 = real token, 0 = pad)
    Returns a Python float.
    """
    # clamp T in case shapes differ (clean vs noisy may tokenize differently)
    T = min(h1.shape[1], h2.shape[1], mask.shape[1])
    h1 = h1[:, :T, :].float()
    h2 = h2[:, :T, :].float()
    m  = mask[:, :T].float()

    cos_sim = F.cosine_similarity(h1, h2, dim=-1)   # [B, T]
    dist    = 1.0 - cos_sim                          # [B, T]

    denom = m.sum()
    if denom < 1e-8:
        return 0.0
    return ((dist * m).sum() / denom).item()


@torch.no_grad()
def compute_lrd_curves(model, tokenizer, perturber, n_samples: int,
                       device) -> dict:
    from datasets import load_dataset

    test_ds    = load_dataset("openai/gsm8k", "main", split="test")
    test_items = list(test_ds)
    random.shuffle(test_items)
    test_items = test_items[:n_samples]

    all_results = {}

    for method, rate in EVAL_CONDITIONS:
        cond_name = f"{method}_{int(rate*100)}pct" if rate > 0 else "clean_baseline"
        print(f"\n--- {cond_name} ---")

        n_layers     = None
        base_sums    = None   # list[float], length = n_layers
        adapted_sums = None
        count        = 0

        for i in range(0, len(test_items), 4):
            batch = test_items[i : i + 4]
            B     = len(batch)

            clean_prompts = [format_question_prompt(tokenizer, it["question"])
                             for it in batch]
            if rate > 0:
                noisy_prompts = [
                    format_question_prompt(
                        tokenizer, perturber.apply(it["question"], method, rate)
                    )
                    for it in batch
                ]
            else:
                noisy_prompts = clean_prompts

            enc_c = tokenizer(clean_prompts, return_tensors="pt", padding=True,
                              truncation=True, max_length=512).to(device)
            enc_n = tokenizer(noisy_prompts, return_tensors="pt", padding=True,
                              truncation=True, max_length=512).to(device)

            # (a) Clean → base model
            with model.disable_adapter():
                out_clean = model(**enc_c, output_hidden_states=True)
            h_clean = [hs.detach() for hs in out_clean.hidden_states]
            del out_clean

            # (b) Perturbed → base model (no adapter)
            with model.disable_adapter():
                out_base = model(**enc_n, output_hidden_states=True)
            h_base = [hs.detach() for hs in out_base.hidden_states]
            del out_base

            # (c) Perturbed → adapted model
            out_adp = model(**enc_n, output_hidden_states=True)
            h_adp   = [hs.detach() for hs in out_adp.hidden_states]
            del out_adp

            # Initialise accumulators on first batch
            if n_layers is None:
                n_layers     = len(h_clean)
                base_sums    = [0.0] * n_layers
                adapted_sums = [0.0] * n_layers

            # Use clean attention mask for pooling (consistent reference length)
            mask = enc_c["attention_mask"]   # [B, T_clean]

            for li in range(n_layers):
                base_sums[li]    += mean_cosine_dist(h_base[li], h_clean[li], mask) * B
                adapted_sums[li] += mean_cosine_dist(h_adp[li],  h_clean[li], mask) * B

            count += B
            del h_clean, h_base, h_adp
            torch.cuda.empty_cache()

        lrd_base    = [s / count for s in base_sums]
        lrd_adapted = [s / count for s in adapted_sums]
        reduction   = [b - a for b, a in zip(lrd_base, lrd_adapted)]

        peak_base_layer = int(lrd_base.index(max(lrd_base)))
        peak_adp_layer  = int(lrd_adapted.index(max(lrd_adapted)))
        max_red_layer   = int(reduction.index(max(reduction)))

        print(f"  Peak baseline LRD : {max(lrd_base):.4f} at layer {peak_base_layer}")
        print(f"  Peak adapted  LRD : {max(lrd_adapted):.4f} at layer {peak_adp_layer}")
        print(f"  Max LRD reduction : {max(reduction):.4f} at layer {max_red_layer}")

        # Early-layer summary (causal window layers 0–4 for Phi-3.5)
        causal_base    = sum(lrd_base[:5]) / 5
        causal_adapted = sum(lrd_adapted[:5]) / 5
        causal_red     = causal_base - causal_adapted
        pct = causal_red / causal_base * 100 if causal_base > 1e-8 else 0.0
        print(f"  Avg layers 0–4: baseline={causal_base:.4f}  "
              f"adapted={causal_adapted:.4f}  "
              f"reduction={causal_red:.4f} ({pct:.1f}%)")

        all_results[cond_name] = {
            "lrd_base":           [round(x, 6) for x in lrd_base],
            "lrd_adapted":        [round(x, 6) for x in lrd_adapted],
            "lrd_reduction":      [round(x, 6) for x in reduction],
            "peak_base_layer":    peak_base_layer,
            "peak_adapted_layer": peak_adp_layer,
            "max_reduction_layer": max_red_layer,
            "causal_window_avg": {
                "baseline":  round(causal_base, 6),
                "adapted":   round(causal_adapted, 6),
                "reduction": round(causal_red, 6),
                "pct":       round(pct, 2),
            },
        }

    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str,
                        default="./stabilizer_weights/lora_v17/lora_final")
    parser.add_argument("--base_model", type=str,
                        default="microsoft/Phi-3.5-mini-instruct")
    parser.add_argument("--n_samples", type=int, default=100,
                        help="GSM8K test questions per condition")
    parser.add_argument("--output", type=str,
                        default="./stabilizer_weights/lora_v17/lrd_adapter_analysis.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"\nLoading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map={"": 0},
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    print(f"Loading LoRA adapter: {args.checkpoint_dir}")
    model = PeftModel.from_pretrained(base_model, args.checkpoint_dir)
    model.eval()

    perturber = PerturbationEngine()

    print(f"\nComputing LRD curves (n={args.n_samples} per condition)...")
    results = compute_lrd_curves(model, tokenizer, perturber, args.n_samples, device)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {args.output}")

    # Final summary table
    print(f"\n{'=' * 70}")
    print(f"{'Condition':<22}  {'Causal LRD (base)':>18}  "
          f"{'Causal LRD (adp)':>18}  {'Reduction':>10}")
    print("-" * 70)
    for cond, r in results.items():
        cw = r["causal_window_avg"]
        print(f"{cond:<22}  {cw['baseline']:>17.4f}  "
              f"{cw['adapted']:>17.4f}  "
              f"{cw['reduction']:>+9.4f} ({cw['pct']:>+5.1f}%)")


if __name__ == "__main__":
    main()
