# === lib path shim (added by repo reorg) ===
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir, 'lib'))
# === end lib path shim ===

"""
expC_capacity_metrics.py - Layer-Wise Capacity / Slack Metrics

Computes intrinsic per-layer properties of the base model (no perturbation)
that may predict where LoRA is effective.

Metrics (on 200 clean GSM8K examples):
  C1  residual update ratio   - ||h_L - h_{L-1}|| / ||h_{L-1}||
  C2  attention entropy       - entropy of per-head attention distribution
                                 averaged over positions and examples
  C3  weight effective rank   - exp(entropy(sigma / sum(sigma))) for q_proj/v_proj
                                 at each layer (static, no forward pass)
  C4  gradient norm           - ||dL_CE / dW||_F for q_proj/v_proj
                                 (requires backward pass on clean loss)

All metrics are then correlated (Spearman rho) with LoRA effectiveness from
the existing window sweep.

Usage:
    python expC_capacity_metrics.py --model phi35
    python expC_capacity_metrics.py --model llama3
    python expC_capacity_metrics.py --model mistral

Output:
    expC_capacity_metrics_{model}.json
    figures/capacity_metrics_{model}.pdf
    expC_correlation_table_{model}.json
"""

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.interpolate import interp1d
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Apply publication styling
from figure_style import apply_style, remove_spines
apply_style()

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG_DIR = os.path.join(ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

for p in [ROOT, os.path.join(ROOT, "Phi3.5")]:
    if p not in sys.path:
        sys.path.insert(0, p)

SW = os.path.join(ROOT, "stabilizer_weights")

MODEL_CFGS = {
    "phi35": {
        "model_id": "microsoft/Phi-3.5-mini-instruct",
        "display_name": "Phi-3.5",
        "system_prompt": "Solve the math problem step by step. The last line must be '#### ANSWER'.",
        "sweep_window_deltas": {
            2:  4.53,
            7:  4.80,
            12: 5.47,
            17: 7.27,
            22: 3.83,
            29: 1.10,
        },
        "q_weight_name": "model.layers.{l}.self_attn.qkv_proj",
        "q_fused": True,
    },
    "llama3": {
        "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "display_name": "Llama-3",
        "system_prompt": "Solve the math problem step by step. The last line must be '#### ANSWER'.",
        "sweep_window_deltas": {
            2:  -1.87,
            7:  -3.67,
            12: -4.57,
            17:  0.00,
            22:  1.17,
            29:  0.00,
        },
        "q_weight_name": "model.layers.{l}.self_attn.q_proj",
        "q_fused": False,
    },
    "mistral": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "display_name": "Mistral",
        "system_prompt": "Solve the math problem step by step. The last line must be '#### ANSWER'.",
        "sweep_window_deltas": {
            2:  -8.30,
            7:  -9.43,
            12: -3.97,
            17: -10.50,
            22:  0.13,
            29: -1.43,
        },
        "q_weight_name": "model.layers.{l}.self_attn.q_proj",
        "q_fused": False,
    },
}


def format_prompt(tokenizer, question: str, system_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def effective_rank(W: torch.Tensor) -> float:
    """exp(entropy(normalised singular values)) - a real-valued rank proxy."""
    with torch.no_grad():
        sv = torch.linalg.svdvals(W.float())
        sv = sv[sv > 0]
        p  = sv / sv.sum()
        ent = -(p * torch.log(p + 1e-12)).sum()
        return float(torch.exp(ent).item())


def attn_entropy(attn_weights: torch.Tensor) -> float:
    """
    attn_weights: [B, H, T, T] attention probabilities.
    Returns mean entropy (nats) averaged over (B, H, T).
    """
    p = attn_weights.float().clamp(min=1e-9)
    ent = -(p * torch.log(p)).sum(dim=-1)
    return float(ent.mean().item())


def spearman_rho(a, b) -> dict:
    a, b = np.array(a), np.array(b)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return {"rho": float("nan"), "p": float("nan")}
    rho, p = spearmanr(a[mask], b[mask])
    return {"rho": round(float(rho), 4), "p": round(float(p), 4)}


def compute_c3_weight_rank(model, n_layers: int, q_fused: bool) -> list:
    ranks = []
    for l in range(n_layers):
        try:
            attn = model.model.layers[l].self_attn
            if q_fused:
                W = attn.qkv_proj.weight
                q_dim  = W.shape[0] // 3
                W_q = W[:q_dim, :]
                W_v = W[2 * q_dim:, :]
            else:
                W_q = attn.q_proj.weight
                W_v = attn.v_proj.weight
            rq = effective_rank(W_q)
            rv = effective_rank(W_v)
            ranks.append((rq + rv) / 2.0)
        except AttributeError:
            ranks.append(float("nan"))
    return ranks


def compute_c1_update_ratio(hidden_states_list: list) -> list:
    """
    hidden_states_list: list of [n_layers+1] per-example pooled arrays.
    Returns per-layer mean update ratio ||h_L - h_{L-1}|| / ||h_{L-1}||.
    """
    n_layers = hidden_states_list[0].shape[0] - 1
    ratios   = np.zeros(n_layers)
    for hs in hidden_states_list:
        for l in range(n_layers):
            h_prev = hs[l]
            h_curr = hs[l + 1]
            denom  = np.linalg.norm(h_prev)
            if denom > 1e-8:
                ratios[l] += np.linalg.norm(h_curr - h_prev) / denom
    ratios /= len(hidden_states_list)
    return ratios.tolist()


def compute_c4_gradient_norms(model, tokenizer, questions: list,
                               system_prompt: str, device: str,
                               q_fused: bool, batch_size: int = 4) -> list:
    """
    Run forward+backward on clean CE loss, accumulate ||dL/dW||_F for
    q_proj and v_proj at each layer.
    """
    n_layers = len(model.model.layers)
    grad_sums = [0.0] * n_layers
    count     = 0

    for i in range(0, len(questions), batch_size):
        batch   = questions[i : i + batch_size]
        prompts = [format_prompt(tokenizer, q, system_prompt) for q in batch]
        enc     = tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=512
        ).to(device)

        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100

        model.zero_grad()
        with torch.enable_grad():
            out  = model(**enc, labels=labels)
            loss = out.loss
            loss.backward()

        for l in range(n_layers):
            attn = model.model.layers[l].self_attn
            try:
                if q_fused:
                    W_q = attn.qkv_proj.weight
                    q_dim = W_q.shape[0] // 3
                    gq = W_q.grad[:q_dim, :] if W_q.grad is not None else None
                    gv = W_q.grad[2 * q_dim:, :] if W_q.grad is not None else None
                else:
                    gq = attn.q_proj.weight.grad
                    gv = attn.v_proj.weight.grad

                norms = []
                if gq is not None:
                    norms.append(gq.float().norm().item())
                if gv is not None:
                    norms.append(gv.float().norm().item())
                if norms:
                    grad_sums[l] += float(np.mean(norms))
            except AttributeError:
                pass

        model.zero_grad()
        count += len(batch)
        torch.cuda.empty_cache()

    return [s / max(count, 1) for s in grad_sums]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     required=True, choices=list(MODEL_CFGS.keys()))
    parser.add_argument("--n_samples", type=int,    default=200)
    parser.add_argument("--batch_size",type=int,    default=4)
    parser.add_argument("--seed",      type=int,    default=42)
    parser.add_argument("--skip_grad", action="store_true",
                        help="Skip C4 (gradient norms) to save memory/time")
    parser.add_argument("--reuse_cache", action="store_true",
                        help="Skip GPU computation if cached JSON exists, just regenerate figures")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg    = MODEL_CFGS[args.model]
    display_name = cfg["display_name"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Check for cached data
    cache_path = os.path.join(ROOT, f"expC_capacity_metrics_{args.model}.json")
    if args.reuse_cache and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        with open(cache_path) as f:
            cached = json.load(f)
        n_samples = cached["n_samples"]
        n_layers = cached["n_layers"]
        metrics = {k: np.array(v) for k, v in cached["metrics"].items()}
        lora_delta = np.array(cached["lora_delta_interp"])
        corr_table = cached["correlations"]
        print(f"  Loaded metrics for {n_layers} layers (n_samples={n_samples})")
    else:
        print("Loading GSM8K...")
        ds        = load_dataset("openai/gsm8k", "main", split="test")
        items     = list(ds)
        random.shuffle(items)
        questions = [it["question"] for it in items[: args.n_samples]]
        n_samples = len(questions)
        print(f"  Using {n_samples} clean examples")

        print(f"\nLoading {cfg['model_id']} ...")
        tokenizer = AutoTokenizer.from_pretrained(
            cfg["model_id"], trust_remote_code=True
        )
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            cfg["model_id"],
            device_map={"": 0},
            torch_dtype=torch.float16,
            trust_remote_code=True,
            attn_implementation="eager",
        )
        n_layers = len(model.model.layers)
        print(f"  n_layers = {n_layers}")

        print("\nComputing C3: weight effective rank...")
        model.eval()
        c3_rank = compute_c3_weight_rank(model, n_layers, cfg["q_fused"])
        print(f"  Done. Sample: {[round(r, 1) for r in c3_rank[:5]]}...")

        print("\nForward passes for C1 (residual ratio) and C2 (attention entropy)...")
        all_pooled_hs = []
        c2_entropy_sums = np.zeros(n_layers)
        c2_count        = 0

        with torch.no_grad():
            for i in range(0, len(questions), args.batch_size):
                batch   = questions[i : i + args.batch_size]
                prompts = [format_prompt(tokenizer, q, cfg["system_prompt"])
                           for q in batch]
                enc     = tokenizer(
                    prompts, return_tensors="pt", padding=True,
                    truncation=True, max_length=512
                ).to(device)

                out = model(**enc, output_hidden_states=True, output_attentions=True)

                mask = enc["attention_mask"]
                for b in range(len(batch)):
                    m  = mask[b].bool()
                    hs = np.stack(
                        [out.hidden_states[l][b][m].float().mean(dim=0).cpu().numpy()
                         for l in range(n_layers + 1)],
                        axis=0
                    )
                    all_pooled_hs.append(hs)

                for l in range(n_layers):
                    if out.attentions[l] is not None:
                        c2_entropy_sums[l] += attn_entropy(out.attentions[l])
                        c2_count += 1

                del out
                torch.cuda.empty_cache()

        c2_count_per_layer = len(questions) // args.batch_size
        c2_entropy = (c2_entropy_sums / max(c2_count_per_layer, 1)).tolist()

        c1_update_ratio = compute_c1_update_ratio(all_pooled_hs)
        print(f"  C1 sample: {[round(v, 4) for v in c1_update_ratio[:5]]}...")
        print(f"  C2 sample: {[round(v, 4) for v in c2_entropy[:5]]}...")

        c4_grad_norms = None
        if not args.skip_grad:
            print("\nComputing C4: gradient norms (forward+backward)...")
            grad_questions = questions[:50]
            c4_grad_norms  = compute_c4_gradient_norms(
                model, tokenizer, grad_questions,
                cfg["system_prompt"], device, cfg["q_fused"],
                batch_size=2,
            )
            model.zero_grad()
            torch.cuda.empty_cache()
            print(f"  C4 sample: {[round(v, 6) for v in c4_grad_norms[:5]]}...")
        else:
            print("\nSkipping C4 (--skip_grad set)")
            c4_grad_norms = [float("nan")] * n_layers

        # Correlate with LoRA effectiveness
        wc   = np.array(sorted(cfg["sweep_window_deltas"].keys()), dtype=float)
        wd   = np.array([cfg["sweep_window_deltas"][k] for k in sorted(cfg["sweep_window_deltas"])])
        interp_fn = interp1d(wc, wd, kind="linear",
                          bounds_error=False, fill_value=(wd[0], wd[-1]))
        lora_delta = interp_fn(np.arange(n_layers, dtype=float))

        metrics = {
            "C1_update_ratio": c1_update_ratio,
            "C2_attn_entropy": c2_entropy,
            "C3_weight_rank":  c3_rank,
            "C4_grad_norm":    c4_grad_norms if c4_grad_norms else [float("nan")] * n_layers,
        }
        corr_table = {}
        for mname, vals in metrics.items():
            corr_table[mname] = spearman_rho(vals, lora_delta.tolist())
            print(f"  {mname} x LoRA delta: rho={corr_table[mname]['rho']:+.3f}, "
                  f"p={corr_table[mname]['p']:.4f}")

    # Two-panel figure (C3 and C4 only)
    layers    = np.arange(n_layers)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    panel_cfg = [
        ("C3_weight_rank",  "Weight effective rank (Q & V proj)", axes[0]),
        ("C4_grad_norm",    r"Gradient norm $\|\partial L/\partial W\|_F$", axes[1]),
    ]

    for mname, ylabel, ax in panel_cfg:
        vals  = np.array(metrics[mname], dtype=float)
        color = "tab:blue"
        ax.plot(layers, vals, color=color, linewidth=2.5)
        ax.set_ylabel(ylabel, color=color, fontsize=11)
        ax.tick_params(axis="y", labelcolor=color)

        ax2 = ax.twinx()
        ax2.plot(layers, lora_delta, color="tab:red",
                 linewidth=2.5, linestyle="--", alpha=0.7)
        ax2.set_ylabel(r"LoRA $\Delta$ acc (interp)", color="tab:red", fontsize=11)
        ax2.tick_params(axis="y", labelcolor="tab:red")

        rho = corr_table[mname]
        ax.set_title(f"{mname}  ($\\rho$={rho['rho']:+.3f}, p={rho['p']:.3f})", fontsize=12)
        ax.set_xlabel("Layer", fontsize=11)
        remove_spines(ax)

    plt.suptitle(f"Capacity Metrics - {display_name} / GSM8K clean (n={n_samples})", fontsize=13)
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, f"capacity_metrics_{args.model}.pdf")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved figure: {fig_path}")

    # Only save JSON if we computed fresh (not from cache)
    if not args.reuse_cache:
        out = {
            "n_samples":        n_samples,
            "n_layers":         n_layers,
            "metrics":          {k: [round(float(v), 6) if not np.isnan(v) else None
                                      for v in vals]
                                  for k, vals in metrics.items()},
            "lora_delta_interp": [round(float(v), 4) for v in lora_delta],
            "correlations":      corr_table,
        }
        json_path = os.path.join(ROOT, f"expC_capacity_metrics_{args.model}.json")
        with open(json_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved: {json_path}")

        corr_path = os.path.join(ROOT, f"expC_correlation_table_{args.model}.json")
        with open(corr_path, "w") as f:
            json.dump(corr_table, f, indent=2)
        print(f"Saved: {corr_path}")


if __name__ == "__main__":
    main()
