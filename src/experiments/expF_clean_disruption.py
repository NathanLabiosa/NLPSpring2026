# === lib path shim (added by repo reorg) ===
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir, 'lib'))
# === end lib path shim ===

"""
expF_clean_disruption.py - LoRA-Induced Clean Disruption Measurement

For each trained LoRA window (no new training), measures how much the adapter
rewrites clean-input representations.

Procedure for each LoRA window:
  1. Run 200 clean GSM8K examples through the base model -> h_base[L]
  2. Run the same examples through the LoRA-adapted model -> h_lora[L]
  3. At every layer L (from the LoRA window onwards), compute mean cosine
     distance between h_base[L] and h_lora[L]

Hypothesis: early-window LoRA causes high downstream disruption; optimal-window
LoRA causes low downstream disruption.

Usage:
    python expF_clean_disruption.py --model phi35     # Phi3.5 (p2_sweep_L*)
    python expF_clean_disruption.py --model llama3    # Llama-3 (llama_sweep_L*)
    python expF_clean_disruption.py --model mistral   # Mistral (mistral_sweep_L*)

Output:
  expF_clean_disruption_{model}.json
  figures/disruption_{model}.pdf
  figures/disruption_{model}_scatter.pdf
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

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset

# Apply publication styling
from figure_style import apply_style, remove_spines, add_legend_below
apply_style()

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG_DIR = os.path.join(ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

for p in [ROOT, os.path.join(ROOT, "Phi3.5")]:
    if p not in sys.path:
        sys.path.insert(0, p)

MODEL_CFGS = {
    "phi35": {
        "model_id": "microsoft/Phi-3.5-mini-instruct",
        "display_name": "Phi-3.5",
        "system_prompt": "Solve the math problem step by step. The last line must be '#### ANSWER'.",
        "trust_remote_code": False,
        "attn_implementation": "sdpa",
        "sweep_dirs": [
            ("phase2_mmlu_phi35_L00-04_seed42", (0, 4)),
            ("phase2_mmlu_phi35_L05-09_seed42", (5, 9)),
            ("phase2_mmlu_phi35_L10-14_seed42", (10, 14)),
            ("phase2_mmlu_phi35_L15-19_seed42", (15, 19)),
            ("phase2_mmlu_phi35_L20-24_seed42", (20, 24)),
            ("phase2_mmlu_phi35_L27-31_seed42", (27, 31)),
        ],
        "sweep_deltas": {
            "phase2_mmlu_phi35_L00-04_seed42": None,
            "phase2_mmlu_phi35_L05-09_seed42": None,
            "phase2_mmlu_phi35_L10-14_seed42": None,
            "phase2_mmlu_phi35_L15-19_seed42": None,
            "phase2_mmlu_phi35_L20-24_seed42": None,
            "phase2_mmlu_phi35_L27-31_seed42": None,
        },
    },
    "llama3": {
        "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "display_name": "Llama-3",
        "system_prompt": "Solve the math problem step by step. The last line must be '#### ANSWER'.",
        "trust_remote_code": False,
        "attn_implementation": "sdpa",
        "sweep_dirs": [
            ("phase2_scatter_llama3_8b_L00-04_seed42", (0, 4)),
            ("phase2_scatter_llama3_8b_L05-09_seed42", (5, 9)),
            ("phase2_scatter_llama3_8b_L10-14_seed42", (10, 14)),
            ("phase2_scatter_llama3_8b_L15-19_seed42", (15, 19)),
            ("phase2_scatter_llama3_8b_L20-24_seed42", (20, 24)),
            ("phase2_scatter_llama3_8b_L27-31_seed42", (27, 31)),
        ],
        "sweep_deltas": {
            "phase2_scatter_llama3_8b_L00-04_seed42": None,
            "phase2_scatter_llama3_8b_L05-09_seed42": None,
            "phase2_scatter_llama3_8b_L10-14_seed42": None,
            "phase2_scatter_llama3_8b_L15-19_seed42": None,
            "phase2_scatter_llama3_8b_L20-24_seed42": None,
            "phase2_scatter_llama3_8b_L27-31_seed42": None,
        },
    },
    "mistral": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "display_name": "Mistral",
        "system_prompt": "Solve the math problem step by step. The last line must be '#### ANSWER'.",
        "trust_remote_code": False,
        "attn_implementation": "sdpa",
        "sweep_dirs": [
            ("phase2_scatter_mistral_7b_v03_L00-04_seed42", (0, 4)),
            ("phase2_scatter_mistral_7b_v03_L05-09_seed42", (5, 9)),
            ("phase2_scatter_mistral_7b_v03_L10-14_seed42", (10, 14)),
            ("phase2_scatter_mistral_7b_v03_L15-19_seed42", (15, 19)),
            ("phase2_scatter_mistral_7b_v03_L20-24_seed42", (20, 24)),
            ("phase2_scatter_mistral_7b_v03_L27-31_seed42", (27, 31)),
        ],
        "sweep_deltas": {
            "phase2_scatter_mistral_7b_v03_L00-04_seed42": None,
            "phase2_scatter_mistral_7b_v03_L05-09_seed42": None,
            "phase2_scatter_mistral_7b_v03_L10-14_seed42": None,
            "phase2_scatter_mistral_7b_v03_L15-19_seed42": None,
            "phase2_scatter_mistral_7b_v03_L20-24_seed42": None,
            "phase2_scatter_mistral_7b_v03_L27-31_seed42": None,
        },
    },
    "qwen": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "display_name": "Qwen2.5-7B",
        "system_prompt": "Solve the math problem step by step. The last line must be '#### ANSWER'.",
        # Qwen-7B needs fp32: bf16 collapses the rank-4 LoRA delta to zero.
        "torch_dtype": "float32",
        "use_chat_template": True,
        "trust_remote_code": False,
        "attn_implementation": "eager",
        "sweep_dirs": [
            ("phase2_mmlu_qwen2.5_7b_L00-04_seed42", (0, 4)),
            ("phase2_mmlu_qwen2.5_7b_L05-09_seed42", (5, 9)),
            ("phase2_mmlu_qwen2.5_7b_L08-11_seed42", (8, 11)),
            ("phase2_mmlu_qwen2.5_7b_L15-19_seed42", (15, 19)),
            ("phase2_mmlu_qwen2.5_7b_L20-23_seed42", (20, 23)),
            ("phase2_mmlu_qwen2.5_7b_L24-27_seed42", (24, 27)),
        ],
        "sweep_deltas": {
            "phase2_mmlu_qwen2.5_7b_L00-04_seed42": None,
            "phase2_mmlu_qwen2.5_7b_L05-09_seed42": None,
            "phase2_mmlu_qwen2.5_7b_L08-11_seed42": None,
            "phase2_mmlu_qwen2.5_7b_L15-19_seed42": None,
            "phase2_mmlu_qwen2.5_7b_L20-23_seed42": None,
            "phase2_mmlu_qwen2.5_7b_L24-27_seed42": None,
        },
    },
    "gemma": {
        "model_id": "google/gemma-2-9b",
        "display_name": "Gemma-2-9B",
        "system_prompt": "Solve the math problem step by step. The last line must be '#### ANSWER'.",
        "torch_dtype": "bfloat16",
        "use_chat_template": False,
        "trust_remote_code": False,
        "attn_implementation": "eager",
        "sweep_dirs": [
            ("gemma2_9b_sweep_L00_05", (0, 5)),
            ("gemma2_9b_sweep_L06_11", (6, 11)),
            ("gemma2_9b_sweep_L12_17", (12, 17)),
            ("gemma2_9b_sweep_L18_23", (18, 23)),
            ("gemma2_9b_sweep_L24_29", (24, 29)),
            ("gemma2_9b_sweep_L30_35", (30, 35)),
            ("gemma2_9b_sweep_L36_41", (36, 41)),
        ],
        "sweep_deltas": {
            "gemma2_9b_sweep_L00_05": None,
            "gemma2_9b_sweep_L06_11": None,
            "gemma2_9b_sweep_L12_17": None,
            "gemma2_9b_sweep_L18_23": None,
            "gemma2_9b_sweep_L24_29": None,
            "gemma2_9b_sweep_L30_35": None,
            "gemma2_9b_sweep_L36_41": None,
        },
    },
}

SW = None  # resolved from --weights_dir in main()


def format_prompt(tokenizer, question: str, system_prompt: str,
                  use_chat_template: bool = True) -> str:
    if not use_chat_template:
        return f"{system_prompt}\n\nQuestion: {question}\n\nAnswer:"
    messages = [
        {"role": "system",  "content": system_prompt},
        {"role": "user",    "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


@torch.no_grad()
def compute_disruption_curves(
    base_model, peft_model, tokenizer,
    questions: list, system_prompt: str,
    device: str, batch_size: int = 4,
    use_chat_template: bool = True,
) -> dict:
    """
    Returns per-layer mean cosine distance between base and adapted hidden
    states on clean inputs: {layer_idx: mean_dist}.
    """
    n_layers   = None
    layer_sums = None
    total      = 0

    for i in range(0, len(questions), batch_size):
        batch    = questions[i : i + batch_size]
        prompts  = [format_prompt(tokenizer, q, system_prompt, use_chat_template)
                    for q in batch]
        enc      = tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=512
        ).to(device)
        mask = enc["attention_mask"]
        B    = len(batch)

        # Base model
        with peft_model.disable_adapter():
            out_base = peft_model(**enc, output_hidden_states=True)
        h_base = [hs.detach().clone() for hs in out_base.hidden_states]
        del out_base

        # Adapted model
        out_lora = peft_model(**enc, output_hidden_states=True)
        h_lora   = [hs.detach().clone() for hs in out_lora.hidden_states]
        del out_lora

        if n_layers is None:
            n_layers   = len(h_base)
            layer_sums = [0.0] * n_layers

        for l in range(n_layers):
            T = min(h_base[l].shape[1], h_lora[l].shape[1], mask.shape[1])
            hb = h_base[l][:, :T, :].float()
            hl = h_lora[l][:, :T, :].float()
            m  = mask[:, :T].float()

            cos_sim  = F.cosine_similarity(hb, hl, dim=-1)
            dist     = 1.0 - cos_sim
            # Zero-norm hidden states at padding positions produce NaN cos_sim;
            # zero them out so they don't poison the mask-weighted mean.
            dist     = torch.where(torch.isnan(dist), torch.zeros_like(dist), dist)
            denom    = m.sum()
            if denom > 1e-8:
                layer_sums[l] += ((dist * m).sum() / denom).item() * B

        total += B
        del h_base, h_lora
        torch.cuda.empty_cache()

    return {l: layer_sums[l] / total for l in range(n_layers)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      required=True,
                        choices=list(MODEL_CFGS.keys()),
                        help="Model to evaluate")
    parser.add_argument("--n_samples",  type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--weights_dir", type=str, default=None,
                        help="Root dir containing adapter checkpoint subdirs "
                             "(default: ROOT/stabilizer_weights)")
    parser.add_argument("--reuse_cache", action="store_true",
                        help="Skip GPU computation if cached JSON exists, just regenerate figures")
    args = parser.parse_args()

    random.seed(args.seed)
    cfg     = MODEL_CFGS[args.model]
    display_name = cfg["display_name"]
    device  = "cuda" if torch.cuda.is_available() else "cpu"

    global SW
    SW = args.weights_dir if args.weights_dir else os.path.join(ROOT, "stabilizer_weights")

    cache_path = os.path.join(ROOT, "results", f"expF_clean_disruption_{args.model}.json")
    if args.reuse_cache and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        with open(cache_path) as f:
            all_results = json.load(f)
        n_samples_cached = all_results.get("_n_samples", args.n_samples)
        print(f"  Loaded {len(all_results) - 1} sweep windows (n_samples={n_samples_cached})")
    else:
        # GPU computation required
        print("Loading GSM8K test set...")
        ds        = load_dataset("openai/gsm8k", "main", split="test")
        items     = list(ds)
        random.shuffle(items)
        questions = [it["question"] for it in items[: args.n_samples]]
        print(f"  Using {len(questions)} clean questions")

        print(f"\nLoading base model: {cfg['model_id']}")
        tokenizer = AutoTokenizer.from_pretrained(
            cfg["model_id"], trust_remote_code=True
        )
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        dtype_str = cfg.get("torch_dtype", "float16")
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                 "float32": torch.float32}[dtype_str]

        all_results = {"_n_samples": args.n_samples}

        for sweep_name, (lo, hi) in cfg["sweep_dirs"]:
            adapter_path = os.path.join(SW, sweep_name, "lora_final")
            if not os.path.exists(adapter_path):
                print(f"\n[SKIP] {sweep_name}: adapter not found at {adapter_path}")
                continue

            print(f"\n--- Window {sweep_name} (layers {lo}-{hi}) ---")
            # Reload base model fresh each iteration: PEFT mutates base_model
            # in-place when wrapping with LoRA, so calling PeftModel.from_pretrained
            # in a loop on the same base_model can produce NaN forwards on some
            # configs (Qwen-7B fp32 in particular). Fresh load per window is the
            # robust fix.
            base_model = AutoModelForCausalLM.from_pretrained(
                cfg["model_id"],
                device_map={"": 0},
                torch_dtype=dtype,
                trust_remote_code=cfg.get("trust_remote_code", False),
                attn_implementation=cfg.get("attn_implementation", "sdpa"),
            )
            base_model.eval()
            peft_model = PeftModel.from_pretrained(base_model, adapter_path)
            peft_model.eval()

            disruption = compute_disruption_curves(
                base_model, peft_model, tokenizer,
                questions, cfg["system_prompt"],
                device, batch_size=args.batch_size,
                use_chat_template=cfg.get("use_chat_template", True),
            )

            total_disruption = sum(disruption.values())
            downstream_lrd   = {
                str(l): v for l, v in disruption.items() if l >= lo
            }
            print(f"  Total downstream disruption (layers {lo}+): "
                  f"{sum(downstream_lrd.values()):.4f}")

            all_results[sweep_name] = {
                "window":       [lo, hi],
                "disruption":   {str(k): v for k, v in disruption.items()},
                "total_disruption": total_disruption,
                "downstream_disruption": sum(v for l, v in disruption.items() if l >= lo),
                "acc_delta": cfg["sweep_deltas"].get(sweep_name, None),
            }

            del peft_model, base_model
            torch.cuda.empty_cache()

        # Save results for future reuse
        with open(cache_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nCached data saved to {cache_path}")

    # Get n_samples for titles
    n_samples = all_results.pop("_n_samples", args.n_samples)

    # Figure 1: disruption curves per window
    n_layers = max(
        max(int(k) for k in r["disruption"]) + 1
        for r in all_results.values()
    )
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for (sweep_name, res), color in zip(all_results.items(), colors):
        lo, hi = res["window"]
        lrd_vals = [res["disruption"].get(str(l), 0.0) for l in range(n_layers)]
        ax.plot(lrd_vals, label=f"L{lo}-{hi}",
                color=color, linewidth=2.5)
        # Removed shading per user request

    ax.set_xlabel("Layer", fontsize=26)
    ax.set_ylabel("Mean cosine distance (base vs LoRA, clean input)", fontsize=14)
    ax.set_title(
        f"LoRA-Induced Clean Disruption - {display_name} / GSM8K (n={n_samples})",
        fontsize=26
    )
    ax.tick_params(axis='both', labelsize=22)

    remove_spines(ax)
    ax.grid(True, axis='y', alpha=0.3, linewidth=0.5)

    # Legend in upper-left with ncol=3
    ax.legend(loc='upper left', fontsize=22, framealpha=0.9, ncol=3)

    plt.tight_layout()
    fig1_path = os.path.join(FIG_DIR, f"disruption_{args.model}.pdf")
    plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved figure: {fig1_path}")

    # Figure 2: scatter - total disruption vs acc delta (taller aspect)
    fig2, ax2 = plt.subplots(figsize=(7, 6))
    x_vals, y_vals, labels = [], [], []
    for sweep_name, res in all_results.items():
        if res["acc_delta"] is not None:
            x_vals.append(res["downstream_disruption"])
            y_vals.append(res["acc_delta"])
            labels.append(f"L{res['window'][0]}-{res['window'][1]}")

    # Larger markers (user requested 60-80)
    ax2.scatter(x_vals, y_vals, s=80, zorder=3, color='tab:blue', edgecolor='black')

    # Text labels with bumped font
    for x, y, lbl in zip(x_vals, y_vals, labels):
        ax2.annotate(lbl, (x, y), textcoords="offset points",
                     xytext=(6, 6), fontsize=11)

    # Fit and show regression line with slope annotation
    if len(x_vals) >= 2:
        m, b = np.polyfit(x_vals, y_vals, 1)
        xs = np.linspace(min(x_vals) - 0.01, max(x_vals) + 0.01, 50)
        ax2.plot(xs, m * xs + b, "r-", linewidth=2.0, alpha=0.7)
        # Slope annotation position: upper-left for Phi, bottom-left for others
        if args.model == "phi35":
            ax2.text(0.05, 0.95, f"slope = {m:.1f}",
                     transform=ax2.transAxes, fontsize=11,
                     color='red', ha='left', va='top')
        else:
            ax2.text(0.05, 0.05, f"slope = {m:.1f}",
                     transform=ax2.transAxes, fontsize=11,
                     color='red', ha='left', va='bottom')

    ax2.axhline(0, color="black", linewidth=1.0, linestyle="--")
    ax2.set_xlabel("Total downstream clean disruption", fontsize=13)
    ax2.set_ylabel("Mean perturbed accuracy gain (%)", fontsize=13)
    ax2.set_title(f"Clean Disruption vs LoRA Effectiveness - {display_name}", fontsize=13)
    ax2.tick_params(axis='both', labelsize=11)

    remove_spines(ax2)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig2_path = os.path.join(FIG_DIR, f"disruption_{args.model}_scatter.pdf")
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved scatter: {fig2_path}")

    out_path = os.path.join(ROOT, "results", f"expF_clean_disruption_{args.model}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
