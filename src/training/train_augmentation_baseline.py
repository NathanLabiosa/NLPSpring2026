# === lib path shim (added by repo reorg) ===
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir, 'lib'))
# === end lib path shim ===

"""
train_augmentation_baseline.py — Data Augmentation Baseline (Experiment 3).

Trains LoRA adapters on Phi-3.5-mini-instruct using ONLY cross-entropy loss
on perturbed training examples — no stability loss. This is the standard
"just do data augmentation with LoRA" approach.

Two variants:
  (A) Layers 15–19 only  (the same window as the best stabilizer result)
  (B) All 32 layers       (rank 4, same method — placement comparison)

Direct comparisons enabled:
  (A) vs stabilizer at layers 15–19  → does stability loss add value?
  (A) vs (B)                          → does placement finding hold without stab loss?
  Both vs no-adapter baseline

Usage:
    python train_augmentation_baseline.py --variant window   # layers 15-19
    python train_augmentation_baseline.py --variant all      # all 32 layers
    python train_augmentation_baseline.py --variant both     # run both sequentially

Output:
    stabilizer_weights/aug_baseline_L15_19/eval_results.json
    stabilizer_weights/aug_baseline_all32/eval_results.json
    aug_baseline_comparison.json     (summary table written after both runs)
"""

import os
import sys
import json
import random
import argparse
from typing import List

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in [ROOT, os.path.join(ROOT, "Phi3.5")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from train_lrd_stabilizer import (
    TrainingPair, collate_batch, tokenize_pair,
    GSM8K_SYSTEM_PROMPT, format_question_prompt,
    _extract_gsm8k_answer, _extract_gsm8k_truth, _gsm8k_correct,
    accuracy_loss,
    EVAL_CONDITIONS, KNOWN_BASELINES,
)
from perturbations import PerturbationEngine

MODEL_ID = "microsoft/Phi-3.5-mini-instruct"

TRAINING_POOL = [
    ("typos",      0.05),
    ("ocr",        0.05),
    ("ocr",        0.15),
    ("speech",     0.10),
    ("homophones", 0.30),
    ("whitespace", 0.10),
    ("case",       0.10),
]

VARIANT_CONFIGS = {
    "window": {
        "name":        "aug_baseline_L15_19",
        "layer_start": 15,
        "layer_end":   19,
        "n_layers":    5,
        "all_layers":  False,
        "output_dir":  os.path.join(ROOT, "stabilizer_weights", "aug_baseline_L15_19"),
        "label":       "Augmentation-only (layers 15–19)",
    },
    "all": {
        "name":        "aug_baseline_all32",
        "layer_start": 0,
        "layer_end":   31,
        "n_layers":    32,
        "all_layers":  True,
        "output_dir":  os.path.join(ROOT, "stabilizer_weights", "aug_baseline_all32"),
        "label":       "Augmentation-only (all 32 layers)",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def build_pairs(perturber: PerturbationEngine,
                n_per_condition: int = 300,
                clean_fraction: float = 0.20) -> List[TrainingPair]:
    print("Building augmentation training pairs...")
    ds    = load_dataset("openai/gsm8k", "main", split="train")
    items = list(ds)
    random.shuffle(items)

    pairs = []
    for method, rate in TRAINING_POOL:
        for i in range(n_per_condition):
            item = items[i % len(items)]
            noisy_q = perturber.apply(item["question"], method, rate)
            pairs.append(TrainingPair(
                clean_question=item["question"],
                noisy_question=noisy_q,
                clean_answer=item["answer"],
                perturbation=method,
                rate=rate,
                is_clean=False,
            ))

    n_clean = int(len(pairs) * clean_fraction)
    for i in range(n_clean):
        item = items[i % len(items)]
        pairs.append(TrainingPair(
            clean_question=item["question"],
            noisy_question=item["question"],
            clean_answer=item["answer"],
            perturbation="none",
            rate=0.0,
            is_clean=True,
        ))

    random.shuffle(pairs)
    print(f"  {len(pairs)} pairs ({n_clean} clean).")
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation (same format as existing sweeps)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, tokenizer, perturber, n_samples: int, device,
             output_dir: str) -> dict:
    print("\n" + "=" * 60 + "\nEVALUATION\n" + "=" * 60)
    model.eval()

    ds    = load_dataset("openai/gsm8k", "main", split="test")
    items = list(ds)
    random.shuffle(items)
    items = items[:n_samples]

    gen_kw = dict(max_new_tokens=300, do_sample=False,
                  pad_token_id=tokenizer.pad_token_id)

    def gen(prompts, use_adapter):
        out = []
        for i in range(0, len(prompts), 4):
            b = prompts[i:i + 4]
            enc = tokenizer(b, return_tensors="pt", padding=True,
                            truncation=True, max_length=512).to(device)
            if use_adapter:
                ids = model.generate(**enc, **gen_kw)
            else:
                with model.disable_adapter():
                    ids = model.generate(**enc, **gen_kw)
            new = ids[:, enc["input_ids"].shape[1]:]
            out.extend(tokenizer.batch_decode(new, skip_special_tokens=True))
        return out

    results = {}
    truths  = [_extract_gsm8k_truth(it["answer"]) for it in items]

    for method, rate in EVAL_CONDITIONS:
        cond = f"{method}_{int(rate * 100)}pct" if rate > 0 else "clean_baseline"
        prompts_p = [format_question_prompt(tokenizer,
                                            perturber.apply(it["question"], method, rate)
                                            if rate > 0 else it["question"])
                     for it in items]
        prompts_c = [format_question_prompt(tokenizer, it["question"]) for it in items]

        preds_no  = gen(prompts_p, False)
        preds_yes = gen(prompts_p, True)
        preds_cln = gen(prompts_c, False) if method != "none" else preds_no

        acc_no   = sum(_gsm8k_correct(_extract_gsm8k_answer(p), t)
                       for p, t in zip(preds_no, truths)) / len(truths) * 100
        acc_yes  = sum(_gsm8k_correct(_extract_gsm8k_answer(p), t)
                       for p, t in zip(preds_yes, truths)) / len(truths) * 100
        acc_cln  = sum(_gsm8k_correct(_extract_gsm8k_answer(p), t)
                       for p, t in zip(preds_cln, truths)) / len(truths) * 100
        delta    = acc_yes - acc_no
        known    = KNOWN_BASELINES.get(method)
        known_s  = f"  [known={known:.1f}%]" if known else ""

        print(f"  {cond:<22} no={acc_no:.1f}% yes={acc_yes:.1f}% Δ={delta:+.1f}%{known_s}")

        results[cond] = {
            "method": method, "rate": rate,
            "n_samples": n_samples,
            "acc_no_adapter":     round(acc_no, 2),
            "acc_with_adapter":   round(acc_yes, 2),
            "acc_clean_baseline": round(acc_cln, 2),
            "delta":              round(delta, 2),
            "known_baseline":     known,
        }

    path = os.path.join(output_dir, "eval_results.json")
    json.dump(results, open(path, "w"), indent=2)
    print(f"Saved: {path}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def run_variant(variant_key: str, args):
    vcfg = VARIANT_CONFIGS[variant_key]
    print("\n" + "=" * 65)
    print(f"VARIANT: {vcfg['label']}")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(vcfg["output_dir"], exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map={"": 0},
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="eager",
    )

    # LoRA — Phi-3.5 uses qkv_proj (fused)
    if vcfg["all_layers"]:
        # Get actual layer count from model
        n_actual = len(base_model.model.layers)
        layer_list = list(range(n_actual))
    else:
        layer_list = list(range(vcfg["layer_start"], vcfg["layer_end"] + 1))

    print(f"LoRA layers: {layer_list[0]}–{layer_list[-1]} ({len(layer_list)} layers)")

    lora_cfg = LoraConfig(
        task_type           = TaskType.CAUSAL_LM,
        r                   = args.lora_rank,
        lora_alpha          = args.lora_alpha,
        target_modules      = ["qkv_proj"],  # Phi-3.5 fused
        layers_to_transform = layer_list,
        lora_dropout        = 0.05,
        bias                = "none",
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.float()

    # Data
    perturber = PerturbationEngine()
    pairs     = build_pairs(perturber,
                            n_per_condition=args.n_per_condition,
                            clean_fraction=args.clean_fraction)

    pad_id    = tokenizer.pad_token_id
    print("Tokenizing...")
    tokenized = []
    for p_pair in tqdm(pairs, desc="Tokenize"):
        td = tokenize_pair(tokenizer, p_pair, args.max_seq_len)
        td["is_clean"]     = p_pair.is_clean
        td["perturbation"] = p_pair.perturbation
        tokenized.append(td)

    # Optimizer
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=args.lr, weight_decay=0.01)
    warmup    = max(1, int(args.n_steps * 0.05))
    sched     = SequentialLR(
        optimizer,
        [LinearLR(optimizer, 0.1, 1.0, warmup),
         CosineAnnealingLR(optimizer, args.n_steps - warmup, eta_min=args.lr * 0.1)],
        milestones=[warmup],
    )

    print(f"\nTraining — {args.n_steps} steps")
    logs = []
    step = 0
    optimizer.zero_grad()

    while step < args.n_steps:
        random.shuffle(tokenized)
        for batch_start in range(0, len(tokenized) - args.batch_size + 1,
                                 args.batch_size):
            if step >= args.n_steps:
                break
            samples = tokenized[batch_start:batch_start + args.batch_size]
            batch   = collate_batch(samples, pad_id=pad_id, device=device)

            model.train()
            out    = model(input_ids=batch.noisy_full_ids,
                           attention_mask=batch.noisy_attn_mask)
            logits = out.logits.float()
            loss   = accuracy_loss(logits, batch.noisy_full_ids, batch.answer_mask)
            (loss / args.grad_accum_steps).backward()

            step += 1
            if step % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                sched.step()
                optimizer.zero_grad()

            if step % 50 == 0:
                print(f"  Step {step:5d}/{args.n_steps} loss={loss.item():.4f}")
                logs.append({"step": step, "loss": round(loss.item(), 5)})

    final = os.path.join(vcfg["output_dir"], "lora_final")
    model.save_pretrained(final)
    json.dump(logs, open(os.path.join(vcfg["output_dir"], "training_logs.json"), "w"),
              indent=2)
    print(f"Adapter saved: {final}")

    results = evaluate(model, tokenizer, perturber, args.n_eval, device,
                       vcfg["output_dir"])

    # Save variant metadata
    meta = {
        "variant":      variant_key,
        "label":        vcfg["label"],
        "layer_start":  layer_list[0],
        "layer_end":    layer_list[-1],
        "n_lora_layers": len(layer_list),
        "lora_rank":    args.lora_rank,
        "lora_alpha":   args.lora_alpha,
        "n_steps":      args.n_steps,
        "loss":         "CE_only (no stability loss)",
        "avg_perturbed_delta": round(
            sum(r["delta"] for k, r in results.items() if k != "clean_baseline") /
            max(1, sum(1 for k in results if k != "clean_baseline")), 2
        ),
        "clean_delta": round(
            results.get("clean_baseline", {}).get("delta", float("nan")), 2
        ),
    }
    json.dump(meta, open(os.path.join(vcfg["output_dir"], "variant_meta.json"), "w"),
              indent=2)

    del model, base_model
    torch.cuda.empty_cache()
    return meta, results


# ─────────────────────────────────────────────────────────────────────────────
# Comparison summary
# ─────────────────────────────────────────────────────────────────────────────

def write_comparison():
    """Load both variant results and write comparison table."""
    comparison = {}
    for vkey, vcfg in VARIANT_CONFIGS.items():
        meta_path = os.path.join(vcfg["output_dir"], "variant_meta.json")
        eval_path = os.path.join(vcfg["output_dir"], "eval_results.json")
        if os.path.exists(meta_path) and os.path.exists(eval_path):
            comparison[vkey] = {
                "meta":    json.load(open(meta_path)),
                "results": json.load(open(eval_path)),
            }

    if not comparison:
        print("No variant results found yet.")
        return

    out_path = os.path.join(ROOT, "aug_baseline_comparison.json")
    json.dump(comparison, open(out_path, "w"), indent=2)
    print(f"\n=== Augmentation Baseline Comparison ===")
    print(f"{'Variant':<35} {'Avg Δ perturbed':>16} {'Clean Δ':>10}")
    print("-" * 65)
    for vkey, data in comparison.items():
        meta = data["meta"]
        print(f"{meta['label']:<35} {meta['avg_perturbed_delta']:>+15.2f}% "
              f"{meta['clean_delta']:>+9.2f}%")
    print(f"\nSaved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variant",        type=str, default="both",
                   choices=["window", "all", "both"],
                   help="window=L15-19 only, all=32 layers, both=run sequentially")
    p.add_argument("--lora_rank",      type=int, default=4)
    p.add_argument("--lora_alpha",     type=int, default=8)
    p.add_argument("--n_steps",        type=int, default=2000,
                   help="Training steps per variant")
    p.add_argument("--n_per_condition",type=int, default=300)
    p.add_argument("--clean_fraction", type=float, default=0.20)
    p.add_argument("--batch_size",     type=int, default=4)
    p.add_argument("--grad_accum_steps",type=int, default=4)
    p.add_argument("--lr",             type=float, default=2e-4)
    p.add_argument("--max_seq_len",    type=int, default=512)
    p.add_argument("--n_eval",         type=int, default=500)
    p.add_argument("--seed",           type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    variants = ["window", "all"] if args.variant == "both" else [args.variant]

    for v in variants:
        run_variant(v, args)

    write_comparison()
