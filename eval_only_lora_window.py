#!/usr/bin/env python3
"""
Evaluation-only script for LoRA adapters that have already been trained.
Loads saved adapters and runs evaluation on GSM8K.
"""

import os
import sys
import json
import random
import argparse
from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

ROOT = os.path.dirname(os.path.abspath(__file__))
for p in [ROOT, os.path.join(ROOT, "Phi3.5")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from train_lrd_stabilizer import (
    format_question_prompt,
    _extract_gsm8k_answer, _extract_gsm8k_truth, _gsm8k_correct,
)
from datasets import load_dataset

try:
    from perturbations import PerturbationEngine
except ImportError:
    sys.path.insert(0, os.path.join(ROOT, "Phi3.5"))
    from perturbations import PerturbationEngine

EVAL_CONDITIONS = [
    ("none",       0.00),
    ("typos",      0.05),
    ("ocr",        0.05),
    ("speech",     0.10),
    ("homophones", 0.20),
    ("whitespace", 0.10),
    ("case",       0.10),
]


def evaluate(model, tokenizer, perturber, eval_items, device, output_dir: str, eval_batch_size: int) -> dict:
    """Evaluate on GSM8K with/without adapter for all conditions."""
    print("\n" + "=" * 55 + "\nEVALUATION\n" + "=" * 55)
    model.eval()

    items = eval_items
    n_samples = len(items)

    gen_kw = dict(max_new_tokens=512, do_sample=False,
                  temperature=None, top_p=None, top_k=None,
                  pad_token_id=tokenizer.pad_token_id,
                  use_cache=True)

    def gen_batch(prompts: List[str], use_adapter: bool, batch_size: int = 8) -> List[str]:
        """Batched generation for faster evaluation."""
        out = []
        n_batches = (len(prompts) + batch_size - 1) // batch_size
        for i in range(0, len(prompts), batch_size):
            batch_idx = i // batch_size + 1
            if batch_idx % 10 == 0 or batch_idx == 1:
                print(f"  Generating batch {batch_idx}/{n_batches}...")
                sys.stdout.flush()
            batch = prompts[i:i + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=1024).to(device)
            if use_adapter:
                ids = model.generate(**enc, **gen_kw)
            else:
                with model.disable_adapter():
                    ids = model.generate(**enc, **gen_kw)
            new_ids = ids[:, enc["input_ids"].shape[1]:]
            out.extend(tokenizer.batch_decode(new_ids, skip_special_tokens=True))
        return out

    results  = {}
    truths   = [_extract_gsm8k_truth(it["answer"]) for it in items]

    for method, rate in EVAL_CONDITIONS:
        cond = f"{method}_{int(rate * 100)}pct" if rate > 0 else "clean_baseline"
        print(f"\n--- {cond} ---")
        sys.stdout.flush()

        prompts_p = [format_question_prompt(tokenizer, perturber.apply(it["question"], method, rate)
                                            if rate > 0 else it["question"])
                     for it in items]
        prompts_c = [format_question_prompt(tokenizer, it["question"]) for it in items]

        print("  Pass 1/3: No adapter on perturbed...")
        sys.stdout.flush()
        preds_no  = gen_batch(prompts_p, use_adapter=False, batch_size=eval_batch_size)
        print("  Pass 2/3: With adapter on perturbed...")
        sys.stdout.flush()
        preds_yes = gen_batch(prompts_p, use_adapter=True, batch_size=eval_batch_size)
        if method != "none":
            print("  Pass 3/3: No adapter on clean...")
            sys.stdout.flush()
            preds_cln = gen_batch(prompts_c, use_adapter=False, batch_size=eval_batch_size)
        else:
            preds_cln = preds_no

        acc_no  = sum(_gsm8k_correct(_extract_gsm8k_answer(p), t)
                      for p, t in zip(preds_no, truths)) / len(truths) * 100
        acc_yes = sum(_gsm8k_correct(_extract_gsm8k_answer(p), t)
                      for p, t in zip(preds_yes, truths)) / len(truths) * 100
        acc_cln = sum(_gsm8k_correct(_extract_gsm8k_answer(p), t)
                      for p, t in zip(preds_cln, truths)) / len(truths) * 100
        delta   = acc_yes - acc_no

        print(f"  No adapter:   {acc_no:.1f}%")
        print(f"  With adapter: {acc_yes:.1f}%  (Δ={delta:+.1f}%)")

        results[cond] = {
            "method":             method,
            "rate":               rate,
            "n_samples":          n_samples,
            "acc_no_adapter":     round(acc_no, 2),
            "acc_with_adapter":   round(acc_yes, 2),
            "acc_clean_baseline": round(acc_cln, 2),
            "delta":              round(delta, 2),
        }

    print("\n" + "=" * 55)
    print(f"{'Condition':<22} {'No Adp':>8} {'W/ Adp':>8} {'Delta':>7}")
    print("-" * 55)
    for cond, r in results.items():
        print(f"{cond:<22} {r['acc_no_adapter']:>7.1f}% {r['acc_with_adapter']:>7.1f}% {r['delta']:>+6.1f}%")

    path = os.path.join(output_dir, "eval_results.json")
    json.dump(results, open(path, "w"), indent=2)
    print(f"\nSaved: {path}")
    return results


def eval_saved_adapter(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print(f"\nLoading base model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Use bfloat16 for Gemma, float16 fallback
    if "gemma" in args.model.lower():
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
        attn_impl = "eager"
    else:
        dtype = torch.bfloat16
        attn_impl = "sdpa"

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map={"": 0},
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )

    # Load the saved LoRA adapter
    adapter_path = os.path.join(args.output_dir, "lora_final")
    print(f"Loading adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    # Load evaluation data
    print("Loading GSM8K test set for evaluation...")
    perturber = PerturbationEngine()
    eval_ds    = load_dataset("openai/gsm8k", "main", split="test")
    eval_items = list(eval_ds)
    random.seed(args.seed)
    random.shuffle(eval_items)
    eval_items = eval_items[:args.n_eval]
    print(f"  {len(eval_items)} evaluation samples ready.")

    # Run evaluation
    evaluate(model, tokenizer, perturber, eval_items, device, args.output_dir, args.eval_batch_size)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",          type=str,   required=True)
    p.add_argument("--output_dir",     type=str,   required=True,
                   help="Directory containing lora_final/ adapter")
    p.add_argument("--n_eval",         type=int,   default=500)
    p.add_argument("--eval_batch_size",type=int,   default=8)
    p.add_argument("--seed",           type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("=" * 55)
    print(f"Model:   {args.model}")
    print(f"Adapter: {args.output_dir}/lora_final")
    print(f"Eval:    {args.n_eval} samples, batch_size={args.eval_batch_size}")
    print("=" * 55)
    eval_saved_adapter(args)
