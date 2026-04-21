

import os
import sys
import json
import random
import argparse
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

Shared infrastructure from existing pipeline
ROOT = os.path.dirname(os.path.abspath(__file__))
for p in [ROOT, os.path.join(ROOT, "Phi3.5")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from train_lrd_stabilizer import (
    TrainingPair, TokenizedBatch, collate_batch, tokenize_pair,
    GSM8K_SYSTEM_PROMPT, format_question_prompt,
    _extract_gsm8k_answer, _extract_gsm8k_truth, _gsm8k_correct,
    accuracy_loss,
)

try:
    from perturbations import PerturbationEngine
except ImportError:
    sys.path.insert(0, os.path.join(ROOT, "Phi3.5"))
    from perturbations import PerturbationEngine

# Evaluation conditions (same as existing sweeps for comparability)

EVAL_CONDITIONS = [
    ("none",       0.00),
    ("typos",      0.05),
    ("ocr",        0.05),
    ("speech",     0.10),
    ("homophones", 0.20),
    ("whitespace", 0.10),
    ("case",       0.10),
]

TRAINING_POOL = [
    ("typos",      0.05),
    ("ocr",        0.05),
    ("ocr",        0.15),
    ("speech",     0.10),
    ("homophones", 0.30),
    ("whitespace", 0.10),
    ("case",       0.10),
]


# Dataset builder

def build_training_pairs(perturber: PerturbationEngine,
                         n_per_condition: int = 150,
                         clean_fraction: float = 0.20) -> List[TrainingPair]:
    """GSM8K training pairs with random perturbations (CE-only baseline)."""
    print("Loading GSM8K training split...")
    ds    = load_dataset("openai/gsm8k", "main", split="train")
    items = list(ds)
    random.shuffle(items)

    pairs: List[TrainingPair] = []

    # build pairs from each perturbation type in the training pool
    for method, rate in TRAINING_POOL:
        for i in range(n_per_condition):
            item    = items[i % len(items)]  # cycle through dataset
            clean_q = item["question"]
            clean_a = item["answer"]
            noisy_q = perturber.apply(clean_q, method, rate)
            pairs.append(TrainingPair(
                clean_question=clean_q,
                noisy_question=noisy_q,
                clean_answer=clean_a,
                perturbation=method,
                rate=rate,
                is_clean=False,
            ))

    # Clean pairs (data augmentation baseline)
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
    print(f"  {len(pairs)} training pairs ({n_clean} clean).")
    return pairs


# Evaluation

@torch.no_grad()
def evaluate(model, tokenizer, perturber, eval_items, device, output_dir: str) -> dict:
    """Evaluate on GSM8K with/without adapter for all conditions.

    Args:
        eval_items: Pre-loaded and shuffled GSM8K test items (already limited to n_samples)
    """
    print("\n" + "=" * 55 + "\nPOST-TRAINING EVALUATION\n" + "=" * 55)
    model.eval()

    items = eval_items

    gen_kw = dict(max_new_tokens=100, do_sample=False,
                  temperature=None, top_p=None, top_k=None,
                  pad_token_id=tokenizer.pad_token_id,
                  use_cache=True)

    def gen_batch(prompts: List[str], use_adapter: bool, batch_size: int = 8) -> List[str]:
        """Batched generation for faster evaluation (use batch_size=8 for speed)."""
        out = []
        n_batches = (len(prompts) + batch_size - 1) // batch_size
        for i in range(0, len(prompts), batch_size):
            batch_idx = i // batch_size + 1
            if batch_idx % 10 == 0 or batch_idx == 1:  # less verbose logging
                print(f"  Generating batch {batch_idx}/{n_batches}...")
                sys.stdout.flush()
            batch = prompts[i:i + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=1024).to(device)
            if use_adapter:
                ids = model.generate(**enc, **gen_kw)
            else:
                with model.disable_adapter():  # baseline: frozen base model
                    ids = model.generate(**enc, **gen_kw)
            new_ids = ids[:, enc["input_ids"].shape[1]:]  # strip prompt
            out.extend(tokenizer.batch_decode(new_ids, skip_special_tokens=True))
        return out

    results  = {}
    truths   = [_extract_gsm8k_truth(it["answer"]) for it in items]
    n_samples = len(items)

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
        preds_no  = gen_batch(prompts_p, use_adapter=False, batch_size=args.eval_batch_size)
        print("  Pass 2/3: With adapter on perturbed...")
        sys.stdout.flush()
        preds_yes = gen_batch(prompts_p, use_adapter=True, batch_size=args.eval_batch_size)
        if method != "none":
            print("  Pass 3/3: No adapter on clean...")
            sys.stdout.flush()
            preds_cln = gen_batch(prompts_c, use_adapter=False, batch_size=args.eval_batch_size)
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


# Training

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load model
    print(f"\nLoading: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Use bfloat16 when available for better numerical stability
    # Gemma2 needs eager attention, Qwen can use sdpa
    if "gemma" in args.model.lower():
        # Try bfloat16 for training stability, fall back to float16 if not supported
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
        attn_impl = "eager"  # gemma fails with sdpa
    else:
        dtype = torch.bfloat16
        attn_impl = "sdpa"  # faster on Qwen/TinyLlama

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map={"": 0},
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )

    # LoRA config
    layer_list = list(range(args.layer_start, args.layer_end + 1))
    print(f"LoRA layers: {layer_list}  modules: {args.target_modules}")

    lora_cfg = LoraConfig(
        task_type         = TaskType.CAUSAL_LM,
        r                 = args.lora_rank,
        lora_alpha        = args.lora_alpha,
        target_modules    = args.target_modules,
        layers_to_transform = layer_list,  # restrict to specified window
        lora_dropout      = 0.05,
        bias              = "none",
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    # Convert LoRA parameters to float32 for training stability
    # Base model stays in float16/bfloat16, but LoRA adapters train in float32
    for name, param in model.named_parameters():
        if param.requires_grad and param.dtype != torch.float32:
            param.data = param.data.to(torch.float32)
    # print(f"Converted {sum(1 for p in model.parameters() if p.requires_grad)} LoRA params to fp32")

    # Data
    perturber = PerturbationEngine()
    pairs     = build_training_pairs(perturber,
                                     n_per_condition=args.n_per_condition,
                                     clean_fraction=args.clean_fraction)

    # Load evaluation data (once, to avoid re-downloading)
    print("Loading GSM8K test set for evaluation...")
    eval_ds    = load_dataset("openai/gsm8k", "main", split="test")
    eval_items = list(eval_ds)
    random.shuffle(eval_items)
    eval_items = eval_items[:args.n_eval]
    print(f"  {len(eval_items)} evaluation samples ready.")

    pad_id   = tokenizer.pad_token_id
    print("Tokenizing pairs...")
    tokenized = []
    for p in tqdm(pairs, desc="Tokenize"):
        td = tokenize_pair(tokenizer, p, args.max_seq_len)
        td["is_clean"]     = p.is_clean
        td["perturbation"] = p.perturbation
        tokenized.append(td)

    # Optimizer 
    trainable = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = AdamW(trainable, lr=args.lr, weight_decay=0.01)

    warmup    = max(1, int(args.n_steps * 0.05))
    sched_w   = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                         total_iters=warmup)
    sched_c   = CosineAnnealingLR(optimizer, T_max=args.n_steps - warmup,
                                  eta_min=args.lr * 0.1)
    scheduler = SequentialLR(optimizer, [sched_w, sched_c], milestones=[warmup])

    # Training loop 
    print(f"\nTraining — {args.n_steps} steps, batch={args.batch_size}, "
          f"grad_accum={args.grad_accum_steps}")

    # Use automatic mixed precision with gradient scaling for float16 models
    use_amp = (dtype == torch.float16)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    print(f"Using AMP: {use_amp}")

    logs = []
    step = 0
    optimizer.zero_grad()

    while step < args.n_steps:
        random.shuffle(tokenized)  # shuffle training pairs each epoch
        for batch_start in range(0, len(tokenized) - args.batch_size + 1,
                                 args.batch_size):
            if step >= args.n_steps:
                break

            samples = tokenized[batch_start:batch_start + args.batch_size]
            batch   = collate_batch(samples, pad_id=pad_id, device=device)

            model.train()

            # Use autocast for mixed precision if enabled
            if use_amp:
                with torch.cuda.amp.autocast():
                    out = model(input_ids=batch.noisy_full_ids,
                               attention_mask=batch.noisy_attn_mask)
                    logits = out.logits.float()
                    loss = accuracy_loss(logits, batch.noisy_full_ids, batch.answer_mask)
            else:
                out = model(input_ids=batch.noisy_full_ids,
                           attention_mask=batch.noisy_attn_mask)
                logits = out.logits.float()
                loss = accuracy_loss(logits, batch.noisy_full_ids, batch.answer_mask)

            # Check for NaN and skip this step if found (happened occasionally with Gemma2)
            if torch.isnan(loss):
                if step % 100 == 0:
                    print(f"WARNING: NaN loss at step {step}, skipping", file=sys.stderr)
                step += 1
                continue

            # Backward with gradient scaling if using AMP
            if use_amp:
                scaler.scale(loss / args.grad_accum_steps).backward()
            else:
                (loss / args.grad_accum_steps).backward()

            step += 1
            if step % args.grad_accum_steps == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()

            if step % 5 == 0:  # more frequent updates for debugging
                print(f"Step {step:5d}/{args.n_steps} | loss={loss.item():.4f}")
                sys.stdout.flush()
            if step % 25 == 0:  # log every 25 steps
                logs.append({"step": step, "loss": round(loss.item(), 5)})
                # print(f"  [DEBUG] lr={scheduler.get_last_lr()[0]:.6f}")

    # Save
    final = os.path.join(args.output_dir, "lora_final")
    model.save_pretrained(final)
    print(f"\nAdapter saved: {final}")
    json.dump(logs, open(os.path.join(args.output_dir, "training_logs.json"), "w"), indent=2)

    # Evaluate
    evaluate(model, tokenizer, perturber, eval_items, device, args.output_dir)

def parse_args():
    p = argparse.ArgumentParser(description="Generic LoRA window sweep")
    p.add_argument("--model",          type=str,   required=True)
    p.add_argument("--layer_start",    type=int,   required=True)
    p.add_argument("--layer_end",      type=int,   required=True)
    p.add_argument("--n_steps",        type=int,   default=300)
    p.add_argument("--lora_rank",      type=int,   default=4)
    p.add_argument("--lora_alpha",     type=int,   default=8)
    p.add_argument("--target_modules", type=str,   nargs="+",
                   default=["q_proj", "v_proj"])
    p.add_argument("--output_dir",     type=str,   required=True)
    p.add_argument("--n_eval",         type=int,   default=500)
    p.add_argument("--n_per_condition",type=int,   default=150)
    p.add_argument("--clean_fraction", type=float, default=0.20)
    p.add_argument("--batch_size",     type=int,   default=4)
    p.add_argument("--eval_batch_size",type=int,   default=8)
    p.add_argument("--grad_accum_steps",type=int,  default=4)
    p.add_argument("--lr",             type=float, default=2e-4)
    p.add_argument("--max_seq_len",    type=int,   default=512)
    p.add_argument("--seed",           type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("=" * 55)
    print(f"Model:   {args.model}")
    print(f"Layers:  {args.layer_start}–{args.layer_end}")
    print(f"Steps:   {args.n_steps}")
    print(f"LoRA:    r={args.lora_rank}, α={args.lora_alpha}, {args.target_modules}")
    print(f"Output:  {args.output_dir}")
    print("=" * 55)
    train(args)
