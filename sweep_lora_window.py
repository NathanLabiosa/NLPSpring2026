"""
sweep_lora_window.py — Generic LoRA Window Sweep for regime-taxonomy validation.

Trains a LoRA adapter restricted to a contiguous window of layers, evaluates
on n_eval GSM8K test examples with all perturbation conditions, and saves
eval_results.json in the same format consumed by expB_three_map.py.

Loss: cross-entropy on perturbed examples + optional stability loss.
  - Default (lambda_stab=0): CE-only data augmentation baseline
  - With lambda_stab>0 and stab_layer: adds representation stability constraint

Usage:
    python sweep_lora_window.py \\
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\
        --layer_start 0 --layer_end 3 \\
        --n_steps 300 \\
        --output_dir ./stabilizer_weights/tinyllama_sweep_L00_03 \\
        --target_modules q_proj v_proj

    python sweep_lora_window.py \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --layer_start 15 --layer_end 19 \\
        --n_steps 1000 \\
        --output_dir ./stabilizer_weights/qwen_sweep_L15_19 \\
        --target_modules q_proj v_proj

Output:
    {output_dir}/eval_results.json   — per-condition acc_no_adapter, acc_with_adapter, delta
    {output_dir}/lora_final/         — saved LoRA adapter
    {output_dir}/training_logs.json  — per-step loss
"""

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

# ── Shared infrastructure from existing pipeline ──────────────────────────────
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


def stability_loss(h_noisy: torch.Tensor, h_clean: torch.Tensor,
                   mask: torch.Tensor) -> torch.Tensor:
    """Cosine-similarity stability loss between noisy and clean hidden states.

    Args:
        h_noisy: Hidden states from noisy input [B, T, D]
        h_clean: Hidden states from clean input [B, T, D]
        mask: Token mask for answer positions [B, T]

    Returns:
        Scalar loss: 1 - mean cosine similarity over masked positions
    """
    # Normalize along hidden dimension
    h_noisy_norm = F.normalize(h_noisy, dim=-1)
    h_clean_norm = F.normalize(h_clean, dim=-1)

    # Cosine similarity per position
    cos_sim = (h_noisy_norm * h_clean_norm).sum(dim=-1)  # [B, T]

    # Mask and average
    mask_float = mask.float()
    if mask_float.sum() > 0:
        loss = 1.0 - (cos_sim * mask_float).sum() / mask_float.sum()
    else:
        loss = torch.tensor(0.0, device=h_noisy.device)

    return loss

try:
    from perturbations import PerturbationEngine
except ImportError:
    sys.path.insert(0, os.path.join(ROOT, "Phi3.5"))
    from perturbations import PerturbationEngine

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation conditions (same as existing sweeps for comparability)
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Dataset builder
# ─────────────────────────────────────────────────────────────────────────────

def build_training_pairs(perturber: PerturbationEngine,
                         n_per_condition: int = 150,
                         clean_fraction: float = 0.20,
                         heldout_perturbation: str = None) -> List[TrainingPair]:
    """GSM8K training pairs with random perturbations (CE-only baseline)."""
    print("Loading GSM8K training split...")
    ds    = load_dataset("openai/gsm8k", "main", split="train")
    items = list(ds)
    random.shuffle(items)

    # Filter training pool if holding out a perturbation type
    training_pool = TRAINING_POOL
    if heldout_perturbation:
        training_pool = [(m, r) for m, r in TRAINING_POOL if m != heldout_perturbation]
        print(f"  Holding out '{heldout_perturbation}' from training pool")
        print(f"  Training pool: {training_pool}")

    pairs: List[TrainingPair] = []

    for method, rate in training_pool:
        for i in range(n_per_condition):
            item    = items[i % len(items)]
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

    # Clean pairs
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


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ── Load model ─────────────────────────────────────────────────────────
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

    # ── LoRA config ──────────────────────────────────────────────────────
    layer_list = list(range(args.layer_start, args.layer_end + 1))
    print(f"LoRA layers: {layer_list}  modules: {args.target_modules}")

    lora_cfg = LoraConfig(
        task_type         = TaskType.CAUSAL_LM,
        r                 = args.lora_rank,
        lora_alpha        = args.lora_alpha,
        target_modules    = args.target_modules,
        layers_to_transform = layer_list,
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

    # ── Data ──────────────────────────────────────────────────────────────
    perturber = PerturbationEngine()
    pairs     = build_training_pairs(perturber,
                                     n_per_condition=args.n_per_condition,
                                     clean_fraction=args.clean_fraction,
                                     heldout_perturbation=args.heldout_perturbation)

    # ── Load evaluation data (once, to avoid re-downloading) ──────────────
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

    # ── Optimizer ─────────────────────────────────────────────────────────
    trainable = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = AdamW(trainable, lr=args.lr, weight_decay=0.01)

    warmup    = max(1, int(args.n_steps * 0.05))
    sched_w   = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                         total_iters=warmup)
    sched_c   = CosineAnnealingLR(optimizer, T_max=args.n_steps - warmup,
                                  eta_min=args.lr * 0.1)
    scheduler = SequentialLR(optimizer, [sched_w, sched_c], milestones=[warmup])

    # ── Training loop ──────────────────────────────────────────────────────
    use_stab = args.lambda_stab > 0 and args.stab_layer is not None
    if use_stab:
        print(f"\nStability loss enabled: λ_stab={args.lambda_stab}, stab_layer={args.stab_layer}")
    else:
        print(f"\nCE-only mode (λ_stab={args.lambda_stab})")
    print(f"Training — {args.n_steps} steps, batch={args.batch_size}, "
          f"grad_accum={args.grad_accum_steps}")

    # Use automatic mixed precision with gradient scaling for float16 models
    use_amp = (dtype == torch.float16)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    print(f"Using AMP: {use_amp}")

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

            # Use autocast for mixed precision if enabled
            if use_amp:
                with torch.cuda.amp.autocast():
                    out = model(input_ids=batch.noisy_full_ids,
                               attention_mask=batch.noisy_attn_mask,
                               output_hidden_states=use_stab)
                    logits = out.logits.float()
                    ce_loss = accuracy_loss(logits, batch.noisy_full_ids, batch.answer_mask)
                    if use_stab:
                        with model.disable_adapter():
                            out_clean = model(input_ids=batch.noisy_full_ids,
                                            attention_mask=batch.noisy_attn_mask,
                                            output_hidden_states=True)
                        stab_loss = stability_loss(out.hidden_states[args.stab_layer],
                                                   out_clean.hidden_states[args.stab_layer],
                                                   batch.answer_mask)
                        loss = ce_loss + args.lambda_stab * stab_loss
                    else:
                        loss = ce_loss
            else:
                out = model(input_ids=batch.noisy_full_ids,
                           attention_mask=batch.noisy_attn_mask,
                           output_hidden_states=use_stab)
                logits = out.logits.float()
                ce_loss = accuracy_loss(logits, batch.noisy_full_ids, batch.answer_mask)
                if use_stab:
                    with model.disable_adapter():
                        out_clean = model(input_ids=batch.noisy_full_ids,
                                        attention_mask=batch.noisy_attn_mask,
                                        output_hidden_states=True)
                    stab_loss = stability_loss(out.hidden_states[args.stab_layer],
                                               out_clean.hidden_states[args.stab_layer],
                                               batch.answer_mask)
                    loss = ce_loss + args.lambda_stab * stab_loss
                else:
                    loss = ce_loss

            # Check for NaN and skip this step if found
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

            if step % 5 == 0:  # More frequent updates for debugging
                print(f"Step {step:5d}/{args.n_steps} | loss={loss.item():.4f}")
                sys.stdout.flush()
            if step % 25 == 0:
                logs.append({"step": step, "loss": round(loss.item(), 5)})

    # ── Save ───────────────────────────────────────────────────────────────
    final = os.path.join(args.output_dir, "lora_final")
    model.save_pretrained(final)
    print(f"\nAdapter saved: {final}")
    json.dump(logs, open(os.path.join(args.output_dir, "training_logs.json"), "w"), indent=2)

    # ── Evaluate ──────────────────────────────────────────────────────────
    evaluate(model, tokenizer, perturber, eval_items, device, args.output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

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
    p.add_argument("--heldout_perturbation", type=str, default=None,
                   help="Perturbation type to exclude from training (for held-out evaluation)")
    p.add_argument("--lambda_stab",    type=float, default=0.0,
                   help="Stability loss weight (0.0 = CE-only baseline)")
    p.add_argument("--stab_layer",     type=int,   default=None,
                   help="Layer index for stability loss (required if lambda_stab > 0)")
    return p.parse_args()


if __name__ == "__main__":
    # ---------------------------------------------------------------------
    # Harness guard. The inline eval at the end of training uses
    # max_new_tokens=100, which truncates chat-template CoT before the
    # `#### NUMBER` extraction line and depresses no_adapter baselines.
    # All accuracy reporting must go through eval_fixed_harness.py instead.
    # See experiment_results.md §5 and HARNESS.md for the audit.
    # Set ALLOW_SWEEP_EVAL=1 only for non-accuracy uses (e.g. training-only).
    # ---------------------------------------------------------------------
    assert os.environ.get("ALLOW_SWEEP_EVAL") == "1", (
        "sweep_lora_window.py is deprecated for accuracy reporting. "
        "Use eval_fixed_harness.py instead (max_new_tokens=512). "
        "See experiment_results.md §5 and HARNESS.md. "
        "To run anyway (training-only, no accuracy claims), set "
        "ALLOW_SWEEP_EVAL=1 in the environment."
    )
    args = parse_args()
    print("=" * 55)
    print(f"Model:   {args.model}")
    print(f"Layers:  {args.layer_start}–{args.layer_end}")
    print(f"Steps:   {args.n_steps}")
    print(f"LoRA:    r={args.lora_rank}, α={args.lora_alpha}, {args.target_modules}")
    print(f"Stab:    λ={args.lambda_stab}, layer={args.stab_layer}")
    print(f"Output:  {args.output_dir}")
    print("=" * 55)
    train(args)
