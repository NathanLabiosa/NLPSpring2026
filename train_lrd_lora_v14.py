"""
train_lrd_lora_v14.py — LoRA fine-tuning for perturbation robustness (v14).

Architecture: LoRA adapters on layers 0–8 (q_proj, v_proj), rank=16, alpha=32.
No external correction modules. The model's own attention layers learn robustness.

Loss:
  L_acc  — task CE on all examples (perturbed + clean), answer tokens only
  L_stab — MSE between LoRA and frozen hidden states at layer 9, clean examples
            only, at answer positions (prevents clean-input regression)

See run14.txt for full design rationale.
"""

import os, sys, json, random, argparse
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# ── Shared infrastructure from existing training script ─────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_lrd_stabilizer import (
    TrainingPair, TokenizedBatch, collate_batch, tokenize_pair,
    EVAL_CONDITIONS, KNOWN_BASELINES,
    DEFAULT_PERTURBATION_POOL, DATASET_BUILDERS,
    GSM8K_SYSTEM_PROMPT, format_question_prompt,
    accuracy_loss,
    _extract_gsm8k_answer, _extract_gsm8k_truth, _gsm8k_correct,
    _import_perturbation_engine,
    PERTURBATION_LOSS_WEIGHTS, PERTURBATION_LOSS_DEFAULT_WEIGHT,
)

PerturbationEngine = _import_perturbation_engine()

# Taxonomy-informed per-perturbation loss weights (used with --taxonomy_weights).
# "Uniform" types (case, whitespace) produce large LRD but don't predict task
# failure; downweighting them focuses LoRA capacity on failure-predictive types.
TAXONOMY_LOSS_WEIGHTS: dict = {
    "whitespace": 0.3,
    "case":       0.3,
}


def weighted_accuracy_loss(
    logits: torch.Tensor,         # [B, T, V]
    target_ids: torch.Tensor,     # [B, T]
    answer_mask: torch.Tensor,    # [B, T] bool
    sample_weights: torch.Tensor, # [B]
) -> torch.Tensor:
    """Per-example CE averaged with per-example sample_weights."""
    shift_logits = logits[:, :-1, :].contiguous()   # [B, T-1, V]
    shift_labels = target_ids[:, 1:].contiguous()    # [B, T-1]
    shift_mask   = answer_mask[:, 1:].bool()         # [B, T-1]

    if not shift_mask.any():
        return logits.sum() * 0.0

    B, T, V = shift_logits.shape
    per_token = F.cross_entropy(
        shift_logits.reshape(-1, V),
        shift_labels.reshape(-1),
        reduction='none',
    ).reshape(B, T)  # [B, T-1]

    n_ans  = shift_mask.float().sum(dim=1).clamp(min=1.0)  # [B]
    per_ex = (per_token * shift_mask.float()).sum(dim=1) / n_ans  # [B]

    w = sample_weights
    return (w * per_ex).sum() / (w.sum() + 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_lora(
    model,
    tokenizer,
    perturber,
    n_samples: int,
    device: torch.device,
    output_dir: str,
    eval_conditions=None,
) -> dict:
    """Evaluate LoRA-adapted model vs. base model on GSM8K test conditions."""
    from datasets import load_dataset

    print("\n" + "=" * 60)
    print("POST-TRAINING EVALUATION")
    print("=" * 60)

    model.eval()

    test_ds    = load_dataset("openai/gsm8k", "main", split="test")
    test_items = list(test_ds)
    random.shuffle(test_items)
    test_items = test_items[:n_samples]

    gen_kwargs = dict(
        max_new_tokens=300, do_sample=False, pad_token_id=tokenizer.pad_token_id
    )

    def run_generation(prompts: List[str], use_adapter: bool) -> List[str]:
        outputs = []
        for i in range(0, len(prompts), 4):
            batch_prompts = prompts[i : i + 4]
            enc = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)
            if use_adapter:
                out_ids = model.generate(**enc, **gen_kwargs)
            else:
                with model.disable_adapter():
                    out_ids = model.generate(**enc, **gen_kwargs)
            new_ids = out_ids[:, enc["input_ids"].shape[1]:]
            outputs.extend(tokenizer.batch_decode(new_ids, skip_special_tokens=True))
        return outputs

    results = {}

    _eval_conds = eval_conditions if eval_conditions is not None else EVAL_CONDITIONS
    for method, rate in _eval_conds:
        cond_name = f"{method}_{int(rate * 100)}pct" if rate > 0 else "clean_baseline"
        print(f"\n--- Condition: {cond_name} (n={n_samples}) ---")

        prompts_perturbed, prompts_clean, truths = [], [], []
        for item in test_items:
            q_clean = item["question"]
            q_noisy = perturber.apply(q_clean, method, rate) if rate > 0 else q_clean
            truths.append(_extract_gsm8k_truth(item["answer"]))
            prompts_clean.append(format_question_prompt(tokenizer, q_clean))
            prompts_perturbed.append(format_question_prompt(tokenizer, q_noisy))

        preds_no_adapter = run_generation(prompts_perturbed, use_adapter=False)
        acc_no_adapter = (
            sum(_gsm8k_correct(_extract_gsm8k_answer(p), t)
                for p, t in zip(preds_no_adapter, truths))
            / len(truths) * 100
        )

        preds_with_adapter = run_generation(prompts_perturbed, use_adapter=True)
        acc_with_adapter = (
            sum(_gsm8k_correct(_extract_gsm8k_answer(p), t)
                for p, t in zip(preds_with_adapter, truths))
            / len(truths) * 100
        )

        if method == "none":
            acc_clean_baseline = acc_no_adapter
        else:
            preds_clean = run_generation(prompts_clean, use_adapter=False)
            acc_clean_baseline = (
                sum(_gsm8k_correct(_extract_gsm8k_answer(p), t)
                    for p, t in zip(preds_clean, truths))
                / len(truths) * 100
            )

        delta    = acc_with_adapter - acc_no_adapter
        known    = KNOWN_BASELINES.get(method)
        known_str = f"  (known LRD baseline: {known:.1f}%)" if known else ""

        print(f"  No adapter (perturbed):   {acc_no_adapter:.1f}%{known_str}")
        print(f"  With adapter (perturbed): {acc_with_adapter:.1f}%  (Δ = {delta:+.1f}%)")
        if method != "none":
            print(f"  Clean baseline:           {acc_clean_baseline:.1f}%")

        results[cond_name] = {
            "method":             method,
            "rate":               rate,
            "n_samples":          n_samples,
            "acc_no_adapter":     round(acc_no_adapter, 2),
            "acc_with_adapter":   round(acc_with_adapter, 2),
            "acc_clean_baseline": round(acc_clean_baseline, 2),
            "delta":              round(delta, 2),
            "known_baseline":     known,
        }

    print("\n" + "=" * 60)
    print(f"{'Condition':<22} {'No Adp':>8} {'W/ Adp':>8} {'Delta':>7} {'Clean':>8}")
    print("-" * 60)
    for cond_name, r in results.items():
        print(
            f"{cond_name:<22} "
            f"{r['acc_no_adapter']:>7.1f}% "
            f"{r['acc_with_adapter']:>7.1f}% "
            f"{r['delta']:>+6.1f}% "
            f"{r['acc_clean_baseline']:>7.1f}%"
        )

    eval_path = os.path.join(output_dir, "eval_results.json")
    with open(eval_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nEval results saved: {eval_path}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Seed for reproducibility
    if hasattr(args, "seed") and args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # ── 1. Load base model ───────────────────────────────────────────────────
    print(f"\nLoading base model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Auto-detect dtype and attention implementation based on model
    if "gemma" in args.model.lower():
        # Gemma2 needs bfloat16 or float16 + AMP, prefer bfloat16
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
        attn_impl = "eager"  # Gemma2 requires eager
    elif "qwen" in args.model.lower():
        dtype = torch.bfloat16
        attn_impl = "sdpa"  # Qwen can use SDPA
    else:
        dtype = torch.float16
        attn_impl = "eager"

    print(f"Using dtype={dtype}, attn_implementation={attn_impl}")
    import sys
    sys.stdout.flush()

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map={"": 0},
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )
    print(f"Model loaded successfully!")
    sys.stdout.flush()

    # ── 2. Wrap with LoRA ────────────────────────────────────────────────────
    print("Configuring LoRA...")
    sys.stdout.flush()
    if args.lora_layer_indices is not None:
        lora_layer_list = sorted(args.lora_layer_indices)
        print(f"Using explicit LoRA layer indices: {lora_layer_list}")
    else:
        lora_layer_list = list(range(args.lora_layers_start,
                                     args.lora_layers_start + args.lora_layers))
    print(f"LoRA will be applied to layers: {lora_layer_list}")
    sys.stdout.flush()

    lora_config = LoraConfig(
        task_type        = TaskType.CAUSAL_LM,
        r                = args.lora_rank,
        lora_alpha       = args.lora_alpha,
        target_modules   = args.target_modules,
        layers_to_transform = lora_layer_list,
        lora_dropout     = args.lora_dropout,
        bias             = "none",
    )
    print("Wrapping model with PEFT...")
    sys.stdout.flush()
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    sys.stdout.flush()

    print("Converting LoRA parameters to float32...")
    sys.stdout.flush()
    # Cast LoRA parameters to float32 for stable training
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.float()
    print("LoRA setup complete!")
    sys.stdout.flush()

    # ── 3. Build dataset ─────────────────────────────────────────────────────
    print("Building training dataset...")
    sys.stdout.flush()
    UNIFORM_PERTURBATION_POOL = [
        ("typos",      0.05),
        ("ocr",        0.05),
        ("speech",     0.10),
        ("homophones", 0.20),
        ("whitespace", 0.10),
        ("case",       0.10),
    ]

    perturber = PerturbationEngine()
    if args.uniform_pool:
        perturbation_pool = UNIFORM_PERTURBATION_POOL
        print(f"Using uniform perturbation pool: {perturbation_pool}")
    elif args.perturbation_types is not None:
        allowed = set(args.perturbation_types)
        perturbation_pool = [(m, r) for m, r in DEFAULT_PERTURBATION_POOL if m in allowed]
        if not perturbation_pool:
            raise ValueError(f"--perturbation_types {args.perturbation_types!r} matched nothing")
        print(f"Perturbation pool filtered to: {perturbation_pool}")
    else:
        perturbation_pool = DEFAULT_PERTURBATION_POOL

    if args.uniform_rate is not None:
        # Override all rates to the same value (matched-rate ablation)
        seen = set()
        deduped = []
        for m, _ in perturbation_pool:
            if m not in seen:
                deduped.append((m, args.uniform_rate))
                seen.add(m)
        perturbation_pool = deduped
        print(f"Uniform rate override ({args.uniform_rate}): {perturbation_pool}")

    # Build matched eval conditions if uniform_rate is set
    if args.uniform_rate is not None:
        matched_eval_conditions = [("none", 0.0)] + [
            (m, args.uniform_rate) for m, _ in EVAL_CONDITIONS if m != "none"
        ]
        # Deduplicate by method name
        seen_eval = set()
        deduped_eval = [("none", 0.0)]
        for m, r in matched_eval_conditions:
            if m != "none" and m not in seen_eval:
                deduped_eval.append((m, r))
                seen_eval.add(m)
        matched_eval_conditions = deduped_eval
        print(f"Matched eval conditions: {matched_eval_conditions}")
    else:
        matched_eval_conditions = None

    if args.clean_only:
        # Build clean-only dataset matching the total budget of a perturbed run
        from datasets import load_dataset as _load_ds
        _ds = _load_ds("openai/gsm8k", "main", split="train")
        _items = list(_ds)
        random.shuffle(_items)
        n_total = int(args.n_per_condition * len(perturbation_pool) * (1 + args.clean_fraction))
        pairs = []
        for i in range(n_total):
            item = _items[i % len(_items)]
            q, a = item["question"], item["answer"]
            pairs.append(TrainingPair(
                clean_question=q, noisy_question=q, clean_answer=a,
                perturbation="none", rate=0.0, is_clean=True,
            ))
        print(f"Clean-only mode: {n_total} clean pairs (matching perturbed budget)")
    else:
        print(f"Using dataset builder: {args.dataset}")
        sys.stdout.flush()
        build_fn = DATASET_BUILDERS[args.dataset]
        print(f"Calling dataset builder function...")
        sys.stdout.flush()
        pairs    = build_fn(
            perturber         = perturber,
            perturbation_pool = perturbation_pool,
            n_per_condition   = args.n_per_condition,
            clean_fraction    = args.clean_fraction,
        )
    print(f"Training pairs built: {len(pairs)}")
    sys.stdout.flush()

    # ── 4. Optimizer & scheduler ─────────────────────────────────────────────
    print("Setting up optimizer and scheduler...")
    sys.stdout.flush()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer  = AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    total_steps = min(args.max_steps, args.epochs * (len(pairs) // args.batch_size + 1))
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    scheduler_warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    scheduler_cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=args.lr * 0.1)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine],
                             milestones=[warmup_steps])
    print("Optimizer and scheduler ready!")
    sys.stdout.flush()

    # ── 5. Training loop ─────────────────────────────────────────────────────
    print(f"\nStarting training — max_steps={total_steps}, batch_size={args.batch_size}, "
          f"grad_accum={args.grad_accum_steps}")
    sys.stdout.flush()

    # Measure baseline clean accuracy before any training (used for early stopping)
    print(f"\nMeasuring pre-training clean baseline (n={args.clean_eval_n} samples)...")
    sys.stdout.flush()
    baseline_clean_acc = _quick_clean_eval(model, tokenizer, perturber, device,
                                           n=args.clean_eval_n, return_acc=True)
    print(f"Pre-training clean baseline: {baseline_clean_acc:.1f}%  "
          f"(early stop threshold: {baseline_clean_acc - args.early_stop_delta:.1f}%)")
    sys.stdout.flush()

    global_step  = 0
    all_logs     = []
    early_stopped = False
    optimizer.zero_grad()
    pad_id = tokenizer.pad_token_id

    # Tokenize all pairs upfront
    print(f"Tokenizing {len(pairs)} training pairs...")
    sys.stdout.flush()
    tokenized = []
    for i, p in enumerate(pairs):
        if i % 100 == 0:
            print(f"  Tokenized {i}/{len(pairs)}...")
            sys.stdout.flush()
        td = tokenize_pair(tokenizer, p, args.max_seq_len)
        td["is_clean"]    = p.is_clean      # needed for clean_mask / L_stab routing
        td["perturbation"] = p.perturbation  # needed for per-condition logging
        tokenized.append(td)
    print(f"Tokenization done: {len(tokenized)} pairs ready.")
    sys.stdout.flush()

    print("Starting training loop...")
    sys.stdout.flush()
    epoch = 0
    while global_step < total_steps:
        epoch += 1
        random.shuffle(tokenized)
        if epoch == 1:
            print(f"Epoch {epoch}: Processing {len(tokenized)} samples in batches of {args.batch_size}")
            sys.stdout.flush()

        for batch_start in range(0, len(tokenized) - args.batch_size + 1, args.batch_size):
            if global_step >= total_steps:
                break

            samples = tokenized[batch_start : batch_start + args.batch_size]
            batch   = collate_batch(samples, pad_id=pad_id, device=device)

            model.train()

            is_clean_t = torch.tensor(batch.is_clean, dtype=torch.bool, device=device)
            clean_mask = is_clean_t
            noisy_mask = ~is_clean_t  # noqa: F841 (available for future use)

            # ── Forward pass (LoRA enabled) on noisy inputs ──────────────────
            # Use a hook to capture layer stab_layer hidden states instead of
            # output_hidden_states=True — avoids materialising all 33 tensors.
            _stab_layer_mod = model.base_model.model.model.layers[args.stab_layer - 1]
            _h_lora_captured = {}
            def _lora_hook(m, inp, out):
                _h_lora_captured['h'] = out[0] if isinstance(out, tuple) else out
            _lora_handle = _stab_layer_mod.register_forward_hook(_lora_hook)
            try:
                out = model(
                    input_ids      = batch.noisy_full_ids,
                    attention_mask = batch.noisy_attn_mask,
                    output_hidden_states = False,
                )
            finally:
                _lora_handle.remove()
            logits = out.logits.float()

            # L_acc — CE on all examples at answer positions
            if args.taxonomy_weights:
                tw = torch.tensor(
                    [TAXONOMY_LOSS_WEIGHTS.get(p, 1.0) for p in batch.perturbations],
                    dtype=torch.float32, device=device,
                )
                L_acc = weighted_accuracy_loss(logits, batch.noisy_full_ids, batch.answer_mask, tw)
            else:
                L_acc = accuracy_loss(logits, batch.noisy_full_ids, batch.answer_mask)

            # L_stab — MSE at layer stab_layer between adapted and frozen, clean only
            L_stab = torch.tensor(0.0, device=device)
            if args.lambda_stab > 0 and clean_mask.any():
                h_lora_clean   = _h_lora_captured['h'][clean_mask].float()
                ans_mask_clean = batch.answer_mask[clean_mask].float()

                # Frozen pass: early-exit after stab_layer via exception hook.
                # Stops before layers stab_layer+1..N and lm_head — ~half the FLOPS
                # and avoids materialising the large lm_head output tensor.
                class _EarlyExit(Exception):
                    pass
                _h_frozen_captured = {}
                def _frozen_hook(m, inp, out):
                    _h_frozen_captured['h'] = (
                        out[0] if isinstance(out, tuple) else out
                    ).detach().float()
                    raise _EarlyExit()
                _frozen_handle = _stab_layer_mod.register_forward_hook(_frozen_hook)
                try:
                    with model.disable_adapter():
                        with torch.no_grad():
                            model(
                                input_ids      = batch.noisy_full_ids[clean_mask],
                                attention_mask = batch.noisy_attn_mask[clean_mask],
                            )
                except _EarlyExit:
                    pass
                finally:
                    _frozen_handle.remove()

                h_frozen_clean = _h_frozen_captured['h']
                if ans_mask_clean.any():
                    if args.stab_cosine:
                        # Cosine distance matching the LRD definition:
                        # mean-pool over answer positions first, then cosine.
                        # LRD(L) = 1 - cos(h_bar_clean, h_bar_noisy)
                        # This gives one scalar per example in the clean batch.
                        mask_f = ans_mask_clean.unsqueeze(-1)  # [B_clean, T, 1]
                        h_lora_pooled = (h_lora_clean * mask_f).sum(dim=1) / (mask_f.sum(dim=1) + 1e-8)
                        h_frozen_pooled = (h_frozen_clean * mask_f).sum(dim=1) / (mask_f.sum(dim=1) + 1e-8)
                        cos_sim = F.cosine_similarity(h_lora_pooled, h_frozen_pooled, dim=-1)
                        L_stab = (1.0 - cos_sim).mean()
                    else:
                        # MSE (original) — per-token, masked to answer positions
                        per_tok = ((h_lora_clean - h_frozen_clean) ** 2).mean(dim=-1)
                        L_stab = (per_tok * ans_mask_clean).sum() / (ans_mask_clean.sum() + 1e-8)

            # Adaptive L_stab scaling: dynamically set lambda_stab so that
            # L_stab contributes exactly stab_fraction of the total loss.
            if args.stab_fraction is not None and L_stab.item() > 1e-8 and L_acc.item() > 1e-8:
                # Want: ls*S / (la*A + ls*S) = f  =>  ls = f*la*A / ((1-f)*S)
                f = args.stab_fraction
                effective_lambda_stab = (f * args.lambda_acc * L_acc.item()) / ((1.0 - f) * L_stab.item())
                loss = args.lambda_acc * L_acc + effective_lambda_stab * L_stab
            else:
                effective_lambda_stab = args.lambda_stab
                loss = args.lambda_acc * L_acc + args.lambda_stab * L_stab
            loss = loss / args.grad_accum_steps
            loss.backward()

            global_step += 1
            is_update_step = (global_step % args.grad_accum_steps == 0)

            gnorm = 0.0
            if is_update_step:
                gnorm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0).item()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # ── Logging ──────────────────────────────────────────────────────
            if global_step % args.log_every == 0:
                batch_type = "CLEAN" if all(batch.is_clean) else ("MIXED" if any(batch.is_clean) else "NOISY")
                stab_frac_actual = (effective_lambda_stab * L_stab.item()) / (loss.item() * args.grad_accum_steps + 1e-8) if L_stab.item() > 1e-8 else 0.0
                extra = f"  λs_eff={effective_lambda_stab:.1f}  frac={stab_frac_actual:.2f}" if args.stab_fraction is not None else ""
                print(
                    f"Step {global_step:5d} | {batch_type:5s} | "
                    f"loss={loss.item() * args.grad_accum_steps:.4f}  "
                    f"acc={L_acc.item():.4f}  "
                    f"stab={L_stab.item():.4f}  "
                    f"gnorm={gnorm:.3f}"
                    f"{extra}"
                )
                log_entry = {
                    "step":       global_step,
                    "loss":       round((loss.item() * args.grad_accum_steps), 6),
                    "L_acc":      round(L_acc.item(), 6),
                    "L_stab":     round(L_stab.item(), 6),
                    "gnorm":      round(gnorm, 4),
                    "batch_type": batch_type,
                }
                if args.stab_fraction is not None:
                    log_entry["lambda_stab_eff"] = round(effective_lambda_stab, 4)
                    log_entry["stab_frac"] = round(stab_frac_actual, 4)
                all_logs.append(log_entry)

            # ── Checkpoint + mid-training eval ───────────────────────────────
            if global_step % args.save_every == 0:
                ckpt_path = os.path.join(args.output_dir, f"lora_step{global_step:06d}")
                model.save_pretrained(ckpt_path)
                print(f"  Checkpoint: {ckpt_path}")

                if args.eval_every > 0 and global_step % args.eval_every == 0:
                    evaluate_lora(
                        model      = model,
                        tokenizer  = tokenizer,
                        perturber  = perturber,
                        n_samples  = args.eval_n_samples,
                        device     = device,
                        output_dir = ckpt_path,
                        eval_conditions = matched_eval_conditions,
                    )
                    model.train()

            # ── Mid-training clean accuracy check + early stopping ───────────
            if global_step % args.clean_eval_every == 0:
                probe_acc = _quick_clean_eval(model, tokenizer, perturber, device,
                                              n=args.clean_eval_n, return_acc=True)
                if args.early_stop_delta > 0 and probe_acc < baseline_clean_acc - args.early_stop_delta:
                    print(f"  [Early stop] Clean acc {probe_acc:.1f}% dropped >"
                          f"{args.early_stop_delta:.1f}% below baseline "
                          f"({baseline_clean_acc:.1f}%). Stopping at step {global_step}.")
                    early_stopped = True
                    break
        if early_stopped:
            break

    # ── 6. Save final adapter ────────────────────────────────────────────────
    final_path = os.path.join(args.output_dir, "lora_final")
    model.save_pretrained(final_path)
    print(f"\nTraining complete. Saved: {final_path}")

    log_path = os.path.join(args.output_dir, "training_logs.json")
    with open(log_path, "w") as f:
        json.dump(all_logs, f, indent=2)
    print(f"Logs: {log_path}")

    # ── 7. Post-training evaluation ──────────────────────────────────────────
    if args.eval_after_training:
        evaluate_lora(
            model      = model,
            tokenizer  = tokenizer,
            perturber  = perturber,
            n_samples  = args.eval_n_samples,
            device     = device,
            output_dir = args.output_dir,
            eval_conditions = matched_eval_conditions,
        )


@torch.no_grad()
def _quick_clean_eval(model, tokenizer, perturber, device, n: int = 500,
                      return_acc: bool = False):
    """Quick clean-accuracy probe during training (early regression detection)."""
    import sys
    from datasets import load_dataset
    print(f"[EVAL] Loading GSM8K dataset...")
    sys.stdout.flush()
    model.eval()
    ds    = load_dataset("openai/gsm8k", "main", split="test")
    items = list(ds)
    random.shuffle(items)
    items = items[:n]
    print(f"[EVAL] Dataset loaded, running inference on {n} examples...")
    sys.stdout.flush()

    prompts = [format_question_prompt(tokenizer, it["question"]) for it in items]
    truths  = [_extract_gsm8k_truth(it["answer"]) for it in items]

    correct = 0
    eval_batch_size = 2  # Small batch for large model + long generation
    for i in range(0, len(prompts), eval_batch_size):
        if i % 10 == 0:
            print(f"[EVAL] Progress: {i}/{n}...")
            sys.stdout.flush()
        enc = tokenizer(
            prompts[i : i + eval_batch_size],
            return_tensors="pt", padding=True, truncation=True, max_length=512,
        ).to(device)
        with model.disable_adapter():
            out_ids = model.generate(
                **enc, max_new_tokens=300, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_ids = out_ids[:, enc["input_ids"].shape[1]:]
        preds   = tokenizer.batch_decode(new_ids, skip_special_tokens=True)
        for p, t in zip(preds, truths[i : i + eval_batch_size]):
            correct += _gsm8k_correct(_extract_gsm8k_answer(p), t)
    print(f"[EVAL] Inference complete!")
    sys.stdout.flush()

    acc = correct / n * 100
    print(f"  [Clean probe] n={n}: {acc:.1f}%")
    model.train()
    if return_acc:
        return acc


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="v14: LoRA fine-tuning for perturbation robustness")

    # Model
    parser.add_argument("--model", type=str, default="microsoft/Phi-3.5-mini-instruct")

    # LoRA
    parser.add_argument("--lora_rank",    type=int,   default=16)
    parser.add_argument("--lora_alpha",   type=int,   default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_layers",  type=int,   default=9,
                        help="Number of consecutive layers to apply LoRA to")
    parser.add_argument("--lora_layers_start", type=int, default=0,
                        help="First layer index for LoRA (layers start..start+n-1)")
    parser.add_argument("--lora_layer_indices", type=int, nargs="+", default=None,
                        help="Explicit layer indices for LoRA (overrides --lora_layers/start). "
                             "E.g. --lora_layer_indices 6 11 17 22 28")
    parser.add_argument("--target_modules", type=str, nargs="+",
                        default=["qkv_proj", "o_proj"],
                        help="LoRA target module names (model-specific). "
                             "Phi-3.5: qkv_proj o_proj")

    # Loss
    parser.add_argument("--lambda_acc",   type=float, default=0.7)
    parser.add_argument("--lambda_stab",  type=float, default=0.3)
    parser.add_argument("--stab_layer",   type=int,   default=9,
                        help="Hidden state index to use for L_stab (hidden_states[i] "
                             "= output after transformer block i-1; 9 = after block 8)")
    parser.add_argument("--stab_cosine", action="store_true", default=False,
                        help="Use cosine distance instead of MSE for L_stab. "
                             "Matches the LRD diagnostic metric.")
    parser.add_argument("--stab_fraction", type=float, default=None,
                        help="Target fraction of total loss from L_stab (e.g. 0.4 for 40%%). "
                             "Dynamically rescales lambda_stab each step so that "
                             "lambda_stab*L_stab / (lambda_acc*L_acc + lambda_stab*L_stab) ≈ stab_fraction. "
                             "Overrides --lambda_stab when set.")

    # Data
    parser.add_argument("--dataset",           type=str, default="mixed")
    parser.add_argument("--n_per_condition",   type=int, default=500)
    parser.add_argument("--clean_fraction",    type=float, default=0.25)
    parser.add_argument("--perturbation_types", type=str, nargs="+", default=None)
    parser.add_argument("--clean_only", action="store_true", default=False,
                        help="Train on clean GSM8K only (no perturbations). "
                             "Creates n_per_condition × len(pool) clean pairs to match "
                             "the total data budget of a perturbed run.")
    parser.add_argument("--uniform_pool", action="store_true", default=False,
                        help="Use one condition per perturbation type at equal weight "
                             "(ablation: removes OCR double-weighting from default pool)")
    parser.add_argument("--taxonomy_weights", action="store_true", default=False,
                        help="Enable per-perturbation loss weighting: whitespace and case "
                             "at 0.3× CE weight, others at 1.0×. Tests whether LRD "
                             "taxonomy-informed weighting improves directional-type robustness.")
    parser.add_argument("--uniform_rate", type=float, default=None,
                        help="Override all perturbation rates to this value (e.g. 0.10). "
                             "Used for matched-rate ablation experiments.")

    # Training
    parser.add_argument("--epochs",          type=int,   default=20)
    parser.add_argument("--max_steps",       type=int,   default=25000)
    parser.add_argument("--batch_size",      type=int,   default=4)
    parser.add_argument("--grad_accum_steps",type=int,   default=4)
    parser.add_argument("--lr",              type=float, default=2e-4)
    parser.add_argument("--warmup_ratio",    type=float, default=0.05)
    parser.add_argument("--max_seq_len",     type=int,   default=512)

    # Output
    parser.add_argument("--output_dir",  type=str, default="./stabilizer_weights/lora_v14")
    parser.add_argument("--log_every",   type=int, default=25)
    parser.add_argument("--save_every",  type=int, default=500)
    parser.add_argument("--eval_every",  type=int, default=500,
                        help="Steps between full evals at each checkpoint. "
                             "Must be a multiple of save_every. 0 = disable.")
    parser.add_argument("--clean_eval_every", type=int, default=500,
                        help="Steps between quick clean-accuracy probes.")
    parser.add_argument("--clean_eval_n", type=int, default=500,
                        help="Number of GSM8K test examples for each clean probe.")
    parser.add_argument("--early_stop_delta", type=float, default=0.0,
                        help="Stop training if clean probe drops more than this many "
                             "percentage points below the pre-training baseline. "
                             "0 = disabled.")

    # Eval
    parser.add_argument("--eval_after_training", action="store_true", default=True)
    parser.add_argument("--no_eval_after_training", dest="eval_after_training",
                        action="store_false")
    parser.add_argument("--eval_n_samples", type=int, default=200)

    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility (data shuffle, weight init).")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
