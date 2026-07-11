"""
eval_fixed_harness_mmlu.py — Fixed MMLU evaluation harness for the LoRA window sweep.

Evaluates a LoRA checkpoint (or bare base model) on a fixed 500-item stratified
MMLU test subset under 7 perturbation conditions. Mirrors eval_fixed_harness.py but
for MMLU: answers are single letters (A-D), max_new_tokens=32.

The first run creates results/mmlu_eval_item_ids.json with the fixed subset indices.
All subsequent runs load from that file to guarantee identical evaluation items.

Usage:
    python src/eval/eval_fixed_harness_mmlu.py \\
        --base_model microsoft/Phi-3.5-mini-instruct \\
        --checkpoint /data/nathanlabiosa/nlp2026_weights/phase2_mmlu_phi35_L10-14_seed42/lora_final \\
        --output_file results/fixed_harness/v2/mmlu/phi35_L10-14_seed42.json \\
        --item_ids_file results/mmlu_eval_item_ids.json
"""

import os
import sys
import json
import re
import random
import argparse
from typing import List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset

# Compatibility shims for transformers API changes
try:
    import compat  # noqa: F401
except ImportError:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in [
    os.path.join(ROOT, "evaluation"),
    os.path.join(ROOT, "src", "lib"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from train_lrd_stabilizer import MMLU_SYSTEM_PROMPT

try:
    from perturbations import PerturbationEngine
except ImportError:
    raise ImportError(
        "Cannot import PerturbationEngine. "
        "Add evaluation/ to PYTHONPATH: export PYTHONPATH=<repo>/evaluation:$PYTHONPATH"
    )

HARNESS_VERSION = "v2"

CONDITIONS: List[Tuple[str, float]] = [
    ("none",       0.0),
    ("typos",      0.05),
    ("ocr",        0.05),
    ("speech",     0.10),
    ("homophones", 0.20),
    ("whitespace", 0.10),
    ("case",       0.10),
]

ANSWER_LETTERS = ["A", "B", "C", "D"]


def _format_mmlu_prompt(item: dict) -> str:
    choices_str = "\n".join(f"{ANSWER_LETTERS[i]}. {c}" for i, c in enumerate(item["choices"]))
    return f"{item['question']}\n{choices_str}"


def _format_mmlu_prompt_perturbed(item: dict, perturber: PerturbationEngine,
                                   method: str, rate: float) -> str:
    """Perturb the question stem only; keep choice labels and text clean."""
    noisy_q = perturber.apply(item["question"], method, rate) if rate > 0 else item["question"]
    choices_str = "\n".join(f"{ANSWER_LETTERS[i]}. {c}" for i, c in enumerate(item["choices"]))
    return f"{noisy_q}\n{choices_str}"


def _apply_chat_template(tokenizer, question_text: str) -> str:
    """Apply the model's chat template with MMLU system prompt."""
    if getattr(tokenizer, "chat_template", None) is not None:
        messages = [
            {"role": "system", "content": MMLU_SYSTEM_PROMPT},
            {"role": "user", "content": question_text},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    # Fallback for base models without chat template
    return (
        f"{MMLU_SYSTEM_PROMPT}\n\nQuestion:\n{question_text}\n\nAnswer:"
    )


def _extract_mmlu_answer(generated: str) -> str:
    """Extract the first A/B/C/D letter from generated text."""
    text = generated.strip()
    # Match a standalone letter (word boundary)
    m = re.search(r'\b([ABCD])\b', text)
    if m:
        return m.group(1)
    # Fallback: first A/B/C/D character anywhere
    for ch in text.upper():
        if ch in "ABCD":
            return ch
    return ""


def _mmlu_correct(pred: str, truth: str) -> bool:
    return pred.strip().upper() == truth.strip().upper()


def _load_or_create_eval_subset(
    item_ids_file: str,
    n_items: int = 500,
    seed: int = 42,
) -> Tuple[List[dict], List[int]]:
    """Load MMLU test items. Creates a fixed stratified subset on first call."""
    ds = load_dataset("cais/mmlu", "all", split="test")
    all_items = list(ds)

    if os.path.exists(item_ids_file):
        with open(item_ids_file) as f:
            meta = json.load(f)
        indices = meta["indices"]
        print(f"Loaded fixed MMLU eval subset ({len(indices)} items) from {item_ids_file}")
    else:
        # Build stratified subset
        rng = random.Random(seed)
        by_subject = {}
        for idx, item in enumerate(all_items):
            subj = item.get("subject", "unknown")
            by_subject.setdefault(subj, []).append(idx)

        subjects = sorted(by_subject.keys())
        per_subject = max(1, n_items // len(subjects))
        indices = []
        for subj in subjects:
            pool = by_subject[subj]
            rng.shuffle(pool)
            indices.extend(pool[:per_subject])
        # Top up to exactly n_items
        remaining = [i for i in range(len(all_items)) if i not in set(indices)]
        rng.shuffle(remaining)
        indices.extend(remaining[: n_items - len(indices)])
        indices = sorted(indices[:n_items])

        os.makedirs(os.path.dirname(os.path.abspath(item_ids_file)), exist_ok=True)
        with open(item_ids_file, "w") as f:
            json.dump({"n_items": len(indices), "seed": seed, "indices": indices}, f, indent=2)
        print(f"Created fixed MMLU eval subset ({len(indices)} items) → {item_ids_file}")

    items = [all_items[i] for i in indices]
    return items, indices


@torch.no_grad()
def evaluate_conditions(
    model,
    tokenizer,
    perturber: PerturbationEngine,
    eval_items: List[dict],
    conditions: List[Tuple[str, float]],
    device: torch.device,
    batch_size: int = 16,
    max_new_tokens: int = 32,
    has_adapter: bool = True,
) -> dict:
    model.eval()
    truths = [ANSWER_LETTERS[item["answer"]] for item in eval_items]

    gen_kw = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None, top_p=None, top_k=None,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )

    def gen_batch(prompts: List[str], use_adapter: bool) -> List[str]:
        out = []
        n_batches = (len(prompts) + batch_size - 1) // batch_size
        for i in range(0, len(prompts), batch_size):
            b_idx = i // batch_size + 1
            if b_idx % 5 == 0 or b_idx == 1:
                print(f"    batch {b_idx}/{n_batches}")
                sys.stdout.flush()
            batch = prompts[i : i + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True,
                            truncation=True, max_length=512).to(device)
            if use_adapter or not has_adapter:
                ids = model.generate(**enc, **gen_kw)
            else:
                with model.disable_adapter():
                    ids = model.generate(**enc, **gen_kw)
            new_ids = ids[:, enc["input_ids"].shape[1]:]
            out.extend(tokenizer.batch_decode(new_ids, skip_special_tokens=True))
        return out

    results = {}
    clean_prompts = [
        _apply_chat_template(tokenizer, _format_mmlu_prompt(it)) for it in eval_items
    ]

    for method, rate in conditions:
        cond = "clean_baseline" if method == "none" else f"{method}_{int(rate * 100)}pct"
        print(f"\n--- {cond} ---")
        sys.stdout.flush()

        if method == "none":
            perturbed_prompts = clean_prompts
        else:
            perturbed_prompts = [
                _apply_chat_template(
                    tokenizer,
                    _format_mmlu_prompt_perturbed(it, perturber, method, rate)
                )
                for it in eval_items
            ]

        print("  [1/3] no adapter on perturbed")
        preds_no = gen_batch(perturbed_prompts, use_adapter=False)
        if has_adapter:
            print("  [2/3] with adapter on perturbed")
            preds_yes = gen_batch(perturbed_prompts, use_adapter=True)
        else:
            preds_yes = preds_no
        if method == "none":
            preds_cln = preds_no
        else:
            print("  [3/3] no adapter on clean")
            preds_cln = gen_batch(clean_prompts, use_adapter=False)

        acc_no = sum(_mmlu_correct(_extract_mmlu_answer(p), t)
                     for p, t in zip(preds_no, truths)) / len(truths) * 100
        acc_yes = sum(_mmlu_correct(_extract_mmlu_answer(p), t)
                      for p, t in zip(preds_yes, truths)) / len(truths) * 100
        acc_cln = sum(_mmlu_correct(_extract_mmlu_answer(p), t)
                      for p, t in zip(preds_cln, truths)) / len(truths) * 100
        delta = acc_yes - acc_no

        print(f"  no adapter:   {acc_no:.1f}%")
        if has_adapter:
            print(f"  with adapter: {acc_yes:.1f}%  (Δ={delta:+.1f}%)")

        results[cond] = {
            "method": method,
            "rate": rate,
            "n_samples": len(eval_items),
            "acc_no_adapter": round(acc_no, 2),
            "acc_with_adapter": round(acc_yes, 2) if has_adapter else None,
            "acc_clean_baseline": round(acc_cln, 2),
            "delta": round(delta, 2) if has_adapter else None,
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="LoRA checkpoint dir. Omit for no-adapter baseline audit.")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--item_ids_file", type=str,
                        default="results/mmlu_eval_item_ids.json")
    parser.add_argument("--n_items", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attn_impl", type=str, default="sdpa")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Base model: {args.base_model}")
    print(f"Checkpoint: {args.checkpoint}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
        attn_implementation=args.attn_impl,
    )

    has_adapter = args.checkpoint is not None
    if has_adapter:
        print("Loading LoRA adapter...")
        model = PeftModel.from_pretrained(base_model, args.checkpoint)
    else:
        print("No checkpoint — running no-adapter baseline.")
        model = base_model

    perturber = PerturbationEngine()

    # Resolve item_ids_file relative to repo root if not absolute
    ids_file = args.item_ids_file
    if not os.path.isabs(ids_file):
        ids_file = os.path.join(ROOT, ids_file)

    eval_items, indices = _load_or_create_eval_subset(ids_file, args.n_items, args.seed)
    print(f"\nEvaluating {len(CONDITIONS)} conditions on {len(eval_items)} MMLU items")

    results = evaluate_conditions(
        model, tokenizer, perturber,
        eval_items, CONDITIONS, device,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        has_adapter=has_adapter,
    )

    perturbed_deltas = [r["delta"] for c, r in results.items()
                        if c != "clean_baseline" and r["delta"] is not None]
    mean_perturbed_delta = (sum(perturbed_deltas) / len(perturbed_deltas)
                             if perturbed_deltas else None)
    clean_baseline = results.get("clean_baseline", {}).get("acc_no_adapter")

    output = {
        "harness_version": HARNESS_VERSION,
        "task": "mmlu",
        "base_model": args.base_model,
        "checkpoint": args.checkpoint,
        "seed": args.seed,
        "n_samples": len(eval_items),
        "max_new_tokens": args.max_new_tokens,
        "item_ids_file": ids_file,
        "clean_baseline": clean_baseline,
        "mean_perturbed_delta": (round(mean_perturbed_delta, 4)
                                  if mean_perturbed_delta is not None else None),
        "results": results,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {args.output_file}")

    print("\n" + "=" * 60)
    print(f"{'Condition':<22} {'No Adp':>8} {'W/ Adp':>8} {'Delta':>7}")
    print("-" * 60)
    for cond, r in results.items():
        adp_str = f"{r['acc_with_adapter']:>7.1f}%" if r["acc_with_adapter"] is not None else f"{'—':>8}"
        delta_str = f"{r['delta']:>+6.1f}%" if r["delta"] is not None else f"{'—':>7}"
        print(f"{cond:<22} {r['acc_no_adapter']:>7.1f}% {adp_str} {delta_str}")
    if mean_perturbed_delta is not None:
        print("-" * 60)
        print(f"{'mean perturbed Δ':<22} {'':>8} {'':>8} {mean_perturbed_delta:>+6.1f}%")


if __name__ == "__main__":
    main()
