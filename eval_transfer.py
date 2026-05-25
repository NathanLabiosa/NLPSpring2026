"""
eval_transfer.py — Zero-shot transfer eval for v17 LoRA on MMLU and BBH.

Tests whether the perturbation-robustness LoRA (trained only on GSM8K) generalizes
to other task formats. Any improvement here is genuine cross-task generalization.

Perturbation conditions match the main GSM8K eval.

Usage:
    python eval_transfer.py \
        --checkpoint_dir ./stabilizer_weights/lora_v17/lora_final \
        --n_samples 200 \
        --output ./stabilizer_weights/lora_v17/transfer_results.json
"""

import os, sys, json, random, argparse
from typing import List, Tuple, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_lrd_stabilizer import (
    EVAL_CONDITIONS, GSM8K_SYSTEM_PROMPT,
    _import_perturbation_engine,
)

PerturbationEngine = _import_perturbation_engine()

# ─────────────────────────────────────────────────────────────────────────────
# MMLU helpers
# ─────────────────────────────────────────────────────────────────────────────

MMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "college_mathematics",
    "elementary_mathematics", "high_school_mathematics", "high_school_physics",
    "high_school_chemistry", "logical_fallacies", "formal_logic",
]
CHOICE_LABELS = ["A", "B", "C", "D"]


def format_mmlu_prompt(tokenizer, question: str, choices: List[str]) -> str:
    choice_str = "\n".join(f"{l}. {c}" for l, c in zip(CHOICE_LABELS, choices))
    user_msg = f"{question}\n{choice_str}\nAnswer with a single letter (A, B, C, or D):"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": user_msg},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def extract_mmlu_answer(text: str) -> Optional[str]:
    for ch in text.strip():
        if ch in CHOICE_LABELS:
            return ch
    for label in CHOICE_LABELS:
        if label + "." in text or label + ")" in text or f" {label} " in text:
            return label
    return None


def load_mmlu(n_per_subject: int = 20) -> List[dict]:
    from datasets import load_dataset
    items = []
    for subject in MMLU_SUBJECTS:
        try:
            ds = load_dataset("cais/mmlu", subject, split="test")
            subset = list(ds)
            random.shuffle(subset)
            for row in subset[:n_per_subject]:
                items.append({
                    "question": row["question"],
                    "choices":  row["choices"],
                    "answer":   CHOICE_LABELS[row["answer"]],
                    "subject":  subject,
                    "task":     "mmlu",
                })
        except Exception as e:
            print(f"  Warning: could not load mmlu/{subject}: {e}")
    return items


# ─────────────────────────────────────────────────────────────────────────────
# BBH helpers
# ─────────────────────────────────────────────────────────────────────────────

BBH_MC_TASKS = [
    "date_understanding", "disambiguation_qa", "geometric_shapes",
    "logical_deduction_five_objects", "movie_recommendation",
    "penguins_in_a_table", "reasoning_about_colored_objects",
    "ruin_names", "salient_translation_error_detection",
    "snarks", "temporal_sequences", "tracking_shuffled_objects_five_objects",
]


def format_bbh_prompt(tokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant. "
                                      "Answer the following question concisely."},
        {"role": "user",   "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def extract_bbh_answer(text: str) -> Optional[str]:
    """Extract answer from BBH output — handles (A)/(B)/... and free-form."""
    import re
    text = text.strip()
    # Multiple choice: (A), (B), ...
    m = re.search(r'\(([A-Z])\)', text)
    if m:
        return m.group(1)
    # Bare letter at start
    m = re.match(r'^([A-Z])[.\s]', text)
    if m:
        return m.group(1)
    return text.split("\n")[0].strip()


def load_bbh(n_per_task: int = 20) -> List[dict]:
    from datasets import load_dataset
    items = []
    for task in BBH_MC_TASKS:
        try:
            ds = load_dataset("lukaemon/bbh", task, split="test")
            subset = list(ds)
            random.shuffle(subset)
            for row in subset[:n_per_task]:
                items.append({
                    "question": row["input"],
                    "answer":   row["target"],
                    "task":     f"bbh_{task}",
                })
        except Exception as e:
            print(f"  Warning: could not load bbh/{task}: {e}")
    return items


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def run_generation(model, tokenizer, prompts: List[str], use_adapter: bool,
                   device, gen_kwargs: dict) -> List[str]:
    outputs = []
    for i in range(0, len(prompts), 4):
        batch = prompts[i : i + 4]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=768).to(device)
        if use_adapter:
            out_ids = model.generate(**enc, **gen_kwargs)
        else:
            with model.disable_adapter():
                out_ids = model.generate(**enc, **gen_kwargs)
        new_ids = out_ids[:, enc["input_ids"].shape[1]:]
        outputs.extend(tokenizer.batch_decode(new_ids, skip_special_tokens=True))
    return outputs


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_transfer(model, tokenizer, perturber, items: List[dict],
                      benchmark: str, device, n_samples: int) -> dict:
    random.shuffle(items)
    items = items[:n_samples]

    gen_kwargs = dict(max_new_tokens=64, do_sample=False,
                      pad_token_id=tokenizer.pad_token_id)

    if benchmark == "mmlu":
        def make_prompt(item, question):
            return format_mmlu_prompt(tokenizer, question, item["choices"])
        def is_correct(pred, item):
            ans = extract_mmlu_answer(pred)
            return ans == item["answer"]
    else:  # bbh
        def make_prompt(item, question):
            return format_bbh_prompt(tokenizer, question)
        def is_correct(pred, item):
            # Apply same extractor to both sides: BBH targets are often "(A)" or
            # "(A) some text" while predictions say "(A) ..." — comparing the
            # extracted letter avoids the full-string mismatch that caused 0%.
            return extract_bbh_answer(pred).strip().lower() == extract_bbh_answer(item["answer"]).strip().lower()

    results = {}

    for method, rate in EVAL_CONDITIONS:
        cond_name = f"{method}_{int(rate * 100)}pct" if rate > 0 else "clean_baseline"
        print(f"  [{benchmark}] {cond_name} (n={len(items)})")

        prompts, truths_items = [], []
        for item in items:
            q_noisy = perturber.apply(item["question"], method, rate) if rate > 0 else item["question"]
            prompts.append(make_prompt(item, q_noisy))
            truths_items.append(item)

        preds_no   = run_generation(model, tokenizer, prompts, False, device, gen_kwargs)
        preds_with = run_generation(model, tokenizer, prompts, True,  device, gen_kwargs)

        acc_no   = sum(is_correct(p, t) for p, t in zip(preds_no,   truths_items)) / len(items) * 100
        acc_with = sum(is_correct(p, t) for p, t in zip(preds_with, truths_items)) / len(items) * 100
        delta    = acc_with - acc_no

        print(f"    No adapter: {acc_no:.1f}%  With adapter: {acc_with:.1f}%  Δ={delta:+.1f}%")

        results[cond_name] = {
            "acc_no_adapter":   round(acc_no, 2),
            "acc_with_adapter": round(acc_with, 2),
            "delta":            round(delta, 2),
            "n":                len(items),
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str,
                        default="./stabilizer_weights/lora_v17/lora_final")
    parser.add_argument("--base_model", type=str,
                        default="microsoft/Phi-3.5-mini-instruct")
    parser.add_argument("--n_samples",  type=int, default=200,
                        help="Examples per benchmark (total, pooled across subjects/tasks)")
    parser.add_argument("--n_per_subject", type=int, default=20,
                        help="MMLU examples per subject")
    parser.add_argument("--n_per_task",    type=int, default=20,
                        help="BBH examples per task")
    parser.add_argument("--output", type=str,
                        default="./stabilizer_weights/lora_v17/transfer_results.json")
    parser.add_argument("--seed",  type=int, default=42)
    parser.add_argument("--skip_mmlu", action="store_true")
    parser.add_argument("--skip_bbh",  action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)

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
    all_results = {}

    if not args.skip_mmlu:
        print("\n" + "=" * 60)
        print("MMLU EVALUATION")
        print("=" * 60)
        mmlu_items = load_mmlu(n_per_subject=args.n_per_subject)
        print(f"Loaded {len(mmlu_items)} MMLU items from {len(MMLU_SUBJECTS)} subjects")
        all_results["mmlu"] = evaluate_transfer(
            model, tokenizer, perturber, mmlu_items, "mmlu", device, args.n_samples
        )

    if not args.skip_bbh:
        print("\n" + "=" * 60)
        print("BBH EVALUATION")
        print("=" * 60)
        bbh_items = load_bbh(n_per_task=args.n_per_task)
        print(f"Loaded {len(bbh_items)} BBH items from {len(BBH_MC_TASKS)} tasks")
        all_results["bbh"] = evaluate_transfer(
            model, tokenizer, perturber, bbh_items, "bbh", device, args.n_samples
        )

    # Summary table
    print("\n" + "=" * 65)
    print(f"{'Benchmark':<10} {'Condition':<22} {'No Adp':>7} {'W/ Adp':>7} {'Δ':>6}")
    print("-" * 65)
    for bench, conds in all_results.items():
        for cond, r in conds.items():
            print(f"{bench:<10} {cond:<22} {r['acc_no_adapter']:>6.1f}% "
                  f"{r['acc_with_adapter']:>6.1f}% {r['delta']:>+5.1f}%")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {args.output}")


if __name__ == "__main__":
    main()
