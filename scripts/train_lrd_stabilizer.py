
from __future__ import annotations

import os
import re
import sys
import json
import random
import argparse
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import stabilizer system  
from stabilizer_system import MultiLayerStabilizerSystem, StabilizerConfig

# Import PerturbationEngine (search common locations)
def _import_perturbation_engine():
    """Try to import PerturbationEngine from common project locations."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    search_paths = [
        script_dir,
        os.path.join(script_dir, "Phi3.5"),
        os.path.join(script_dir, "shared"),
    ]
    for p in search_paths:
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        from perturbations import PerturbationEngine
        return PerturbationEngine
    except ImportError:
        raise ImportError(
            "Cannot find perturbations.py. Run from a directory that contains it, "
            "or add it to PYTHONPATH."
        )


PerturbationEngine = _import_perturbation_engine()



# Data Types

@dataclass
class TrainingPair:
    """A single training example: clean question, noisy question, clean answer."""
    clean_question: str
    noisy_question: str
    clean_answer:   str
    perturbation:   str     # e.g. "typos"
    rate:           float   # e.g. 0.05
    is_clean:       bool = False   # True when noisy_question == clean_question
    system_prompt:  str  = ""      # empty → tokenize_pair uses GSM8K_SYSTEM_PROMPT default


@dataclass
class TokenizedBatch:
    """Tensors for one training batch."""
    clean_input_ids:    torch.Tensor   # [B, T_clean]  for h_clean capture
    clean_attn_mask:    torch.Tensor   # [B, T_clean]
    noisy_full_ids:     torch.Tensor   # [B, T_noisy_full]  noisy q + clean answer
    noisy_attn_mask:    torch.Tensor   # [B, T_noisy_full]
    answer_mask:        torch.Tensor   # [B, T_noisy_full] True at answer positions
    noisy_prompt_lens:  torch.Tensor   # [B] int — number of prompt tokens (no answer)
    is_clean:           List[bool]
    perturbations:      List[str]      # perturbation type per item (e.g. "typos", "none")



# Perturbation Pool


# Per-condition L_stab / L_dir loss weight.
# Whitespace perturbation causes large token-count changes that inflate cosine
# loss relative to its actual semantic impact on model behaviour.  Down-weighting
# it prevents it from dominating training signal at the cost of modest coverage.
PERTURBATION_LOSS_WEIGHTS: dict = {
    "whitespace": 0.3,  # changed from 1.0 to 0.3 after noticing training was overfitting to whitespace
}
PERTURBATION_LOSS_DEFAULT_WEIGHT: float = 1.0

# (method_name, rate) pairs drawn at training time.
# Directional types (primary signal) weighted more than uniform types.
DEFAULT_PERTURBATION_POOL: List[Tuple[str, float]] = [
    ("typos",      0.05),
    ("ocr",        0.05),   # low-rate OCR (historical baseline)
    ("ocr",        0.15),   # high-rate OCR (v11: OCR at 5% was underrepresented and hardest failure)
    ("speech",     0.10),
    ("homophones", 0.30),   # v11: collapsed from 3 conditions (20/40/50%) single representative
    ("whitespace", 0.10),
    ("case",       0.10),
]


# Dataset Builders

def build_gsm8k_pairs(
    perturber: PerturbationEngine,
    perturbation_pool: List[Tuple[str, float]],
    n_per_condition: int = 200,
    clean_fraction: float = 0.25,
    split: str = "train",
) -> List[TrainingPair]:
    
    from datasets import load_dataset

    print(f"Loading GSM8K ({split} split)...")
    dataset = load_dataset("openai/gsm8k", "main", split=split)

    # Shuffle and pool all questions/answers
    all_items = list(dataset)
    random.shuffle(all_items)

    pairs: List[TrainingPair] = []

    for method, rate in perturbation_pool:
        # Sample n_per_condition items (cycle if dataset is smaller)
        for i in range(n_per_condition):
            item = all_items[i % len(all_items)]  # wrap around if we exhaust the dataset
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

    # Add clean→clean pairs (teach gate to stay closed on unperturbed input)
    n_clean = int(len(pairs) * clean_fraction)
    for i in range(n_clean):
        item = all_items[i % len(all_items)]
        q, a = item["question"], item["answer"]
        pairs.append(TrainingPair(
            clean_question=q,
            noisy_question=q,
            clean_answer=a,
            perturbation="none",
            rate=0.0,
            is_clean=True,
        ))

    random.shuffle(pairs)
    print(
        f"  GSM8K pairs: {len(pairs)} total "
        f"({len(pairs) - n_clean} perturbed, {n_clean} clean)"
    )
    # print(f"  [DEBUG] n_per_condition={n_per_condition}, clean_fraction={clean_fraction}")
    return pairs


def build_mmlu_pairs(
    perturber: PerturbationEngine,
    perturbation_pool: List[Tuple[str, float]],
    n_per_condition: int = 150,
    clean_fraction: float = 0.25,
    split: str = "auxiliary_train",
) -> List[TrainingPair]:
    
    from datasets import load_dataset

    print(f"Loading MMLU ({split} split)...")
    dataset = load_dataset("cais/mmlu", "all", split=split)
    all_items = list(dataset)
    random.shuffle(all_items)

    def _format_mmlu_question(item) -> str:
        choices_str = "\n".join(
            f"{chr(65+i)}. {c}" for i, c in enumerate(item["choices"])
        )
        return f"{item['question']}\n{choices_str}"

    def _format_mmlu_answer(item) -> str:
        return chr(65 + item["answer"])

    pairs: List[TrainingPair] = []

    for method, rate in perturbation_pool:
        for i in range(n_per_condition):
            item = all_items[i % len(all_items)]
            clean_q = _format_mmlu_question(item)
            clean_a = _format_mmlu_answer(item)
            noisy_q = perturber.apply(clean_q, method, rate)
            pairs.append(TrainingPair(
                clean_question=clean_q,
                noisy_question=noisy_q,
                clean_answer=clean_a,
                perturbation=method,
                rate=rate,
                is_clean=False,
                system_prompt=MMLU_SYSTEM_PROMPT,
            ))

    n_clean = int(len(pairs) * clean_fraction)
    for i in range(n_clean):
        item = all_items[i % len(all_items)]
        q = _format_mmlu_question(item)
        a = _format_mmlu_answer(item)
        pairs.append(TrainingPair(
            clean_question=q,
            noisy_question=q,
            clean_answer=a,
            perturbation="none",
            rate=0.0,
            is_clean=True,
            system_prompt=MMLU_SYSTEM_PROMPT,
        ))

    random.shuffle(pairs)
    print(
        f"  MMLU pairs: {len(pairs)} total "
        f"({len(pairs) - n_clean} perturbed, {n_clean} clean)"
    )
    return pairs


def build_c4_pairs(
    perturber: PerturbationEngine,
    perturbation_pool: List[Tuple[str, float]],
    n_per_condition: int = 150,
    clean_fraction: float = 0.25,
    min_words: int = 40,
    context_words: int = 30,
) -> List[TrainingPair]:
    
    from datasets import load_dataset

    all_texts: List[str] = []
    try:
        print("Loading C4 (streaming)...")
        ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
        for ex in ds:
            text = ex["text"].strip()
            words = text.split()
            if len(words) >= min_words:
                all_texts.append(text)
            if len(all_texts) >= max(n_per_condition * len(perturbation_pool) * 4, 2000):
                break
        print(f"  Collected {len(all_texts)} C4 passages.")
    except Exception as e:
        print(f"  C4 load failed ({e}); falling back to wikitext-103.")
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        for ex in ds:
            text = ex["text"].strip()
            words = text.split()
            if len(words) >= min_words:
                all_texts.append(text)
            if len(all_texts) >= 2000:
                break
        print(f"  Collected {len(all_texts)} wikitext passages.")

    random.shuffle(all_texts)

    def _split_passage(text: str):
        words = text.split()
        ctx = " ".join(words[:context_words])
        cont = " ".join(words[context_words:])
        return ctx, cont

    pairs: List[TrainingPair] = []

    for method, rate in perturbation_pool:
        for i in range(n_per_condition):
            text = all_texts[i % len(all_texts)]
            clean_q, clean_a = _split_passage(text)
            noisy_q = perturber.apply(clean_q, method, rate)
            pairs.append(TrainingPair(
                clean_question=clean_q,
                noisy_question=noisy_q,
                clean_answer=clean_a,
                perturbation=method,
                rate=rate,
                is_clean=False,
                system_prompt=C4_SYSTEM_PROMPT,
            ))

    n_clean = int(len(pairs) * clean_fraction)
    for i in range(n_clean):
        text = all_texts[i % len(all_texts)]
        q, a = _split_passage(text)
        pairs.append(TrainingPair(
            clean_question=q,
            noisy_question=q,
            clean_answer=a,
            perturbation="none",
            rate=0.0,
            is_clean=True,
            system_prompt=C4_SYSTEM_PROMPT,
        ))

    random.shuffle(pairs)
    print(
        f"  C4 pairs: {len(pairs)} total "
        f"({len(pairs) - n_clean} perturbed, {n_clean} clean)"
    )
    return pairs


def build_mixed_pairs(
    perturber: PerturbationEngine,
    perturbation_pool: List[Tuple[str, float]],
    n_per_condition: int = 500,
    clean_fraction: float = 0.25,
) -> List[TrainingPair]:
    
    n_gsm   = int(n_per_condition * 0.40)   # 200 / 500
    n_mmlu  = int(n_per_condition * 0.30)   # 150 / 500
    n_c4    = n_per_condition - n_gsm - n_mmlu  # remainder (150 / 500)

    print(f"Building mixed dataset: GSM8K={n_gsm}, MMLU={n_mmlu}, C4={n_c4} per condition")

    gsm_pairs  = build_gsm8k_pairs(perturber, perturbation_pool,
                                   n_per_condition=n_gsm,  clean_fraction=clean_fraction)
    mmlu_pairs = build_mmlu_pairs(perturber, perturbation_pool,
                                  n_per_condition=n_mmlu, clean_fraction=clean_fraction)
    c4_pairs   = build_c4_pairs(perturber, perturbation_pool,
                                n_per_condition=n_c4,   clean_fraction=clean_fraction)

    all_pairs = gsm_pairs + mmlu_pairs + c4_pairs
    random.shuffle(all_pairs)
    print(f"  Mixed total: {len(all_pairs)} pairs")
    return all_pairs


# Registry for dataset builders — add new datasets here.
# Each builder has signature: (perturber, pool, n_per_condition, ...) -> List[TrainingPair]
DATASET_BUILDERS: dict = {
    "gsm8k": build_gsm8k_pairs,
    "mmlu":  build_mmlu_pairs,
    "c4":    build_c4_pairs,
    "mixed": build_mixed_pairs,
}


# Tokenization Helpers

GSM8K_SYSTEM_PROMPT = (
    "You are a helpful assistant. Solve the math problem step by step. "
    "The last line of your response must be '#### <number>'."
)

MMLU_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the following multiple-choice question "
    "with only the letter of the correct answer (A, B, C, or D)."
)

C4_SYSTEM_PROMPT = (
    "You are a helpful assistant. Continue the following passage naturally."
)


# Few-shot template for base models (no chat template). Matches the format used
# by predict_optimal_window.py so base-model prompting is consistent across the
# pipeline. The trailing "Answer:" lets teacher forcing splice the clean answer
# directly after the prompt.
GSM8K_FEW_SHOT = """\
Solve each math problem step by step. The final answer must be on the last line as '#### NUMBER'.

Question: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much does she make every day at the farmers' market?
Answer: She eats 3 eggs and uses 4 for muffins, so she uses 3 + 4 = 7 eggs. She has 16 - 7 = 9 eggs left to sell. She earns 9 × $2 = $18.
#### 18

Question: A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?
Answer: White fiber needed is 2 / 2 = 1 bolt. Total is 2 + 1 = 3 bolts.
#### 3

Question: Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?
Answer: The value increased by 80,000 × 1.5 = $120,000. New value is 80,000 + 120,000 = $200,000. Total cost was 80,000 + 50,000 = $130,000. Profit is 200,000 - 130,000 = $70,000.
#### 70000

Question: {question}
Answer:"""


def format_question_prompt(
    tokenizer,
    question: str,
    system_prompt: str = GSM8K_SYSTEM_PROMPT,
) -> str:
    """
    Apply the tokenizer's chat template to a question (no answer appended).
    add_generation_prompt=True so the string ends with the assistant header,
    ready for the model to continue generating.

    If the tokenizer has no chat template (base models like Gemma-2-9b), fall
    back to a plain text prompt. For GSM8K we use a 3-shot template so the
    base model has in-context examples of the required '#### NUMBER' format.
    """
    if getattr(tokenizer, "chat_template", None) is None:
        # base model: no chat template, use few-shot format
        if system_prompt == GSM8K_SYSTEM_PROMPT:
            return GSM8K_FEW_SHOT.format(question=question)
        return f"{system_prompt}\n\nQuestion: {question}\nAnswer:"

    # instruct model: use chat template
    messages = [
        {"role": "system",  "content": system_prompt},
        {"role": "user",    "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def tokenize_pair(
    tokenizer,
    pair: TrainingPair,
    max_seq_len: int,
    system_prompt: str = GSM8K_SYSTEM_PROMPT,
) -> dict:
    
    sp = pair.system_prompt if pair.system_prompt else system_prompt
    # Format question prompts (both clean and noisy, no answer appended)
    clean_prompt_str = format_question_prompt(tokenizer, pair.clean_question, sp)
    noisy_prompt_str = format_question_prompt(tokenizer, pair.noisy_question, sp)

    # Tokenize prompts
    clean_prompt_ids = tokenizer(
        clean_prompt_str,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=max_seq_len,
    ).input_ids.squeeze(0)

    noisy_prompt_ids = tokenizer(
        noisy_prompt_str,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=max_seq_len,
    ).input_ids.squeeze(0)

    # Tokenize the clean answer (append EOS to signal end-of-generation)
    answer_ids = tokenizer(
        pair.clean_answer + tokenizer.eos_token,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.squeeze(0)

    # Noisy full = [noisy_prompt_tokens] + [answer_tokens]
    noisy_full_ids = torch.cat([noisy_prompt_ids, answer_ids])

    # Answer mask: True for the answer token positions (for teacher forcing)
    answer_mask = torch.zeros(len(noisy_full_ids), dtype=torch.bool)
    answer_mask[len(noisy_prompt_ids):] = True

    # Truncate to max_seq_len, keeping as much answer as possible
    if len(noisy_full_ids) > max_seq_len:
        noisy_full_ids = noisy_full_ids[:max_seq_len]
        answer_mask    = answer_mask[:max_seq_len]
        # print(f"[WARN] Truncated noisy_full_ids from {len(noisy_full_ids)} to {max_seq_len}")

    return {
        "clean_prompt_ids":  clean_prompt_ids,
        "noisy_full_ids":    noisy_full_ids,
        "answer_mask":       answer_mask,
        "noisy_prompt_len":  int(len(noisy_prompt_ids)),   # prompt tokens only, no answer
    }


def collate_batch(
    samples: List[dict],
    pad_id: int,
    device: torch.device,
) -> TokenizedBatch:
    
    def _pad_left(tensors: List[torch.Tensor], pad_value: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Left-pad tensors to same length, return padded tensor + attention mask."""
        max_len = max(t.shape[0] for t in tensors)
        padded = torch.full((len(tensors), max_len), pad_value, dtype=torch.long)
        mask   = torch.zeros(len(tensors), max_len, dtype=torch.long)
        for i, t in enumerate(tensors):
            padded[i, max_len - t.shape[0]:] = t  # pad left
            mask[i,   max_len - t.shape[0]:] = 1
        return padded, mask

    def _pad_bool_left(tensors: List[torch.Tensor]) -> torch.Tensor:
        """Left-pad boolean tensors (for answer masks)."""
        max_len = max(t.shape[0] for t in tensors)
        padded = torch.zeros(len(tensors), max_len, dtype=torch.bool)
        for i, t in enumerate(tensors):
            padded[i, max_len - t.shape[0]:] = t
        return padded

    clean_ids, clean_mask = _pad_left([s["clean_prompt_ids"] for s in samples], pad_id)
    noisy_ids, noisy_mask = _pad_left([s["noisy_full_ids"]   for s in samples], pad_id)
    ans_mask              = _pad_bool_left([s["answer_mask"] for s in samples])
    is_clean              = [s.get("is_clean", False) for s in samples]
    perturbations         = [s.get("perturbation", "none") for s in samples]

    # Left-padded layout per item: [PAD ... PAD | prompt tokens | answer tokens]
    # Answer tokens start at position: max_noisy_len - answer_len
    # where answer_len = total_len - prompt_len.
    # The hook applies correction to h[:, :prompt_end, :] (pads + prompt, no answer).
    max_noisy_len = noisy_ids.shape[1]
    padded_prompt_ends = torch.tensor(
        [max_noisy_len - (s["noisy_full_ids"].shape[0] - s["noisy_prompt_len"])
         for s in samples],
        dtype=torch.long,
    )

    return TokenizedBatch(
        clean_input_ids  = clean_ids.to(device),
        clean_attn_mask  = clean_mask.to(device),
        noisy_full_ids   = noisy_ids.to(device),
        noisy_attn_mask  = noisy_mask.to(device),
        answer_mask      = ans_mask.to(device),
        noisy_prompt_lens = padded_prompt_ends.to(device),
        is_clean         = is_clean,
        perturbations    = perturbations,
    )

# Evaluation

# Perturbation conditions used for eval (mirrors main.py experiments)
EVAL_CONDITIONS: List[Tuple[str, float]] = [
    ("none",       0.00),   # clean baseline
    ("typos",      0.05),
    ("ocr",        0.05),
    ("whitespace", 0.10),
    ("case",       0.10),
    ("speech",     0.10),
    ("homophones", 0.20),
]

# Known Phi-3.5 baselines (unmodified model, from LRD diagnostic runs)
KNOWN_BASELINES: dict = {
    "none":       None,     # measured fresh each eval run
    "typos":      70.0,
    "ocr":        67.0,
    "whitespace": 71.0,
    "case":       68.0,
    "speech":     76.5,
    "homophones": 91.5,
}


def _extract_gsm8k_answer(text: str) -> Optional[str]:
    """Extract the numeric answer from a GSM8K model output."""
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", "")
    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    return nums[-1].replace(",", "") if nums else None


def _extract_gsm8k_truth(answer_field: str) -> Optional[str]:
    """Extract the numeric ground truth from a GSM8K dataset answer field."""
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", answer_field)
    return m.group(1).replace(",", "") if m else None


def _gsm8k_correct(prediction: Optional[str], truth: Optional[str]) -> bool:
    if prediction is None or truth is None:
        return False
    try:
        return float(prediction) == float(truth)
    except ValueError:
        return False


@torch.no_grad()
def evaluate_stabilizer(
    model,
    tokenizer,
    system: MultiLayerStabilizerSystem,
    perturber,
    n_samples: int,
    device: torch.device,
    output_dir: str,
) -> dict:
    
    import re
    from datasets import load_dataset

    print("\n" + "="*60)
    print("POST-TRAINING EVALUATION")
    print("="*60)

    model.eval()
    system.stabilizers.eval()

    # Load GSM8K test split
    test_ds = load_dataset("openai/gsm8k", "main", split="test")
    test_items = list(test_ds)
    random.shuffle(test_items)
    test_items = test_items[:n_samples]

    results = {}

    for method, rate in EVAL_CONDITIONS:
        cond_name = f"{method}_{int(rate*100)}pct" if rate > 0 else "clean_baseline"
        print(f"\n--- Condition: {cond_name} (n={n_samples}) ---")

        # Build prompts
        prompts_perturbed, prompts_clean, truths = [], [], []
        for item in test_items:
            q_clean = item["question"]
            q_noisy = perturber.apply(q_clean, method, rate) if rate > 0 else q_clean
            truths.append(_extract_gsm8k_truth(item["answer"]))
            prompts_clean.append(
                format_question_prompt(tokenizer, q_clean, GSM8K_SYSTEM_PROMPT)
            )
            prompts_perturbed.append(
                format_question_prompt(tokenizer, q_noisy, GSM8K_SYSTEM_PROMPT)
            )

        gen_kwargs = dict(max_new_tokens=300, do_sample=False, pad_token_id=tokenizer.pad_token_id)

        def run_generation(prompts: List[str]) -> List[str]:
            """Run greedy generation on a list of prompt strings, batch_size=4."""
            outputs = []
            for i in range(0, len(prompts), 4):
                batch_prompts = prompts[i:i+4]
                enc = tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                ).to(device)
                with torch.no_grad():
                    out_ids = model.generate(
                        **enc,
                        **gen_kwargs,
                    )
                # Decode only newly generated tokens
                new_ids = out_ids[:, enc["input_ids"].shape[1]:]
                outputs.extend(tokenizer.batch_decode(new_ids, skip_special_tokens=True))
            return outputs

        #  Accuracy WITHOUT stabilizer (perturbed input, no correction) 
        preds_no_stab = run_generation(prompts_perturbed)
        acc_no_stab = sum(
            _gsm8k_correct(_extract_gsm8k_answer(p), t)
            for p, t in zip(preds_no_stab, truths)
        ) / len(truths) * 100

        #  Accuracy WITH stabilizer (perturbed input + correction hooks) 
        inf_hooks = system.register_inference_hooks()
        try:
            # Also capture gate stats during eval generation
            system.gate_alphas.clear()
            gate_hook_handles = []
            for L in system.config.inject_layers:
                stab = system.stabilizers[str(L)]
                def _make_gate_hook(idx, s):
                    def _h(module, args, output):
                        # Gate stats are captured inside register_inference_hooks already;
                        # we just re-capture alphas here for logging.
                        pass
                    return _h
                # (gate stats collected during run_correction; use a proxy here)

            preds_with_stab = run_generation(prompts_perturbed)
        finally:
            for h in inf_hooks:
                h.remove()

        acc_with_stab = sum(
            _gsm8k_correct(_extract_gsm8k_answer(p), t)
            for p, t in zip(preds_with_stab, truths)
        ) / len(truths) * 100

        #  Clean baseline (clean input, no stabilizer) 
        if method == "none":
            acc_clean_baseline = acc_no_stab   # clean pass is the same
        else:
            preds_clean = run_generation(prompts_clean)
            acc_clean_baseline = sum(
                _gsm8k_correct(_extract_gsm8k_answer(p), t)
                for p, t in zip(preds_clean, truths)
            ) / len(truths) * 100

        delta = acc_with_stab - acc_no_stab
        known = KNOWN_BASELINES.get(method)
        known_str = f"  (known LRD baseline: {known:.1f}%)" if known else ""

        print(f"  No stabilizer (perturbed):   {acc_no_stab:.1f}%{known_str}")
        print(f"  With stabilizer (perturbed): {acc_with_stab:.1f}%  (Δ = {delta:+.1f}%)")
        if method != "none":
            print(f"  Clean baseline:              {acc_clean_baseline:.1f}%")

        results[cond_name] = {
            "method":              method,
            "rate":                rate,
            "n_samples":           n_samples,
            "acc_no_stabilizer":   round(acc_no_stab, 2),
            "acc_with_stabilizer": round(acc_with_stab, 2),
            "acc_clean_baseline":  round(acc_clean_baseline, 2),
            "delta":               round(delta, 2),
            "known_baseline":      known,
        }

    # Print summary table
    print("\n" + "="*60)
    print(f"{'Condition':<22} {'No Stab':>8} {'W/ Stab':>8} {'Delta':>7} {'Clean':>8}")
    print("-"*60)
    for cond_name, r in results.items():
        print(
            f"{cond_name:<22} "
            f"{r['acc_no_stabilizer']:>7.1f}% "
            f"{r['acc_with_stabilizer']:>7.1f}% "
            f"{r['delta']:>+6.1f}% "
            f"{r['acc_clean_baseline']:>7.1f}%"
        )

    # Save results
    eval_path = os.path.join(output_dir, "eval_results.json")
    with open(eval_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nEval results saved: {eval_path}")

    return results


# Accuracy Loss Helper

def accuracy_loss(
    logits: torch.Tensor,           # [B, T, vocab]
    target_ids: torch.Tensor,       # [B, T]
    answer_mask: torch.Tensor,      # [B, T] bool
) -> torch.Tensor:
    """
    Cross-entropy on the answer token positions only (teacher-forcing).

    Standard causal LM shift: logit at position i predicts token i+1.
    We include position i in the loss when token i+1 is an answer token
    (i.e., answer_mask[:, 1:] is True).

    Returns scalar CE loss, or zeros if no answer tokens are present.
    """
    shift_logits = logits[:, :-1, :].contiguous()              # [B, T-1, vocab]
    shift_labels = target_ids[:, 1:].contiguous()               # [B, T-1]
    shift_mask   = answer_mask[:, 1:].bool()                    # [B, T-1]

    if not shift_mask.any():
        dummy = logits.sum() * 0.0
        return dummy

    active_logits = shift_logits[shift_mask]                    # [N, vocab]
    active_labels = shift_labels[shift_mask]                    # [N]

    return F.cross_entropy(active_logits, active_labels)


# Training Loop

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    #  Load frozen backbone 
    print(f"\nLoading frozen backbone: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map={"": 0},
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    n_backbone = sum(p.numel() for p in model.parameters())
    print(f"Backbone frozen. Total params: {n_backbone:,}")

    #  Build stabilizer system 
    hidden_dim = model.config.hidden_size
    # Compute lambda_acc: explicit override, or auto with floor to prevent
    # accuracy signal from being crowded out as other weights increase.
    lambda_acc = args.lambda_acc if args.lambda_acc is not None else \
        max(1.0 - args.lambda_stab - args.lambda_prop, args.lambda_acc_floor)

    config = StabilizerConfig(
        hidden_dim              = hidden_dim,
        # Stage 1 (embedding)
        embed_layer             = not args.no_embed_stage,
        bottleneck_dim          = args.bottleneck_dim,      # 128 for stage 1
        # Stage 2 (mid-network)
        inject_layers           = args.inject_layers,       # [2, 4]
        stage2_bottleneck_dim   = args.stage2_bottleneck_dim,
        # Shared
        use_gate                = not args.no_gate,
        max_norm                = args.max_norm,
        gate_dim                = args.gate_dim,
        dropout                 = args.dropout,
        ema_decay               = args.ema_decay,
        # Loss weights
        lambda_embed_mse        = args.lambda_embed_mse,
        lambda_stab             = args.lambda_stab,
        lambda_acc              = lambda_acc,
        lambda_gate             = args.lambda_gate,
        lambda_suppress         = args.lambda_suppress,
        lambda_adapter          = args.lambda_adapter,
        lambda_prop             = args.lambda_prop,
        probe_layers            = args.probe_layers,
        probe_weights           = args.probe_weights,
    )
    system = MultiLayerStabilizerSystem(model, config).to(device)
    # Stabilizers train in float32 even if backbone is float16
    system.stabilizers = system.stabilizers.float()

    n_trainable = system.num_trainable_params()
    gate_str = "GATELESS" if args.no_gate else f"gated (gate_dim={args.gate_dim})"
    print(f"Stabilizer system: {n_trainable:,} trainable params ({n_trainable / n_backbone:.3%} of backbone)  [{gate_str}]")
    if not args.no_embed_stage:
        print(f"Stage 1 (embed): bottleneck={args.bottleneck_dim}  λ_embed_mse={args.lambda_embed_mse:.2f}")
    else:
        print(f"Stage 1 (embed): DISABLED")
    print(f"Stage 2 (layers {args.inject_layers}): bottleneck={args.stage2_bottleneck_dim}  λ_stab={args.lambda_stab:.2f}")
    print(f"λ_acc={lambda_acc:.2f}  λ_prop={args.lambda_prop:.2f}  λ_adapter={args.lambda_adapter:.3f}")
    print(f"λ_gate={args.lambda_gate:.3f}  λ_suppress={args.lambda_suppress:.3f}")
    print(f"max_norm={args.max_norm}  ema_decay={args.ema_decay}")
    if args.gate_warmup_steps > 0:
        print(f"Curriculum: gate frozen α=1.0 for {args.gate_warmup_steps} steps, "
              f"then suppress/gate/adapter ramp 0→full over {args.reg_ramp_end_steps} steps | "
              f"stab warmup {args.stab_warmup_frac:.0%}")
    else:
        print(f"Curriculum: suppress/gate ramp {args.reg_ramp_end_steps} steps | "
              f"stab warmup {args.stab_warmup_frac:.0%}")
    if args.probe_layers:
        print(f"Probe layers: {args.probe_layers}  weights: {config.probe_weights or 'geometric-decay'}")

    #  Build training data 
    if args.dataset not in DATASET_BUILDERS:
        raise ValueError(
            f"Unknown dataset '{args.dataset}'. "
            f"Available: {list(DATASET_BUILDERS.keys())}"
        )

    perturber = PerturbationEngine()
    build_fn  = DATASET_BUILDERS[args.dataset]
    if args.perturbation_types is not None:
        allowed = set(args.perturbation_types)
        perturbation_pool = [(m, r) for m, r in DEFAULT_PERTURBATION_POOL if m in allowed]
        if not perturbation_pool:
            raise ValueError(f"--perturbation_types {args.perturbation_types!r} matched no entries in DEFAULT_PERTURBATION_POOL")
        print(f"Perturbation pool filtered to: {perturbation_pool}")
    else:
        perturbation_pool = DEFAULT_PERTURBATION_POOL
    pairs     = build_fn(
        perturber         = perturber,
        perturbation_pool = perturbation_pool,
        n_per_condition   = args.n_per_condition,
        clean_fraction    = args.clean_fraction,
    )
    print(f"Training pairs: {len(pairs)}")

    #  Optimizer & scheduler 
    optimizer = AdamW(system.stabilizers.parameters(), lr=args.lr, weight_decay=1e-2)
    total_steps = min(args.max_steps, args.epochs * (len(pairs) // args.batch_size + 1))
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr * 0.1)

    #  Training loop  
    print(f"\nStarting training — epochs={args.epochs}, max_steps={args.max_steps}, batch_size={args.batch_size}")

    global_step = 0
    all_logs = []
    optimizer.zero_grad()

    # Running EMAs for gate_ratio_stage1 / gate_ratio_stage2.
    # Tracks separate clean and noisy gate_mean per stage via EMA across steps.
    _GATE_EMA_D = 0.95
    _gate_s1_clean_ema = 0.05
    _gate_s1_noisy_ema = 0.05
    _gate_s2_clean_ema = 0.05
    _gate_s2_noisy_ema = 0.05

    # Milestone eval steps: snapshot eval at these checkpoints
    EVAL_MILESTONE_STEPS = {5000, 10000, 15000, 20000, 25000}

    for epoch in range(args.epochs):
        random.shuffle(pairs)

        for batch_start in tqdm(range(0, len(pairs), args.batch_size),
                                desc=f"Epoch {epoch+1}/{args.epochs}"):

            if global_step >= args.max_steps:
                break

            #  Curriculum weights 
            step_frac = min(1.0, global_step / max(1, args.max_steps))
            stab_mult = min(1.0, step_frac / max(1e-6, args.stab_warmup_frac))

            gate_frozen = (args.gate_warmup_steps > 0 and global_step < args.gate_warmup_steps)
            system.set_stage1_alpha_override(1.0 if gate_frozen else None)

            if gate_frozen:
                lam_gate_cur = 0.0
                lam_sup_cur  = 0.0
                lam_adp_cur  = 0.0
            else:
                # Ramp suppress/gate/adapter from 0 to full over reg_ramp_end_steps
                # measured from when the gate unfroze.
                steps_since_unfreeze = global_step - args.gate_warmup_steps
                if steps_since_unfreeze < args.reg_ramp_end_steps:
                    ramp = steps_since_unfreeze / max(1, args.reg_ramp_end_steps)
                    lam_gate_cur = ramp * config.lambda_gate
                    lam_sup_cur  = ramp * config.lambda_suppress
                    lam_adp_cur  = ramp * config.lambda_adapter
                else:
                    lam_gate_cur = config.lambda_gate
                    lam_sup_cur  = config.lambda_suppress
                    lam_adp_cur  = config.lambda_adapter

            lam_stab_cur = config.lambda_stab * stab_mult

            batch_pairs = pairs[batch_start : batch_start + args.batch_size]
            if not batch_pairs:
                continue

            #  Tokenize batch 
            try:
                samples = [
                    tokenize_pair(tokenizer, p, args.max_seq_len)
                    | {"is_clean": p.is_clean, "perturbation": p.perturbation}
                    for p in batch_pairs
                ]
            except Exception as e:
                print(f"  [skip] Tokenization error at step {global_step}: {e}")
                continue

            batch = collate_batch(samples, tokenizer.pad_token_id, device)

            #  Pass 1: clean reference (no grad) 
            with torch.no_grad(), system.capture_clean():
                model(
                    input_ids      = batch.clean_input_ids,
                    attention_mask = batch.clean_attn_mask,
                )
            # Update stage 1 embedding EMA from this clean pass (every step).
            # Stage 1 gate aug is always active from step 0 — no warmup needed.
            system.update_clean_ema(args.ema_decay)

            #  Pass 2: corrected pass (grad through stabilizers) 
            system.stabilizers.train()
            with system.run_correction(prompt_lengths=batch.noisy_prompt_lens):
                output = model(
                    input_ids      = batch.noisy_full_ids,
                    attention_mask = batch.noisy_attn_mask,
                )

            #  Losses  
            # Per-example masks — split the batch so that noisy examples always
            # contribute to the opening signal (L_embed_mse) and clean examples
            # always contribute to the closing signal (L_suppress, L_adapter),
            # regardless of what else happens to share the batch.
            is_clean_t  = torch.tensor(batch.is_clean, dtype=torch.bool, device=device)
            clean_mask  = is_clean_t                    # [B]
            noisy_mask  = ~is_clean_t                   # [B]

            # Per-sample loss weights for L_stab / L_dir: downweight whitespace
            # (and any other perturbation types listed in PERTURBATION_LOSS_WEIGHTS)
            # relative to directional types so they don't dominate training signal.
            sample_weights = torch.tensor(
                [
                    PERTURBATION_LOSS_WEIGHTS.get(p, PERTURBATION_LOSS_DEFAULT_WEIGHT)
                    for p in batch.perturbations
                ],
                dtype=torch.float32,
                device=device,
            )

            # Stage 2 MSE reconstruction (curriculum warmup).
            L_stab = system.stabilization_loss(sample_weights=sample_weights)
            L_acc  = accuracy_loss(
                logits      = output.logits.float(),
                target_ids  = batch.noisy_full_ids,
                answer_mask = batch.answer_mask,
            )
            L_prop = system.propagation_loss()

            # Max-based gate sparsity (both stages, curriculum reg_mult).
            L_gate = system.gate_sparsity_loss()

            # Clean-example penalties — computed on clean rows only.
            if clean_mask.any():
                L_suppress = system.gate_suppression_loss(sample_mask=clean_mask)
                L_adapter  = system.adapter_magnitude_loss(sample_mask=clean_mask)
            else:
                L_suppress = torch.zeros(1, device=device)
                L_adapter  = torch.zeros(1, device=device)

            # Stage 1 MSE opening signal — computed on noisy rows only.
            # (On clean examples h_noisy == h_clean so the loss would be zero, but
            # including them would dilute the gradient when the batch is mixed.)
            if config.lambda_embed_mse > 0 and noisy_mask.any():
                L_embed_mse = system.embed_mse_loss(
                    clean_ids       = batch.clean_input_ids,
                    noisy_ids       = batch.noisy_full_ids,
                    clean_attn_mask = batch.clean_attn_mask,
                    noisy_attn_mask = batch.noisy_attn_mask,
                    sample_weights  = sample_weights,
                    sample_mask     = noisy_mask,
                )
            else:
                L_embed_mse = torch.zeros(1, device=device)

            loss = (
                config.lambda_embed_mse * L_embed_mse
                + lam_stab_cur          * L_stab
                + config.lambda_acc     * L_acc
                + lam_gate_cur          * L_gate
                + config.lambda_prop    * L_prop
                + lam_sup_cur           * L_suppress
                + lam_adp_cur           * L_adapter
            )

            #  Backward + gradient accumulation 
            # No loss clamp here — cosine stab loss is in [0,2] and CE acc is
            # bounded, so the combined loss is well-scaled. Gradient clipping
            # below handles any remaining instability without zeroing gradients.
            loss = loss / args.grad_accum_steps
            loss.backward()

            if (global_step + 1) % args.grad_accum_steps == 0:
                # Measure grad norms before clipping to monitor propagation signal.
                grad_norm_total = torch.nn.utils.clip_grad_norm_(
                    system.stabilizers.parameters(), max_norm=1.0
                ).item()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            else:
                grad_norm_total = 0.0

            
            _d = _GATE_EMA_D
            if clean_mask.any():
                gs_clean = system.gate_stats(sample_mask=clean_mask)
                _gate_s1_clean_ema = _d * _gate_s1_clean_ema + (1 - _d) * gs_clean.get("gate_s1_max", 0.0)
                _gate_s2_clean_ema = _d * _gate_s2_clean_ema + (1 - _d) * gs_clean.get("gate_s2_max", 0.0)
            if noisy_mask.any():
                gs_noisy = system.gate_stats(sample_mask=noisy_mask)
                _gate_s1_noisy_ema = _d * _gate_s1_noisy_ema + (1 - _d) * gs_noisy.get("gate_s1_max", 0.0)
                _gate_s2_noisy_ema = _d * _gate_s2_noisy_ema + (1 - _d) * gs_noisy.get("gate_s2_max", 0.0)
            gate_stats = system.gate_stats()   # unmasked, for logging
            gate_ratio_s1 = _gate_s1_noisy_ema / max(_gate_s1_clean_ema, 1e-8)
            gate_ratio_s2 = _gate_s2_noisy_ema / max(_gate_s2_clean_ema, 1e-8)

            #  Logging  
            if global_step % args.log_every == 0:
                # Per-condition gate stats split by stage 1 and stage 2.
                cond_gate_s1: dict = {}
                cond_gate_s2: dict = {}
                if system.gate_alphas:
                    from stabilizer_system import _EMBED_KEY
                    # Stage 1 (embed)
                    if _EMBED_KEY in system.gate_alphas:
                        s1_per_item = system.gate_alphas[_EMBED_KEY].mean(dim=(1, 2))  # [B]
                        for cond, g in zip(batch.perturbations, s1_per_item.tolist()):
                            cond_gate_s1.setdefault(cond, []).append(g)
                    # Stage 2 (mid-network layers)
                    s2_alphas = {L: a for L, a in system.gate_alphas.items() if L != _EMBED_KEY}
                    if s2_alphas:
                        s2_per_item = torch.stack(
                            [a.mean(dim=(1, 2)) for a in s2_alphas.values()], dim=0
                        ).mean(dim=0)  # [B]
                        for cond, g in zip(batch.perturbations, s2_per_item.tolist()):
                            cond_gate_s2.setdefault(cond, []).append(g)

                # Per-condition L_stab breakdown (detached, logging only).
                cond_stab: dict = {}
                stab_per_sample = system.stabilization_loss_per_sample()   # [B] or None
                if stab_per_sample is not None:
                    for cond, s in zip(batch.perturbations, stab_per_sample.tolist()):
                        cond_stab.setdefault(cond, []).append(s)

                cond_stats_flat = {
                    **{f"gate_cond_s1_{c}": round(sum(v)/len(v), 6) for c, v in cond_gate_s1.items()},
                    **{f"gate_cond_s2_{c}": round(sum(v)/len(v), 6) for c, v in cond_gate_s2.items()},
                    **{f"stab_{c}":         round(sum(v)/len(v), 6) for c, v in cond_stab.items()},
                }

                log_entry = {
                    "step":            global_step,
                    "epoch":           epoch,
                    "loss_total":      (loss.item() * args.grad_accum_steps),
                    "loss_embed_mse":  L_embed_mse.item() if hasattr(L_embed_mse, "item") else float(L_embed_mse),
                    "loss_stab":       L_stab.item(),
                    "loss_prop":       L_prop.item(),
                    "loss_acc":        L_acc.item(),
                    "loss_gate":       L_gate.item(),
                    "loss_suppress":   L_suppress.item() if hasattr(L_suppress, "item") else float(L_suppress),
                    "loss_adapter":    L_adapter.item() if hasattr(L_adapter, "item") else float(L_adapter),
                    "grad_norm":       grad_norm_total,
                    "n_clean":         int(clean_mask.sum().item()),
                    "n_noisy":         int(noisy_mask.sum().item()),
                    "gate_frozen":     gate_frozen,
                    
                    "gate_max_ratio_s1":       round(gate_ratio_s1, 4),
                    "gate_max_ratio_s2":       round(gate_ratio_s2, 4),
                    "gate_s1_max_clean_ema":   round(_gate_s1_clean_ema, 6),
                    "gate_s1_max_noisy_ema":   round(_gate_s1_noisy_ema, 6),
                    "gate_s2_max_clean_ema":   round(_gate_s2_clean_ema, 6),
                    "gate_s2_max_noisy_ema":   round(_gate_s2_noisy_ema, 6),
                    **gate_stats,
                    **cond_stats_flat,
                }
                all_logs.append(log_entry)
                prop_str     = f"  prop={L_prop.item():.4f}" if config.probe_layers else ""
                suppress_str = f"  sup={log_entry['loss_suppress']:.4f}" if clean_mask.any() else ""
                adapter_str  = f"  adp={log_entry['loss_adapter']:.4f}" if clean_mask.any() else ""
                mse_str      = f"  emb_mse={L_embed_mse.item():.4f}" if noisy_mask.any() and config.lambda_embed_mse > 0 else ""
                batch_type   = "MIXED" if (clean_mask.any() and noisy_mask.any()) else ("CLEAN" if clean_mask.any() else "NOISY")
                print(
                    f"Step {global_step:5d} | "
                    f"{batch_type:5s} | "
                    f"{'FROZEN' if gate_frozen else f'sup={lam_sup_cur:.3f}'} stab_w={lam_stab_cur:.3f} | "
                    f"loss={log_entry['loss_total']:.4f}  "
                    f"stab={L_stab.item():.4f}"
                    f"{mse_str}"
                    f"{prop_str}"
                    f"{suppress_str}"
                    f"{adapter_str}  "
                    f"acc={L_acc.item():.4f}  "
                    f"s1_mx_n={_gate_s1_noisy_ema:.4f}  s1_mx_c={_gate_s1_clean_ema:.4f}  "
                    f"r1_max={gate_ratio_s1:.3f}  "
                    f"gnorm={grad_norm_total:.3f}"
                )

            #  Checkpoint  
            if global_step > 0 and global_step % args.save_every == 0:
                ckpt_path = os.path.join(
                    args.output_dir, f"stabilizer_step{global_step:06d}.pt"
                )
                system.save(ckpt_path, extra={"step": global_step, "epoch": epoch})
                print(f"  Checkpoint: {ckpt_path}")

            #  Milestone eval 
            # Quick accuracy snapshot at key steps so we can abort early if
            # gate_ratio_s1 is still ~1.0 at step 3000 (architecture check).
            if global_step in EVAL_MILESTONE_STEPS and args.eval_after_training:
                milestone_dir = os.path.join(args.output_dir, f"eval_step{global_step:06d}")
                os.makedirs(milestone_dir, exist_ok=True)
                print(f"\n  [Milestone eval at step {global_step}]  "
                      f"gate_max_ratio_s1={gate_ratio_s1:.3f}  gate_max_ratio_s2={gate_ratio_s2:.3f}")
                evaluate_stabilizer(
                    model=model, tokenizer=tokenizer, system=system,
                    perturber=perturber, n_samples=args.eval_n_samples,
                    device=device, output_dir=milestone_dir,
                )
                system.stabilizers.train()   # return to train mode after eval

            global_step += 1

        if global_step >= args.max_steps:
            break

    #  Save final weights + logs 
    final_path = os.path.join(args.output_dir, "stabilizer_final.pt")
    system.save(
        final_path,
        extra={
            "step":    global_step,
            "model":   args.model,
            "dataset": args.dataset,
        },
    )
    print(f"\nTraining complete. Saved: {final_path}")

    log_path = os.path.join(args.output_dir, "training_logs.json")
    with open(log_path, "w") as f:
        json.dump(all_logs, f, indent=2)
    print(f"Logs: {log_path}")

    #  Post-training evaluation 
    if args.eval_after_training:
        evaluate_stabilizer(
            model      = model,
            tokenizer  = tokenizer,
            system     = system,
            perturber  = perturber,
            n_samples  = args.eval_n_samples,
            device     = device,
            output_dir = args.output_dir,
        )

    return system


#  
# CLI
#  

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train multi-layer LRD stabilizer on clean/noisy prompt pairs."
    )

    #  Model  
    parser.add_argument("--model",   type=str, default="microsoft/Phi-3.5-mini-instruct",
                        help="HuggingFace model ID or local path.")
    parser.add_argument("--dataset", type=str, default="gsm8k",
                        choices=list(DATASET_BUILDERS.keys()),
                        help="Training dataset.")

    #  Architecture  
    parser.add_argument("--inject_layers",  type=int, nargs="+",
                        default=[2, 4],
                        help="Stage 2 transformer layer indices (0-indexed). Default [2, 4].")
    parser.add_argument("--bottleneck_dim", type=int, default=128,
                        help="Stage 1 (embedding) correction MLP bottleneck width.")
    parser.add_argument("--stage2_bottleneck_dim", type=int, default=32,
                        help="Stage 2 (mid-network) correction MLP bottleneck width.")
    parser.add_argument("--no_gate",        action="store_true",
                        help="Disable the gate: apply delta directly (h_out = h + delta). "
                             "gate_net is not created. Norm clip controls regression risk.")
    parser.add_argument("--no_embed_stage", action="store_true",
                        help="Disable stage 1 (embedding-level correction). "
                             "Only stage 2 (inject_layers) is active.")
    parser.add_argument("--max_norm",       type=float, default=0.05,
                        help="Correction cap as a fraction of input embedding norm (e.g. 0.05 = 5%%). "
                             "Adaptive: delta is clipped to max_norm * ‖h‖ per token.")
    parser.add_argument("--gate_dim",       type=int, default=64,
                        help="Gate controller hidden dimension (unused when --no_gate).")
    parser.add_argument("--dropout",        type=float, default=0.1)

    #  Loss weights  
    parser.add_argument("--lambda_embed_mse", type=float, default=0.35,
                        help="Weight for stage 1 MSE reconstruction loss (v9, replaces contrastive).")
    parser.add_argument("--lambda_stab",    type=float, default=0.15,
                        help="Weight for stage 2 MSE reconstruction loss.")
    parser.add_argument("--lambda_acc",     type=float, default=None,
                        help="Weight for L_accuracy. If not set, auto-computed with floor.")
    parser.add_argument("--lambda_acc_floor", type=float, default=0.25,
                        help="Minimum lambda_acc when auto-computing.")
    parser.add_argument("--lambda_gate",    type=float, default=0.15,
                        help="Max-based gate sparsity weight (both stages) — final value after ramp.")
    parser.add_argument("--lambda_gate_start", type=float, default=0.02,
                        help="Initial lambda_gate at step 0; ramps up to --lambda_gate by --reg_ramp_end_steps.")
    parser.add_argument("--lambda_suppress", type=float, default=0.20,
                        help="Gate suppression weight for clean steps — final value after ramp.")
    parser.add_argument("--lambda_suppress_start", type=float, default=0.02,
                        help="Initial lambda_suppress at step 0; ramps up to --lambda_suppress by --reg_ramp_end_steps.")
    parser.add_argument("--lambda_adapter", type=float, default=0.15,
                        help="Adapter magnitude weight for clean steps (both stages).")
    parser.add_argument("--lambda_prop",    type=float, default=0.10,
                        help="Weight for propagation loss at probe layers.")
    parser.add_argument("--probe_layers",   type=int, nargs="*", default=[9],
                        help="Read-only probe layer indices. Default [9].")
    parser.add_argument("--probe_weights",  type=float, nargs="*", default=[],
                        help="Per-probe-layer weights. Defaults to geometric decay.")
    parser.add_argument("--ema_decay",      type=float, default=0.99,
                        help="EMA decay for stage 1 embedding clean-state reference.")

    #  Curriculum  
    parser.add_argument("--gate_warmup_steps",   type=int, default=0,
                        help="Steps to hard-fix stage 1 α=1.0 so fc2 learns correction direction "
                             "before gate is trained. 0 disables (gate always learned).")
    parser.add_argument("--reg_warmup_steps",    type=int, default=2000,
                        help="(Legacy, unused with gate_warmup_steps>0) Steps at full reg_mult=2.0.")
    parser.add_argument("--reg_decay_end_steps", type=int, default=8000,
                        help="(Legacy, unused with gate_warmup_steps>0) Step reg_mult reaches 1.0.")
    parser.add_argument("--reg_ramp_end_steps",  type=int, default=3000,
                        help="Steps after gate unfreezes over which suppress/gate/adapter ramp "
                             "from 0 to their full lambda values.")
    parser.add_argument("--stab_warmup_frac",    type=float, default=0.25,
                        help="Fraction of training over which L_stab ramps from 0 to full.")

    #  Training  
    parser.add_argument("--epochs",          type=int,   default=5)
    parser.add_argument("--max_steps",       type=int,   default=5000)
    parser.add_argument("--batch_size",      type=int,   default=4)
    parser.add_argument("--lr",              type=float, default=1e-4)
    parser.add_argument("--grad_accum_steps",type=int,   default=4,
                        help="Gradient accumulation steps (effective batch = batch_size * grad_accum_steps).")
    parser.add_argument("--max_seq_len",     type=int,   default=512,
                        help="Max sequence length (question + answer).")

    #  Data  
    parser.add_argument("--n_per_condition", type=int,   default=200,
                        help="Training pairs per perturbation condition.")
    parser.add_argument("--perturbation_types", type=str, nargs="+", default=None,
                        help="Restrict training pool to these perturbation types "
                             "(e.g. --perturbation_types typos). Default: use all.")
    parser.add_argument("--clean_fraction",  type=float, default=0.25,
                        help="Fraction of training data that is clean→clean "
                             "(teaches gate to stay closed on unperturbed input).")

    #  Output  
    parser.add_argument("--output_dir", type=str, default="./stabilizer_weights")
    parser.add_argument("--log_every",  type=int, default=25)
    parser.add_argument("--save_every", type=int, default=500)

    #  Post-training evaluation 
    parser.add_argument("--eval_after_training", action="store_true", default=True,
                        help="Run accuracy eval on GSM8K test split after training completes.")
    parser.add_argument("--no_eval_after_training", dest="eval_after_training",
                        action="store_false",
                        help="Skip post-training evaluation.")
    parser.add_argument("--eval_n_samples", type=int, default=100,
                        help="Number of GSM8K test examples per condition during eval.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
