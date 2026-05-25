"""
train_stabilizer.py

Training Script for the Plug-and-Play Depth-Wise Stabilizer
=============================================================
Strategy:
    - Backbone (Phi-3.5-mini) is FULLY FROZEN throughout.
    - Only the DepthWiseStabilizer parameters are trained.
    - Training corpus: diverse general text (C4 + Wikipedia + code)
      to ensure zero-shot generalization to evaluation benchmarks.

Data Pipeline:
    - For each batch, we run TWO forward passes through the frozen LLM:
        1. Clean pass   -> h_clean   (reference hidden states)
        2. Noisy pass   -> h_noisy   (hidden states under perturbation)
    - We capture hidden states at:
        - Injection layer L   (where the stabilizer sits)
        - Previous layer L-1  (for depth-wise loss)
    - The stabilizer corrects h_noisy -> h_noisy_stabilized
    - Loss pulls h_noisy_stabilized toward h_clean

Injection:
    - We use register_forward_hook to capture intermediate hidden states
      non-invasively, without modifying the LLM's architecture.

Usage:
    python train_stabilizer.py \
        --model microsoft/Phi-3.5-mini-instruct \
        --inject_layer 16 \
        --epochs 3 \
        --batch_size 4 \
        --output_dir ./stabilizer_weights
"""

import os
import json
import argparse
import random
from contextlib import contextmanager

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, interleave_datasets
from tqdm import tqdm

from stabilizer import DepthWiseStabilizer, compute_total_loss
from perturbations import PerturbationEngine


# ==============================================================================
# Configuration
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train the Depth-Wise Stabilizer")

    # Model
    parser.add_argument("--model", type=str, default="microsoft/Phi-3.5-mini-instruct")
    parser.add_argument("--inject_layer", type=int, default=16,
                        help="Which transformer layer to inject the stabilizer (0-indexed). "
                             "Phi-3.5 has 32 layers; 16 is the midpoint.")

    # Training
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_seq_len", type=int, default=256,
                        help="Truncate sequences to this length to fit in VRAM.")
    parser.add_argument("--max_steps", type=int, default=5000,
                        help="Total training steps. Overrides epochs if reached first.")
    parser.add_argument("--grad_accum_steps", type=int, default=4,
                        help="Gradient accumulation for effective larger batch size.")

    # Loss weights
    parser.add_argument("--lambda_robustness", type=float, default=1.0)
    parser.add_argument("--lambda_depth",      type=float, default=0.5)
    parser.add_argument("--lambda_gate",       type=float, default=0.01,
                        help="Sparsity pressure on gate. Higher = more selective intervention.")

    # Data
    parser.add_argument("--corpus_dir", type=str, default=None,
                        help="Path to pre-downloaded corpus (from download_corpus.py). "
                             "Strongly recommended on HPC clusters to avoid network timeouts.")
    parser.add_argument("--corpus_samples", type=int, default=50000,
                        help="Number of text samples to draw from the training corpus.")

    # Output
    parser.add_argument("--output_dir", type=str, default="./stabilizer_weights")
    parser.add_argument("--log_every",  type=int, default=50)
    parser.add_argument("--save_every", type=int, default=500)

    return parser.parse_args()


# ==============================================================================
# Corpus Loading — Diverse General Text (NOT evaluation benchmarks)
# ==============================================================================

def load_training_corpus(tokenizer, max_seq_len: int, n_samples: int, corpus_dir: str = None):
    """
    Load training corpus from local disk (preferred) or stream from HuggingFace.

    On HPC clusters, always pre-download with download_corpus.py first:
        python download_corpus.py --save_dir /scratch/$USER/stabilizer_corpus
    Then pass --corpus_dir /scratch/$USER/stabilizer_corpus to this script.

    Falls back to streaming if corpus_dir is not provided, but this will
    time out on clusters with restricted outbound network access.
    """
    from datasets import load_from_disk

    all_texts = []

    if corpus_dir and os.path.exists(corpus_dir):
        # ------------------------------------------------------------------
        # LOCAL DISK PATH — fast, no network, recommended for HPC
        # ------------------------------------------------------------------
        print(f"Loading corpus from local disk: {corpus_dir}")

        for split_name, weight in [("c4", 0.4), ("wiki", 0.4), ("code", 0.2)]:
            split_path = os.path.join(corpus_dir, split_name)
            if not os.path.exists(split_path):
                print(f"  Warning: {split_name} not found at {split_path}, skipping.")
                continue

            ds = load_from_disk(split_path)
            n_take = int(n_samples * weight)
            texts = [row["text"] for row in ds.select(range(min(n_take, len(ds))))]
            all_texts.extend(texts)
            print(f"  Loaded {len(texts)} samples from {split_name}")

    else:
        # ------------------------------------------------------------------
        # STREAMING FALLBACK — may time out on HPC, use download_corpus.py
        # ------------------------------------------------------------------
        print("corpus_dir not set or not found. Streaming from HuggingFace (may time out on HPC).")
        print("Tip: run `python download_corpus.py --save_dir /scratch/$USER/stabilizer_corpus` first.\n")

        sources = [
            ("allenai/c4",           {"name": "en"},          "text",    0.4),
            ("wikimedia/wikipedia",  {"name": "20231101.en"}, "text",    0.4),
            # Open, non-gated code datasets in order of preference
            ("TokenBender/code_instructions_122k_alpaca_style", {}, "output", 0.2),
        ]
        for ds_name, kwargs, text_col, weight in sources:
            try:
                ds = load_dataset(ds_name, split="train", streaming=True, **kwargs)
                n_take = int(n_samples * weight)
                count = 0
                for item in ds:
                    if count >= n_take:
                        break
                    text = item.get(text_col, "").strip()
                    if len(text) > 50:
                        all_texts.append(text)
                        count += 1
                print(f"  Streamed {count} samples from {ds_name}")
            except Exception as e:
                print(f"  Failed to stream {ds_name}: {e}")

    # ------------------------------------------------------------------
    # Tokenize all collected texts
    # ------------------------------------------------------------------
    import random
    random.shuffle(all_texts)
    all_texts = all_texts[:n_samples]

    samples = []
    print(f"\nTokenizing {len(all_texts)} corpus samples (max_len={max_seq_len})...")

    for text in tqdm(all_texts):
        text = text.strip()
        if len(text) < 50:
            continue

        encoded = tokenizer(
            text,
            max_length=max_seq_len,
            truncation=True,
            padding=False,
            return_tensors="pt"
        )

        if encoded["input_ids"].shape[1] < 16:
            continue

        samples.append(encoded["input_ids"].squeeze(0))

    print(f"Collected {len(samples)} valid training samples.")
    return samples


def collate_fn(batch, pad_token_id: int, device: torch.device):
    """Pad a batch of variable-length sequences to the same length."""
    max_len = max(t.shape[0] for t in batch)
    padded = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)

    for i, t in enumerate(batch):
        padded[i, :t.shape[0]] = t
        attention_mask[i, :t.shape[0]] = 1

    return padded.to(device), attention_mask.to(device)


# ==============================================================================
# Hidden State Capture via Forward Hooks
# ==============================================================================

class HiddenStateCapture:
    """
    Context manager that registers forward hooks on two consecutive transformer
    layers to capture their output hidden states.

    We capture:
        - layer L   (injection point): where the stabilizer will operate
        - layer L-1 (previous layer):  needed for the depth-wise stability loss

    The hook captures the FIRST element of the layer's output tuple,
    which is the hidden state tensor [B, T, D] for all HuggingFace models.
    """

    def __init__(self, model: nn.Module, inject_layer: int):
        self.inject_layer = inject_layer
        self.h_inject = None  # Hidden states at layer L
        self.h_prev   = None  # Hidden states at layer L-1
        self._hooks   = []

        # Resolve the transformer layer list
        # Works for Phi-3.5, Llama-3, Mistral (all use model.model.layers)
        self.layers = model.model.layers

    @contextmanager
    def capture(self):
        """Use as: with capture_obj.capture(): model(...)"""
        try:
            self._register()
            yield self
        finally:
            self._remove()

    def _register(self):
        L = self.inject_layer

        def hook_prev(module, args, output):
            # output[0] is hidden states; some layers return tuples
            hs = output[0] if isinstance(output, tuple) else output
            self.h_prev = hs.detach().clone()  # Detach: we don't backprop through backbone

        def hook_inject(module, args, output):
            hs = output[0] if isinstance(output, tuple) else output
            self.h_inject = hs.detach().clone()

        self._hooks.append(self.layers[L - 1].register_forward_hook(hook_prev))
        self._hooks.append(self.layers[L].register_forward_hook(hook_inject))

    def _remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ==============================================================================
# Perturbation Sampling
# ==============================================================================

PERTURBATION_POOL = [
    ("typos",      0.05),
    ("ocr",        0.05),
    ("whitespace", 0.10),
    ("homophones", 0.20),
    ("speech",     0.10),
]

def sample_perturbation(perturber: PerturbationEngine, text: str) -> str:
    """Randomly pick one perturbation type and apply it to the text."""
    method, rate = random.choice(PERTURBATION_POOL)
    return perturber.apply(text, method, rate)


# ==============================================================================
# Main Training Loop
# ==============================================================================

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load Frozen Backbone
    # ------------------------------------------------------------------
    print(f"Loading frozen backbone: {args.model}")
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
    model.eval()  # FROZEN: always in eval mode

    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    print(f"Backbone frozen. Total params: {sum(p.numel() for p in model.parameters()):,}")

    # ------------------------------------------------------------------
    # 2. Initialize Stabilizer (the ONLY thing that trains)
    # ------------------------------------------------------------------
    hidden_dim = model.config.hidden_size  # 3072 for Phi-3.5-mini
    stabilizer = DepthWiseStabilizer(hidden_dim=hidden_dim).to(device).to(torch.float32)

    trainable_params = sum(p.numel() for p in stabilizer.parameters())
    print(f"Stabilizer initialized. Trainable params: {trainable_params:,}")
    print(f"  Parameter ratio: {trainable_params / sum(p.numel() for p in model.parameters()):.4%}")

    # ------------------------------------------------------------------
    # 3. Load Training Corpus
    # ------------------------------------------------------------------
    corpus = load_training_corpus(tokenizer, args.max_seq_len, args.corpus_samples, args.corpus_dir)
    perturber = PerturbationEngine()

    # ------------------------------------------------------------------
    # 4. Optimizer & Scheduler
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        stabilizer.parameters(),
        lr=args.lr,
        weight_decay=1e-2,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.max_steps,
        eta_min=args.lr * 0.1,
    )

    # ------------------------------------------------------------------
    # 5. Hidden State Capture
    # ------------------------------------------------------------------
    capture = HiddenStateCapture(model, inject_layer=args.inject_layer)

    # ------------------------------------------------------------------
    # 6. Training Loop
    # ------------------------------------------------------------------
    print(f"\nStarting training: inject_layer={args.inject_layer}, "
          f"epochs={args.epochs}, max_steps={args.max_steps}")

    global_step = 0
    all_logs = []

    for epoch in range(args.epochs):
        random.shuffle(corpus)

        # Simple manual batching
        for batch_start in tqdm(range(0, len(corpus), args.batch_size),
                                desc=f"Epoch {epoch+1}/{args.epochs}"):

            if global_step >= args.max_steps:
                break

            batch_tokens = corpus[batch_start : batch_start + args.batch_size]
            if len(batch_tokens) == 0:
                continue

            input_ids, attention_mask = collate_fn(batch_tokens, tokenizer.pad_token_id, device)

            # ----------------------------------------------------------------
            # A. CLEAN FORWARD PASS — capture reference hidden states
            # ----------------------------------------------------------------
            with torch.no_grad():
                with capture.capture():
                    model(input_ids=input_ids, attention_mask=attention_mask)

            h_clean = capture.h_inject.to(torch.float32)  # [B, T, D]
            h_prev  = capture.h_prev.to(torch.float32)    # [B, T, D]

            # ----------------------------------------------------------------
            # B. NOISY FORWARD PASS — decode tokens, perturb text, re-encode
            # 50% of steps use a perturbed pass; 50% use the clean text again.
            # This gives the gate a contrastive signal: it should open on noisy
            # batches and stay closed on clean ones, teaching it to be dynamic
            # rather than settling at a fixed uniform intervention level.
            # ----------------------------------------------------------------
            texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)

            is_noisy_step = (global_step % 2 == 0)
            if is_noisy_step:
                noisy_texts = [sample_perturbation(perturber, t) for t in texts]
            else:
                noisy_texts = texts  # Clean pass: gate should learn to stay near 0

            noisy_encoded = tokenizer(
                noisy_texts,
                max_length=args.max_seq_len,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                with capture.capture():
                    model(
                        input_ids=noisy_encoded["input_ids"],
                        attention_mask=noisy_encoded["attention_mask"],
                    )

            h_noisy = capture.h_inject.to(torch.float32)  # [B, T', D]

            # ----------------------------------------------------------------
            # C. Align sequence lengths (noisy text may tokenize differently)
            # ----------------------------------------------------------------
            # Truncate both to the shorter sequence length for loss computation
            min_seq = min(h_clean.shape[1], h_noisy.shape[1])
            h_clean_aligned = h_clean[:, :min_seq, :]
            h_prev_aligned  = h_prev[:,  :min_seq, :]
            h_noisy_aligned = h_noisy[:, :min_seq, :]

            # ----------------------------------------------------------------
            # D. STABILIZER FORWARD + LOSS
            # ----------------------------------------------------------------
            h_noisy_stabilized, alpha = stabilizer(h_noisy_aligned)

            loss, loss_dict = compute_total_loss(
                h_noisy_stabilized = h_noisy_stabilized,
                h_clean            = h_clean_aligned.detach(),
                h_prev_layer       = h_prev_aligned.detach(),
                alpha              = alpha,
                lambda_robustness  = args.lambda_robustness,
                lambda_depth       = args.lambda_depth,
                lambda_gate        = args.lambda_gate,
            )

            # On clean steps, add a suppression loss so the gate receives a direct
            # "stay closed" gradient. Target: gate mean near 0.05 on clean inputs.
            if not is_noisy_step:
                gate_suppression = (alpha.mean() - 0.05).clamp(min=0.0) * 2.0
                loss = loss + gate_suppression
                loss_dict["gate_suppression"] = gate_suppression.item()
            else:
                loss_dict["gate_suppression"] = 0.0

            loss_dict["step_type"] = "noisy" if is_noisy_step else "clean"

            # Gradient accumulation
            # Clamp before dividing to prevent a single outlier batch from
            # causing a destructive weight update even after grad clipping.
            loss = torch.clamp(loss, max=50.0) / args.grad_accum_steps
            loss.backward()

            if (global_step + 1) % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(stabilizer.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # ----------------------------------------------------------------
            # E. Logging
            # Always log both a noisy AND a clean step so we can see the
            # gate contrast, regardless of how log_every aligns with step parity.
            # ----------------------------------------------------------------
            should_log = (global_step % args.log_every == 0) or \
                         (global_step % args.log_every == 1)  # log consecutive pairs

            if should_log:
                log_entry = {"step": global_step, "epoch": epoch, **loss_dict}
                all_logs.append(log_entry)
                print(
                    f"Step {global_step:5d} | "
                    f"{loss_dict['step_type'].upper():5s} | "
                    f"Loss: {loss_dict['loss_total']:.4f} | "
                    f"Rob: {loss_dict['loss_robustness']:.4f} | "
                    f"Gate mean: {loss_dict['gate_alpha_mean']:.4f} | "
                    f"Gate max: {loss_dict['gate_alpha_max']:.4f} | "
                    f"Suppress: {loss_dict['gate_suppression']:.4f}"
                )

            # ----------------------------------------------------------------
            # F. Checkpoint
            # ----------------------------------------------------------------
            if global_step % args.save_every == 0 and global_step > 0:
                ckpt_path = os.path.join(args.output_dir, f"stabilizer_step{global_step}.pt")
                torch.save({
                    "step": global_step,
                    "stabilizer_state_dict": stabilizer.state_dict(),
                    "optimizer_state_dict":  optimizer.state_dict(),
                    "args": vars(args),
                }, ckpt_path)
                print(f"Checkpoint saved: {ckpt_path}")

            global_step += 1

        if global_step >= args.max_steps:
            break

    # ------------------------------------------------------------------
    # 7. Save Final Weights + Logs
    # ------------------------------------------------------------------
    final_path = os.path.join(args.output_dir, "stabilizer_final.pt")
    torch.save({
        "step": global_step,
        "stabilizer_state_dict": stabilizer.state_dict(),
        "args": vars(args),
    }, final_path)
    print(f"\nTraining complete. Final weights saved: {final_path}")

    log_path = os.path.join(args.output_dir, "training_logs.json")
    with open(log_path, "w") as f:
        json.dump(all_logs, f, indent=2)
    print(f"Training logs saved: {log_path}")

    return stabilizer


if __name__ == "__main__":
    args = parse_args()
    train(args)
