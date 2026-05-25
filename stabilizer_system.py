"""
stabilizer_system.py

Multi-Layer Gated Stabilizer System — v9 Two-Stage Architecture
=================================================================
Model-agnostic stabilizer for frozen LLMs.

v9 design (two-stage):
  Stage 1 — Embedding-level correction (before layer 0)
    Gate reads concat([emb, |emb - emb_clean_mean|]).  At the embedding level,
    character-level perturbations (typos, OCR) are directly visible as embedding
    distance, giving the gate a real discrimination signal for the first time.
    Objective: MSE reconstruction toward h_clean (replaces v7/v8 contrastive).
    v8 failure: contrastive loss requires a detectable shift in cosine distance;
    with max_norm=1.0 and embedding magnitude ~55, the correction is 0.45% of
    embedding scale so d_corr ≈ d_noisy always and gradient underflows to zero.
    MSE gives gradient proportional to (h_corr - h_clean) regardless of magnitude.
    EMA augmentation always active — embedding distributions are stable from step 0.

  Stage 2 — Mid-network refinement at layers specified in inject_layers (e.g. 2, 4)
    Gate reads h_norm only (no EMA aug — mid-network signal is too weak).
    Objective: MSE reconstruction toward h_clean at those layers.
    Smaller bottleneck (stage2_bottleneck_dim=32) — these are safety-valve corrections.

Compatible with: Phi-3.5, Llama-3, Mistral, Qwen2.5, Falcon, GPT-NeoX, etc.

Typical usage (training)
------------------------
    config = StabilizerConfig(
        hidden_dim=3072,
        inject_layers=[2, 4],      # stage 2 layers
        embed_layer=True,          # stage 1
        lambda_embed_mse=0.35,
        lambda_stab=0.15,
        lambda_acc=0.40,
    )
    system = MultiLayerStabilizerSystem(model, config)

    with torch.no_grad(), system.capture_clean():
        model(clean_question_ids, attention_mask=clean_attn_mask)

    with system.run_correction(prompt_lengths=batch.noisy_prompt_lens):
        output = model(noisy_full_ids, attention_mask=noisy_attn_mask)

    L_embed_mse = system.embed_mse_loss(sample_weights)     # stage 1
    L_stab      = system.stabilization_loss(sample_weights) # stage 2
    L_acc       = ce_on_answer_tokens(output.logits, noisy_full_ids, answer_mask)
    L_gate      = system.gate_sparsity_loss()
    loss = config.lambda_embed_mse * L_embed_mse + config.lambda_stab * L_stab + ...

Typical usage (inference)
-------------------------
    system = MultiLayerStabilizerSystem.load(model, "stabilizer_final.pt")
    handles = system.register_inference_hooks()
    # ... run model normally; stabilizer corrects in-place ...
    for h in handles:
        h.remove()
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StabilizerConfig:
    """
    All hyperparameters for the two-stage multi-layer stabilizer system.

    Stage 1 (embed_layer=True):
        Injection point: embed_tokens output (before layer 0).
        Bottleneck dim:  bottleneck_dim (default 128).
        Gate input:      concat([emb_norm, |emb_norm - emb_clean_mean|]).
        Loss:            MSE reconstruction — lambda_embed_mse * L_embed_mse.

    Stage 2 (inject_layers=[2, 4] or similar):
        Injection point: after transformer block at each listed layer index.
        Bottleneck dim:  stage2_bottleneck_dim (default 32).
        Gate input:      h_norm only (no EMA augmentation).
        Loss:            MSE reconstruction — lambda_stab * L_stab.

    Hidden dimensions by model:
        Phi-3.5-mini:   3072    (32 layers)
        Llama-3-8B:     4096    (32 layers)
        Mistral-7B:     4096    (32 layers)
        Qwen2.5-7B:     3584    (28 layers)
    """
    # ── Architecture ──────────────────────────────────────────────────────
    hidden_dim: int = 3072

    # Stage 1 (embedding level)
    embed_layer: bool = True            # Enable stage 1 embedding correction
    bottleneck_dim: int = 128           # Stage 1 correction MLP bottleneck (D → r → D)

    # Stage 2 (mid-network)
    stage2_bottleneck_dim: int = 32     # Stage 2 correction MLP bottleneck (smaller)

    # Shared
    use_gate: bool = True              # If False, delta applied directly (no gate_net created)
    max_norm: float = 0.05             # Correction cap as fraction of input ‖h‖ (adaptive)
    gate_dim: int = 64                 # Gate controller hidden dimension
    dropout: float = 0.1

    # ── Injection points ──────────────────────────────────────────────────
    inject_layers: List[int] = field(default_factory=lambda: [2, 4])  # stage 2

    # ── Loss weights ──────────────────────────────────────────────────────
    lambda_embed_mse: float = 0.0     # stage 1 MSE reconstruction loss
    lambda_stab: float = 0.15         # stage 2 cosine-sim alignment
    lambda_acc: float = 0.40          # cross-entropy accuracy (independent — no sum constraint)
    lambda_gate: float = 0.15         # max gate sparsity
    lambda_suppress: float = 0.20     # mean gate suppression — clean steps only
    lambda_adapter: float = 0.15      # adapter delta magnitude — clean steps only

    # ── Propagation-aware loss ─────────────────────────────────────────────
    probe_layers: List[int] = field(default_factory=list)
    probe_weights: List[float] = field(default_factory=list)
    lambda_prop: float = 0.0

    # ── EMA for stage 1 gate augmentation ─────────────────────────────────
    # Stage 1 always uses EMA augmentation (no warmup needed for embeddings).
    # ema_decay controls the update rate of the embedding clean-state EMA.
    ema_decay: float = 0.99

    # ── Deprecated fields — kept for backward compat when loading old checkpoints ──
    lora_rank: int = 0
    bottleneck_ratio: int = 4
    lambda_lrd: float = 0.0
    lambda_dir: float = 0.0
    gate_aug: bool = True
    gate_aug_warmup_steps: int = 500
    lambda_contrast: float = 0.0      # v7/v8 — replaced by lambda_embed_mse in v9
    contrastive_margin: float = 0.1   # v7/v8 — unused in v9

    # ── Model adapter ─────────────────────────────────────────────────────
    layers_attr: Optional[str] = None   # Override transformer layer resolution
    embed_attr: Optional[str] = None    # Override embedding module resolution


# ─────────────────────────────────────────────────────────────────────────────
# Per-Layer Stabilizer Module
# ─────────────────────────────────────────────────────────────────────────────

class DepthWiseStabilizer(nn.Module):
    """
    Norm-clipped bottleneck MLP correction module.

    Used for both stage 1 (embedding) and stage 2 (mid-network):

    Stage 1 (use_clean_mean_aug=True, aug_active=True from step 0):
        Gate input: concat([h_norm, |h_norm - h_clean_mean|])  [2D]
        Provides direct perturbation discrimination at the embedding level.

    Stage 2 (use_clean_mean_aug=False):
        Gate input: h_norm only  [D]
        These gates are expected to stay mostly closed; EMA divergence signal
        is not useful at mid-network layers due to spike-and-suppress collapsing LRD.

    Correction path (both stages):
        h_norm  = LayerNorm(h)
        delta   = fc2(GELU(fc1(h_norm)))              [B, T, D]
        delta   = delta * clamp(‖h‖*max_norm/‖delta‖, max=1)  adaptive norm clip
        alpha   = gate_net(gate_in)                   [B, T, 1]
        h_out   = h + alpha * delta

    max_norm is a *fraction* of the input embedding norm, not an absolute value.
    With max_norm=0.05 and ‖h‖≈55 (Phi-3.5 embeddings), the correction cap is
    ~2.75 per token (~5% of embedding magnitude).  This keeps the correction
    proportional regardless of model or layer.
    """

    def __init__(
        self,
        hidden_dim: int,
        bottleneck_dim: int = 128,
        gate_dim: int = 64,
        dropout: float = 0.1,
        max_norm: float = 1.0,
        use_clean_mean_aug: bool = False,
        use_gate: bool = True,
        # Legacy args accepted but ignored
        lora_rank: int = 0,
        bottleneck_ratio: int = 4,
    ):
        super().__init__()
        self.max_norm = max_norm
        self.use_clean_mean_aug = use_clean_mean_aug
        self.use_gate = use_gate
        self.aug_active = False          # flip to True for stage 1 from init; for stage 2 unused
        self.alpha_override: Optional[float] = None  # when set, gate_net is bypassed entirely

        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Bottleneck MLP: D → bottleneck_dim → D  (fc2=0 init → delta=0 at step 0)
        self.fc1 = nn.Linear(hidden_dim, bottleneck_dim, bias=True)
        self.fc2 = nn.Linear(bottleneck_dim, hidden_dim, bias=False)

        # Gate controller (only created when use_gate=True).
        # When use_gate=False the correction is applied directly: h_out = h + delta.
        if use_gate:
            gate_in_dim = hidden_dim * 2 if use_clean_mean_aug else hidden_dim
            self.gate_net = nn.Sequential(
                nn.Linear(gate_in_dim, gate_dim),
                nn.GELU(),
                nn.Linear(gate_dim, 1),
                nn.Sigmoid(),
            )
        else:
            self.gate_net = None

        # EMA of clean normalised hidden states at this layer [D].
        # Only used when use_clean_mean_aug=True (stage 1).
        self.register_buffer("h_clean_mean", torch.zeros(hidden_dim))

        self._init_weights()

    def _init_weights(self):
        # MLP: fc1 kaiming; fc2 init depends on stage.
        # Stage 1 (use_clean_mean_aug=True): small random init on fc2 so that
        #   delta ≠ 0 from step 0, giving contrastive loss a non-zero gradient
        #   w.r.t. fc2 from the very first step (zero-init creates a dead zone).
        # Stage 2 (use_clean_mean_aug=False): zero init → exact no-op at step 0.
        nn.init.kaiming_uniform_(self.fc1.weight, a=math.sqrt(5))
        nn.init.zeros_(self.fc1.bias)
        if self.use_clean_mean_aug:
            nn.init.normal_(self.fc2.weight, std=0.01)  # stage 1: break gradient dead zone
        else:
            nn.init.zeros_(self.fc2.weight)             # stage 2: exact no-op start

        # Gate: warm-start at sigmoid(-1) ≈ 0.27 (v11 fix — was -3.0 → 0.05).
        # Higher init means:
        #   • sigmoid gradient is 4.5× larger at -1 vs -3 → fc2 gets real gradient
        #     from step 0, allowing it to learn the correction direction before
        #     regularization drives the gate down.
        #   • Combined with scheduled reg ramp (suppress/gate start at 0.02, ramp
        #     to full by reg_ramp_end_steps), opening force can dominate early.
        if self.gate_net is not None:
            nn.init.xavier_uniform_(self.gate_net[0].weight, gain=0.1)
            nn.init.zeros_(self.gate_net[0].bias)
            nn.init.zeros_(self.gate_net[2].weight)
            nn.init.constant_(self.gate_net[2].bias, -1.0)

    @torch.no_grad()
    def update_ema(self, h_clean: torch.Tensor, decay: float = 0.99) -> None:
        """Update the clean-state EMA. h_clean: [B, T, D] raw (pre-LN) states."""
        h_norm = self.layer_norm(h_clean.float())
        batch_mean = h_norm.mean(dim=(0, 1))               # [D]
        self.h_clean_mean.mul_(decay).add_(batch_mean * (1.0 - decay))

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            h: Hidden states [B, T, D]
        Returns:
            h_corrected: [B, T, D]
            alpha:       [B, T, 1]
            delta:       [B, T, D]  norm-clipped correction (before gate)
        """
        h_norm = self.layer_norm(h)                                     # [B, T, D]

        delta = self.fc2(F.gelu(self.fc1(h_norm)))                      # [B, T, D]

        # Adaptive per-token norm clip: correction ≤ max_norm * ‖h‖ per token.
        # max_norm is a fraction (e.g. 0.05 = 5% of input norm), not an absolute
        # value.  This ensures the correction scales with the local embedding
        # magnitude regardless of model or layer, fixing the v8 problem where
        # a fixed cap of 1.0 was 0.45% of Phi-3.5 embedding norm (~55).
        h_norm_mag = h.norm(dim=-1, keepdim=True).detach()              # [B, T, 1]
        delta_norm = delta.norm(dim=-1, keepdim=True)                   # [B, T, 1]
        scale_target = h_norm_mag * self.max_norm                       # [B, T, 1]
        scale = (scale_target / delta_norm.clamp(min=1e-8)).clamp(max=1.0)
        delta = delta * scale

        if not self.use_gate:
            # Gateless: apply correction directly.  alpha=1 tensor for logging.
            alpha = torch.ones(h.shape[0], h.shape[1], 1, dtype=h.dtype, device=h.device)
        elif self.alpha_override is not None:
            # Gate frozen: use a constant alpha (no gradient through gate_net).
            # fc2 receives full gradient from the loss via the correction path.
            alpha = torch.full(
                (h.shape[0], h.shape[1], 1),
                self.alpha_override,
                dtype=h.dtype, device=h.device,
            )
        else:
            if self.use_clean_mean_aug:
                if self.aug_active:
                    diff = (h_norm - self.h_clean_mean).abs()           # [B, T, D]
                else:
                    diff = torch.zeros_like(h_norm)
                gate_in = torch.cat([h_norm, diff], dim=-1)             # [B, T, 2D]
            else:
                gate_in = h_norm                                        # [B, T, D]
            alpha = self.gate_net(gate_in)                              # [B, T, 1]
        return h + alpha * delta, alpha, delta


# ─────────────────────────────────────────────────────────────────────────────
# Model Architecture Adapters
# ─────────────────────────────────────────────────────────────────────────────

_LAYER_ATTR_CANDIDATES = [
    "model.layers",          # Phi-3.5, Llama-3, Mistral, Qwen2.5
    "model.model.layers",    # Some wrapped HF models
    "transformer.h",         # GPT-2, older Falcon
    "model.decoder.layers",  # BART, mBART
    "encoder.layers",        # BERT (encoder-only)
]

_EMBED_ATTR_CANDIDATES = [
    "model.embed_tokens",        # Phi-3.5, Llama-3, Mistral, Qwen2.5
    "model.model.embed_tokens",  # some wrapped HF models
    "transformer.wte",           # GPT-2
    "model.wte",                 # some GPT-2 variants
]


def resolve_layer_list(model: nn.Module) -> nn.ModuleList:
    """Auto-resolve the transformer layer ModuleList from a HuggingFace model."""
    for attr_path in _LAYER_ATTR_CANDIDATES:
        try:
            obj = model
            for attr in attr_path.split("."):
                obj = getattr(obj, attr)
            if isinstance(obj, nn.ModuleList):
                return obj
        except AttributeError:
            continue
    raise ValueError(
        f"Cannot auto-resolve transformer layers for {type(model).__name__}. "
        f"Set StabilizerConfig.layers_attr manually (e.g., layers_attr='model.layers')."
    )


def resolve_embed_module(model: nn.Module) -> nn.Embedding:
    """Auto-resolve the token embedding module from a HuggingFace model."""
    for attr_path in _EMBED_ATTR_CANDIDATES:
        try:
            obj = model
            for attr in attr_path.split("."):
                obj = getattr(obj, attr)
            if isinstance(obj, nn.Embedding):
                return obj
        except AttributeError:
            continue
    raise ValueError(
        f"Cannot auto-resolve embedding module for {type(model).__name__}. "
        f"Set StabilizerConfig.embed_attr manually (e.g., embed_attr='model.embed_tokens')."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Layer Stabilizer System
# ─────────────────────────────────────────────────────────────────────────────

# Embedding layer is stored under key "embed" in self.stabilizers and
# key -1 in clean_states / noisy_states / corrected_states / gate_alphas.
_EMBED_KEY = -1

class MultiLayerStabilizerSystem(nn.Module):
    """
    Manages a two-stage stabilizer system for a frozen LLM.

    Stage 1: DepthWiseStabilizer hooked at embed_tokens output (key "embed" / -1).
    Stage 2: DepthWiseStabilizer hooked after each layer in inject_layers.

    Context managers:
        capture_clean()      — read-only hooks; populates self.clean_states.
        run_correction()     — applying hooks; populates corrected_states, gate_alphas.

    Loss helpers:
        contrastive_loss()   — stage 1 contrastive loss
        stabilization_loss() — stage 2 cosine-sim alignment loss
        gate_sparsity_loss() — max-based gate penalty (both stages)
        gate_suppression_loss() — mean gate on clean steps (both stages)
        adapter_magnitude_loss() — delta magnitude on clean steps (both stages)
        propagation_loss()   — downstream probe alignment

    Persistence:
        system.save(path)
        MultiLayerStabilizerSystem.load(model, path)
        system.register_inference_hooks()
    """

    def __init__(self, model: nn.Module, config: StabilizerConfig):
        super().__init__()
        self.config = config

        # ── Resolve backbone modules ────────────────────────────────────────
        if config.layers_attr:
            obj = model
            for attr in config.layers_attr.split("."):
                obj = getattr(obj, attr)
            self._layers = obj
        else:
            self._layers = resolve_layer_list(model)

        # Stage 1 embedding module
        self._embed_module = None
        if config.embed_layer:
            if config.embed_attr:
                obj = model
                for attr in config.embed_attr.split("."):
                    obj = getattr(obj, attr)
                self._embed_module = obj
            else:
                try:
                    self._embed_module = resolve_embed_module(model)
                except ValueError as e:
                    print(f"[StabilizerSystem] Warning: {e}. Disabling stage 1.")

        # ── Build stabilizer modules ────────────────────────────────────────
        # All modules go into self.stabilizers (ModuleDict) so they are
        # automatically included in .parameters() and state_dict().
        #
        # "embed" → Stage 1 (use_clean_mean_aug=True, aug_active=True from init)
        # "2", "4", ... → Stage 2 (use_clean_mean_aug=False)
        self.stabilizers = nn.ModuleDict()

        if self._embed_module is not None:
            emb_stab = DepthWiseStabilizer(
                hidden_dim=config.hidden_dim,
                bottleneck_dim=config.bottleneck_dim,    # 128
                gate_dim=config.gate_dim,
                dropout=config.dropout,
                max_norm=config.max_norm,
                use_clean_mean_aug=True,
                use_gate=config.use_gate,
            )
            emb_stab.aug_active = True   # always active from step 0 for embeddings
            self.stabilizers["embed"] = emb_stab

        for L in config.inject_layers:
            self.stabilizers[str(L)] = DepthWiseStabilizer(
                hidden_dim=config.hidden_dim,
                bottleneck_dim=config.stage2_bottleneck_dim,   # 32
                gate_dim=config.gate_dim,
                dropout=config.dropout,
                max_norm=config.max_norm,
                use_clean_mean_aug=False,   # hidden state only for stage 2
                use_gate=config.use_gate,
            )

        # ── State populated by context managers ────────────────────────────
        # Key -1 = embedding layer, int keys = transformer layer indices.
        self.clean_states: Dict      = {}
        self.corrected_states: Dict  = {}
        self.gate_alphas: Dict       = {}
        self.noisy_states: Dict      = {}
        self.correction_deltas: Dict = {}
        self.probe_states: Dict      = {}

    # ── Context Managers ────────────────────────────────────────────────────

    @contextmanager
    def capture_clean(self):
        """Register read-only hooks to capture clean hidden states and embeddings."""
        self.clean_states.clear()
        hooks = []

        # Stage 1: capture clean embeddings
        if self._embed_module is not None:
            def _embed_hook(module, args, output):
                self.clean_states[_EMBED_KEY] = output.detach().float().clone()
            hooks.append(self._embed_module.register_forward_hook(_embed_hook))

        # Stage 2 + probe layers: capture clean hidden states
        capture_at = sorted(set(self.config.inject_layers) | set(self.config.probe_layers))
        for L in capture_at:
            def _make_hook(idx: int):
                def _hook(module, args, output):
                    hs = output[0] if isinstance(output, tuple) else output
                    self.clean_states[idx] = hs.detach().float().clone()
                return _hook
            hooks.append(self._layers[L].register_forward_hook(_make_hook(L)))

        try:
            yield self
        finally:
            for h in hooks:
                h.remove()

    @contextmanager
    def run_correction(self, prompt_lengths: Optional[torch.Tensor] = None):
        """
        Register applying hooks that call stabilizers in-place.

        Args:
            prompt_lengths: Optional int tensor [B] of noisy prompt lengths.
                When provided, corrections are applied ONLY to prompt token positions
                (teacher-forcing fix — critical, must apply at both stages).
        """
        self.corrected_states.clear()
        self.gate_alphas.clear()
        self.noisy_states.clear()
        self.correction_deltas.clear()
        self.probe_states.clear()
        hooks = []

        # ── Stage 1: embedding-level correction ────────────────────────────
        if self._embed_module is not None and "embed" in self.stabilizers:
            emb_stab = self.stabilizers["embed"]

            def _make_embed_hook(s: DepthWiseStabilizer):
                def _hook(module, args, output):
                    emb = output.float()

                    # Store noisy embedding BEFORE correction (detached reference)
                    self.noisy_states[_EMBED_KEY] = emb.detach().clone()

                    # Apply stage 1 stabilizer
                    h_corrected, alpha, delta = s(emb)
                    self.corrected_states[_EMBED_KEY] = h_corrected   # in graph
                    self.gate_alphas[_EMBED_KEY]      = alpha
                    self.correction_deltas[_EMBED_KEY] = delta

                    # Prompt-position masking: only correct prompt tokens
                    if prompt_lengths is not None:
                        prompt_mask = torch.zeros(
                            emb.shape[0], emb.shape[1],
                            dtype=torch.bool, device=emb.device,
                        )
                        for b in range(emb.shape[0]):
                            plen = min(int(prompt_lengths[b]), emb.shape[1])
                            prompt_mask[b, :plen] = True
                        h_out = torch.where(
                            prompt_mask.unsqueeze(-1), h_corrected, emb
                        ).to(output.dtype)
                    else:
                        h_out = h_corrected.to(output.dtype)

                    return h_out
                return _hook

            hooks.append(self._embed_module.register_forward_hook(_make_embed_hook(emb_stab)))

        # ── Stage 2: mid-network layer correction ──────────────────────────
        for L in self.config.inject_layers:
            stab = self.stabilizers[str(L)]

            def _make_hook(idx: int, s: DepthWiseStabilizer):
                def _hook(module, args, output):
                    hs = output[0] if isinstance(output, tuple) else output
                    hs_f = hs.float()
                    self.noisy_states[idx] = hs_f

                    h_corrected, alpha, delta = s(hs_f)
                    self.corrected_states[idx] = h_corrected
                    self.gate_alphas[idx]       = alpha
                    self.correction_deltas[idx] = delta

                    if prompt_lengths is not None:
                        prompt_mask = torch.zeros(
                            hs_f.shape[0], hs_f.shape[1],
                            dtype=torch.bool, device=hs_f.device,
                        )
                        for b in range(hs_f.shape[0]):
                            plen = min(int(prompt_lengths[b]), hs_f.shape[1])
                            prompt_mask[b, :plen] = True
                        h_out = torch.where(
                            prompt_mask.unsqueeze(-1), h_corrected, hs_f
                        ).to(hs.dtype)
                    else:
                        h_out = h_corrected.to(hs.dtype)

                    if isinstance(output, tuple):
                        return (h_out,) + output[1:]
                    return h_out
                return _hook

            hooks.append(self._layers[L].register_forward_hook(_make_hook(L, stab)))

        # ── Probe hooks (read-only, downstream of injection window) ────────
        for L in self.config.probe_layers:
            def _make_probe_hook(idx: int):
                def _hook(module, args, output):
                    hs = output[0] if isinstance(output, tuple) else output
                    self.probe_states[idx] = hs.float()   # in graph for L_prop
                return _hook
            hooks.append(self._layers[L].register_forward_hook(_make_probe_hook(L)))

        try:
            yield self
        finally:
            for h in hooks:
                h.remove()

    # ── Loss Components ─────────────────────────────────────────────────────

    def contrastive_loss(
        self,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        L_contrastive (stage 1): push corrected embedding closer to clean
        than the noisy embedding was, by at least margin.

            d_corr  = cosine_distance(h_corrected_emb, h_clean_emb)
            d_noisy = cosine_distance(h_noisy_emb,     h_clean_emb)
            L = mean(relu(d_corr - d_noisy + margin))

        Loss is zero when corrected is already closer to clean than noisy was
        (by margin). This naturally handles non-disruptive perturbations
        (case, whitespace) where noisy ≈ clean: d_noisy ≈ 0, loss ≈ 0.

        Should only be called on noisy steps.
        Requires capture_clean() and run_correction() to have been called.
        """
        if _EMBED_KEY not in self.corrected_states \
                or _EMBED_KEY not in self.clean_states \
                or _EMBED_KEY not in self.noisy_states:
            dummy = next(self.stabilizers.parameters())
            return dummy.sum() * 0.0

        h_corr  = self.corrected_states[_EMBED_KEY]                         # [B, T, D], in graph
        h_clean = self.clean_states[_EMBED_KEY].to(device=h_corr.device,
                                                    dtype=h_corr.dtype)     # [B, T_c, D], detached
        h_noisy = self.noisy_states[_EMBED_KEY].to(device=h_corr.device,
                                                    dtype=h_corr.dtype)     # [B, T, D], detached

        min_T = min(h_corr.shape[1], h_clean.shape[1])
        h_corr  = h_corr[:,  :min_T, :]
        h_clean = h_clean[:, :min_T, :]
        h_noisy = h_noisy[:, :min_T, :]

        # Per-token cosine distances
        d_corr  = 1.0 - F.cosine_similarity(h_corr,  h_clean,          dim=-1)   # [B, T]
        d_noisy = 1.0 - F.cosine_similarity(h_noisy, h_clean.detach(), dim=-1)   # [B, T], no grad

        # Contrastive: penalise only when corrected is not sufficiently closer than noisy
        per_token  = F.relu(d_corr - d_noisy.detach() + self.config.contrastive_margin)  # [B, T]
        per_sample = per_token.mean(dim=-1)                                               # [B]

        if sample_weights is not None:
            w = sample_weights.to(device=per_sample.device, dtype=per_sample.dtype)
            return (per_sample * w).sum() / w.sum().clamp(min=1e-6)
        return per_sample.mean()

    def embed_mse_loss(
        self,
        clean_ids:       Optional[torch.Tensor] = None,   # [B, T_clean]
        noisy_ids:       Optional[torch.Tensor] = None,   # [B, T_noisy]
        clean_attn_mask: Optional[torch.Tensor] = None,   # [B, T_clean]
        noisy_attn_mask: Optional[torch.Tensor] = None,   # [B, T_noisy]
        sample_weights:  Optional[torch.Tensor] = None,
        sample_mask:     Optional[torch.Tensor] = None,   # [B] bool — select subset of batch
    ) -> torch.Tensor:
        """
        L_embed_mse (stage 1, v9): position-wise MSE restricted to positions where
        clean and noisy token IDs actually differ.

        When clean_ids / noisy_ids are provided:
            changed_mask = (clean_ids[:, :T_min] != noisy_ids[:, :T_min])
                           & clean_attn_mask[:, :T_min]   # exclude padding
                           & noisy_attn_mask[:, :T_min]

            L = mean over changed positions of ||h_corrected - h_clean||²

        This gives:
          • Full per-token signal strength (no dilution by ~T unchanged positions).
          • No gradient at unchanged tokens (gate correctly learns to stay closed there).
          • No gradient at padding positions (attention-mask intersection).
          • For tokenization-changing perturbations (typos that split one token into N),
            positions after the split are in changed_mask but semantically misaligned.
            Their gradients are noisy but average out over training; the correctly-aligned
            positions dominate. Accepted imperfection — far better than mean-pooling.

        If no IDs are provided, falls back to position-wise MSE over all T_min positions.
        If the changed mask is empty (e.g. whitespace adds tokens beyond T_min),
        returns 0.

        Should only be called on noisy steps.
        Requires capture_clean() and run_correction() to have been called.
        """
        if _EMBED_KEY not in self.corrected_states or _EMBED_KEY not in self.clean_states:
            dummy = next(self.stabilizers.parameters())
            return dummy.sum() * 0.0

        h_corr  = self.corrected_states[_EMBED_KEY]                        # [B, T, D], in graph
        h_clean = self.clean_states[_EMBED_KEY].to(device=h_corr.device,
                                                   dtype=h_corr.dtype)    # [B, T_c, D], detached
        dev = h_corr.device

        # Filter to a subset of the batch (e.g. noisy-only rows in a mixed batch).
        if sample_mask is not None:
            m = sample_mask.to(dev)
            h_corr          = h_corr[m]
            h_clean         = h_clean[m]
            if clean_ids        is not None: clean_ids        = clean_ids[m]
            if noisy_ids        is not None: noisy_ids        = noisy_ids[m]
            if clean_attn_mask  is not None: clean_attn_mask  = clean_attn_mask[m]
            if noisy_attn_mask  is not None: noisy_attn_mask  = noisy_attn_mask[m]
            if sample_weights   is not None: sample_weights   = sample_weights[m]
            if h_corr.shape[0] == 0:
                dummy = next(self.stabilizers.parameters())
                return dummy.sum() * 0.0

        min_T     = min(h_corr.shape[1], h_clean.shape[1])
        h_corr_t  = h_corr[:,  :min_T, :]                                  # [B, T_min, D]
        h_clean_t = h_clean[:, :min_T, :]                                  # [B, T_min, D]

        # Per-position MSE averaged over hidden dim
        per_pos = ((h_corr_t - h_clean_t) ** 2).mean(dim=-1)              # [B, T_min]

        if clean_ids is not None and noisy_ids is not None:
            c_ids = clean_ids[:, :min_T].to(dev)                           # [B, T_min]
            n_ids = noisy_ids[:, :min_T].to(dev)                           # [B, T_min]
            mask  = (c_ids != n_ids)                                       # [B, T_min]
            # Exclude padding positions from both sides
            if clean_attn_mask is not None:
                mask = mask & clean_attn_mask[:, :min_T].bool().to(dev)
            if noisy_attn_mask is not None:
                mask = mask & noisy_attn_mask[:, :min_T].bool().to(dev)

            if mask.sum() == 0:
                # No changed non-padding positions in this batch
                dummy = next(self.stabilizers.parameters())
                return dummy.sum() * 0.0

            if sample_weights is not None:
                w = sample_weights.to(dev, dtype=h_corr.dtype)             # [B]
                n_per_sample = mask.float().sum(dim=1).clamp(min=1)        # [B]
                per_sample   = (per_pos * mask.float()).sum(dim=1) / n_per_sample  # [B]
                return (per_sample * w).sum() / w.sum().clamp(min=1e-6)
            return per_pos[mask].mean()

        else:
            # Fallback: all T_min positions (no ID mask — used if IDs not passed)
            per_sample = per_pos.mean(dim=-1)                              # [B]
            if sample_weights is not None:
                w = sample_weights.to(dev, dtype=h_corr.dtype)
                return (per_sample * w).sum() / w.sum().clamp(min=1e-6)
            return per_sample.mean()

    def stabilization_loss(
        self,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        L_stab (stage 2): cosine-similarity loss between corrected and clean
        hidden states at inject_layers. Only uses stage 2 layers (not embedding).
        """
        layer_losses = []

        for L in self.config.inject_layers:
            if L not in self.corrected_states or L not in self.clean_states:
                continue
            h_corr  = self.corrected_states[L]
            h_clean = self.clean_states[L].to(device=h_corr.device, dtype=h_corr.dtype)
            min_T   = min(h_corr.shape[1], h_clean.shape[1])
            cos_sim = F.cosine_similarity(
                h_corr[:, :min_T, :], h_clean[:, :min_T, :], dim=-1
            )                                                                 # [B, min_T]
            per_sample = (1.0 - cos_sim).mean(dim=-1)                        # [B]
            if sample_weights is not None:
                w = sample_weights.to(device=per_sample.device, dtype=per_sample.dtype)
                layer_losses.append(
                    (per_sample * w).sum() / w.sum().clamp(min=1e-6)
                )
            else:
                layer_losses.append(per_sample.mean())

        if not layer_losses:
            dummy = next(self.stabilizers.parameters())
            return dummy.sum() * 0.0

        return torch.stack(layer_losses).mean()

    @torch.no_grad()
    def stabilization_loss_per_sample(self) -> Optional[torch.Tensor]:
        """
        Per-sample L_stab: mean cosine-distance loss across inject_layers, no weighting.
        Returns [B] float tensor (detached), or None if states are not populated.
        Used for per-condition logging only — not for gradient computation.
        """
        layer_per_sample = []
        for L in self.config.inject_layers:
            if L not in self.corrected_states or L not in self.clean_states:
                continue
            h_corr  = self.corrected_states[L].detach().float()
            h_clean = self.clean_states[L].to(device=h_corr.device, dtype=h_corr.dtype)
            min_T   = min(h_corr.shape[1], h_clean.shape[1])
            cos_sim = F.cosine_similarity(
                h_corr[:, :min_T, :], h_clean[:, :min_T, :], dim=-1
            )                                                                 # [B, min_T]
            layer_per_sample.append((1.0 - cos_sim).mean(dim=-1))            # [B]
        if not layer_per_sample:
            return None
        return torch.stack(layer_per_sample, dim=0).mean(dim=0)              # [B]

    def gate_sparsity_loss(self) -> torch.Tensor:
        """
        L_gate^max: mean of per-stage maximum gate activations.
        Includes both embed (stage 1) and mid-network (stage 2) gates.
        """
        if not self.gate_alphas:
            dummy = next(self.stabilizers.parameters())
            return dummy.sum() * 0.0
        return torch.stack([a.max() for a in self.gate_alphas.values()]).mean()

    def gate_suppression_loss(
        self,
        sample_mask: Optional[torch.Tensor] = None,   # [B] bool — select subset of batch
    ) -> torch.Tensor:
        """
        L_suppress^clean: mean gate activation across all positions and stages.
        Pass sample_mask to restrict to clean examples in a mixed batch.
        """
        if not self.gate_alphas:
            dummy = next(self.stabilizers.parameters())
            return dummy.sum() * 0.0
        terms = []
        for a in self.gate_alphas.values():             # a: [B, T, 1]
            if sample_mask is not None:
                a = a[sample_mask.to(a.device)]         # [n_clean, T, 1]
            if a.numel() == 0:
                continue
            terms.append(a.mean())
        if not terms:
            dummy = next(self.stabilizers.parameters())
            return dummy.sum() * 0.0
        return torch.stack(terms).mean()

    def adapter_magnitude_loss(
        self,
        sample_mask: Optional[torch.Tensor] = None,   # [B] bool — select subset of batch
    ) -> torch.Tensor:
        """
        L_adapter^clean: mean squared magnitude of correction deltas (both stages).
        Pass sample_mask to restrict to clean examples in a mixed batch.
        """
        if not self.correction_deltas:
            dummy = next(self.stabilizers.parameters())
            return dummy.sum() * 0.0
        terms = []
        for d in self.correction_deltas.values():       # d: [B, T, D]
            if sample_mask is not None:
                d = d[sample_mask.to(d.device)]         # [n_clean, T, D]
            if d.numel() == 0:
                continue
            terms.append(d.pow(2).mean())
        if not terms:
            dummy = next(self.stabilizers.parameters())
            return dummy.sum() * 0.0
        return torch.stack(terms).mean()

    def propagation_loss(self) -> torch.Tensor:
        """
        L_propagation: weighted cosine-similarity loss at probe layers.
        Measures how well stage 2 corrections survive downstream.
        """
        if not self.probe_states or not self.config.probe_layers:
            dummy = next(self.stabilizers.parameters())
            return dummy.sum() * 0.0

        n = len(self.config.probe_layers)
        if self.config.probe_weights and len(self.config.probe_weights) == n:
            raw_weights = list(self.config.probe_weights)
        else:
            raw_weights = [0.5 ** i for i in range(n)]
        total = sum(raw_weights)
        weights = [w / total for w in raw_weights]

        layer_losses = []
        for L, w in zip(self.config.probe_layers, weights):
            if L not in self.probe_states or L not in self.clean_states:
                continue
            h_probe = self.probe_states[L]
            h_clean = self.clean_states[L].to(device=h_probe.device, dtype=h_probe.dtype)
            min_T   = min(h_probe.shape[1], h_clean.shape[1])
            cos_sim = F.cosine_similarity(
                h_probe[:, :min_T, :], h_clean[:, :min_T, :], dim=-1
            )
            layer_losses.append(w * (1.0 - cos_sim).mean())

        if not layer_losses:
            dummy = next(self.stabilizers.parameters())
            return dummy.sum() * 0.0

        return torch.stack(layer_losses).sum()

    # ── Logging Helpers ──────────────────────────────────────────────────────

    def gate_stats(
        self,
        sample_mask: Optional[torch.Tensor] = None,   # [B] bool — select subset of batch
    ) -> Dict[str, float]:
        """
        Return gate statistics split by stage for detailed logging.
        Pass sample_mask to restrict to a subset of the batch (e.g. noisy-only rows).

        Keys returned:
            gate_s1_mean, gate_s1_max  — stage 1 (embedding)
            gate_s2_mean, gate_s2_max  — stage 2 (mid-network average)
            gate_L{N}                  — per-layer stage 2 mean
            gate_mean, gate_max        — aggregate (backward compat)
        """
        if not self.gate_alphas:
            return {}

        def _sel(a: torch.Tensor) -> torch.Tensor:
            """Select rows; a is [B, T, 1]."""
            if sample_mask is None:
                return a
            m = sample_mask.to(a.device)
            return a[m]

        stats = {}

        # Stage 1
        if _EMBED_KEY in self.gate_alphas:
            a1 = _sel(self.gate_alphas[_EMBED_KEY])
            if a1.numel() > 0:
                stats["gate_s1_mean"] = a1.mean().item()
                stats["gate_s1_max"]  = a1.max().item()

        # Stage 2
        s2_alphas = {L: _sel(a) for L, a in self.gate_alphas.items() if L != _EMBED_KEY}
        s2_alphas = {L: a for L, a in s2_alphas.items() if a.numel() > 0}
        if s2_alphas:
            stats.update({f"gate_L{L}": a.mean().item() for L, a in s2_alphas.items()})
            all_s2 = torch.cat([a.reshape(-1) for a in s2_alphas.values()])
            stats["gate_s2_mean"] = all_s2.mean().item()
            stats["gate_s2_max"]  = all_s2.max().item()

        # Aggregate (backward compat)
        all_a_list = []
        if _EMBED_KEY in self.gate_alphas:
            a1 = _sel(self.gate_alphas[_EMBED_KEY])
            if a1.numel() > 0:
                all_a_list.append(a1.reshape(-1))
        for a in s2_alphas.values():
            all_a_list.append(a.reshape(-1))
        if all_a_list:
            all_a = torch.cat(all_a_list)
            stats["gate_mean"] = all_a.mean().item()
            stats["gate_max"]  = all_a.max().item()

        return stats

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.stabilizers.parameters())

    def set_stage1_alpha_override(self, value: Optional[float]) -> None:
        """
        Hard-fix the stage 1 (embedding) gate to a constant alpha.
        When value is not None, gate_net is bypassed and fc2 receives full gradient.
        Set to None to re-enable the learned gate.
        """
        if "embed" in self.stabilizers:
            self.stabilizers["embed"].alpha_override = value

    # ── EMA and Gate-Aug Management ──────────────────────────────────────────

    @torch.no_grad()
    def update_clean_ema(self, decay: float = 0.99) -> None:
        """
        Update EMA references from the most recent clean pass.

        Stage 1 embedding EMA: updated from clean_states[-1] (embeddings).
        Stage 2 EMAs: only updated if the stabilizer has use_clean_mean_aug=True
                      (for v7 stage 2, this is False — update is a no-op).

        Must be called after capture_clean() has populated self.clean_states.
        """
        # Stage 1 embedding EMA
        if _EMBED_KEY in self.clean_states and "embed" in self.stabilizers:
            self.stabilizers["embed"].update_ema(self.clean_states[_EMBED_KEY], decay)

        # Stage 2 (no-op for v7 since use_clean_mean_aug=False, but kept for compat)
        for L in self.config.inject_layers:
            if L not in self.clean_states:
                continue
            stab = self.stabilizers[str(L)]
            if hasattr(stab, "update_ema") and stab.use_clean_mean_aug:
                stab.update_ema(self.clean_states[L], decay)

    def set_gate_aug_active(self, active: bool) -> None:
        """
        Enable or disable gate augmentation on stage 2 stabilizers only.
        Stage 1 embedding gate is always active and is not modified here.
        """
        for key, stab in self.stabilizers.items():
            if key == "embed":
                continue   # embed stage 1: always active, don't touch
            if hasattr(stab, "aug_active"):
                stab.aug_active = active

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str, extra: dict = None) -> str:
        """Save stabilizer weights and config to a .pt checkpoint."""
        payload = {
            "stabilizer_state_dict": self.stabilizers.state_dict(),
            "config": {
                # Core
                "hidden_dim":           self.config.hidden_dim,
                "bottleneck_dim":       self.config.bottleneck_dim,
                "stage2_bottleneck_dim": self.config.stage2_bottleneck_dim,
                "max_norm":             self.config.max_norm,
                "gate_dim":             self.config.gate_dim,
                "dropout":              self.config.dropout,
                "inject_layers":        self.config.inject_layers,
                "embed_layer":          self.config.embed_layer,
                # Loss weights
                "lambda_contrast":      self.config.lambda_contrast,
                "contrastive_margin":   self.config.contrastive_margin,
                "lambda_stab":          self.config.lambda_stab,
                "lambda_acc":           self.config.lambda_acc,
                "lambda_gate":          self.config.lambda_gate,
                "lambda_suppress":      self.config.lambda_suppress,
                "lambda_adapter":       self.config.lambda_adapter,
                "lambda_prop":          self.config.lambda_prop,
                "probe_layers":         self.config.probe_layers,
                "probe_weights":        self.config.probe_weights,
                "ema_decay":            self.config.ema_decay,
                # Deprecated (kept for cross-version compat)
                "lora_rank":            self.config.lora_rank,
                "bottleneck_ratio":     self.config.bottleneck_ratio,
                "lambda_lrd":           self.config.lambda_lrd,
                "lambda_dir":           self.config.lambda_dir,
                "gate_aug":             self.config.gate_aug,
                "gate_aug_warmup_steps": self.config.gate_aug_warmup_steps,
            },
        }
        if extra:
            payload["metadata"] = extra
        torch.save(payload, path)
        return path

    @classmethod
    def load(cls, model: nn.Module, path: str, device=None) -> "MultiLayerStabilizerSystem":
        """Restore a MultiLayerStabilizerSystem from a checkpoint."""
        if device is None:
            device = next(model.parameters()).device
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = ckpt["config"]

        # Backward compat: defaults for fields added in v5/v6/v7
        cfg.setdefault("lora_rank",            0)
        cfg.setdefault("bottleneck_ratio",      4)
        cfg.setdefault("bottleneck_dim",        128)
        cfg.setdefault("max_norm",              1.0)
        cfg.setdefault("embed_layer",           False)   # old checkpoints had no stage 1
        cfg.setdefault("stage2_bottleneck_dim", cfg.get("bottleneck_dim", 128))
        cfg.setdefault("lambda_contrast",       0.0)
        cfg.setdefault("contrastive_margin",    0.1)
        cfg.setdefault("lambda_lrd",            0.0)
        cfg.setdefault("lambda_dir",            0.0)
        cfg.setdefault("gate_aug",              False)
        cfg.setdefault("ema_decay",             0.99)
        cfg.setdefault("gate_aug_warmup_steps", 500)
        cfg.setdefault("probe_layers",          [])
        cfg.setdefault("probe_weights",         [])
        cfg.setdefault("lambda_prop",           0.0)
        cfg.setdefault("lambda_suppress",       0.0)
        cfg.setdefault("lambda_adapter",        0.0)

        config = StabilizerConfig(**cfg)
        system = cls(model, config)
        system.stabilizers.load_state_dict(ckpt["stabilizer_state_dict"])
        system.stabilizers = system.stabilizers.to(device).float()

        # Ensure stage 1 aug is always active on loaded checkpoints
        if "embed" in system.stabilizers:
            system.stabilizers["embed"].aug_active = True

        return system

    def register_inference_hooks(self) -> List:
        """
        Register permanent correction hooks for inference.

        Stage 1 (embed): aug always active.
        Stage 2: aug always False (no clean-mean EMA at inference time).

        Returns list of hook handles — call handle.remove() to detach.
        """
        self.stabilizers.eval()
        # Stage 1: always active
        if "embed" in self.stabilizers:
            self.stabilizers["embed"].aug_active = True

        handles = []

        # Determine backbone dtype for output casting.
        # IMPORTANT: stabilizers are kept in float32 (LayerNorm requires float32
        # inputs on CUDA even if weights are float16). We cast the hook *input*
        # to float32 before calling the stabilizer, then cast the output back to
        # backbone_dtype before returning it to the model.
        try:
            backbone_dtype = next(self._layers[self.config.inject_layers[0]].parameters()).dtype
        except (StopIteration, IndexError):
            backbone_dtype = torch.float32

        # Stage 1: embed hook
        if self._embed_module is not None and "embed" in self.stabilizers:
            stab_inf = self.stabilizers["embed"]  # stays float32 — do NOT cast to backbone_dtype

            def _make_embed_inf_hook(s: DepthWiseStabilizer, bdtype):
                def _hook(module, args, output):
                    with torch.no_grad():
                        h_corrected, _, _ = s(output.float())
                    return h_corrected.to(bdtype)
                return _hook

            handles.append(
                self._embed_module.register_forward_hook(_make_embed_inf_hook(stab_inf, backbone_dtype))
            )

        # Stage 2: layer hooks
        for L in self.config.inject_layers:
            stab_inf = self.stabilizers[str(L)]  # stays float32

            def _make_hook(s: DepthWiseStabilizer, bdtype):
                def _hook(module, args, output):
                    hs = output[0] if isinstance(output, tuple) else output
                    with torch.no_grad():
                        h_corrected, _, _ = s(hs.float())
                    h_out = h_corrected.to(bdtype)
                    if isinstance(output, tuple):
                        return (h_out,) + output[1:]
                    return h_out
                return _hook

            handles.append(self._layers[L].register_forward_hook(_make_hook(stab_inf, backbone_dtype)))

        return handles
