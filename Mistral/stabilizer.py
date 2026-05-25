"""
stabilizer.py

Plug-and-Play Depth-Wise Stabilizer — Adjustment A
====================================================
Architecture:
    - A lightweight MLP bottleneck that sits inside a frozen LLM's residual stream.
    - An input-dependent gate network produces a per-token alpha in (0,1).
      This replaces the original scalar gate_logit, which got stuck because
      averaged gradients from clean/noisy batches cancelled each other out.
    - On clean tokens, the gate network learns to output values near 0.
    - On perturbed tokens, it learns to output values near 1, opening the
      correction channel only where it is needed.

Loss (three components):
    1. Robustness Loss   — Cosine similarity between stabilized noisy state and clean reference
    2. Depth-Wise Loss   — Cosine similarity penalty on layer-to-layer jumps
                           (applied ACROSS DEPTH for the SAME token, not across tokens)
    3. Gate Sparsity Loss — L1 on gate outputs to encourage sparse intervention

Injection Point:
    - Attaches as a register_forward_hook on a specific transformer layer.
    - The hook intercepts the layer's OUTPUT hidden states, runs them through
      the stabilizer, and returns the corrected tensor in-place.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthWiseStabilizer(nn.Module):
    """
    Lightweight MLP stabilizer with input-dependent gating.

    The gate is a small 2-layer network that reads the incoming hidden state
    and produces a per-token scalar in (0, 1). This allows the stabilizer to
    dynamically vary its intervention strength token-by-token, rather than
    applying a fixed global correction.

    Args:
        hidden_dim (int):    Hidden state dimension of the target LLM.
                             Phi-3.5-mini = 3072, Llama-3-8B = 4096.
        bottleneck_dim (int): Inner dimension of the correction MLP.
                              Default: hidden_dim // 8 (e.g., 384 for Phi-3.5).
        gate_dim (int):      Inner dimension of the gate network.
                             Kept very small (default: 64) for efficiency.
        dropout (float):     Dropout rate during training.
    """

    def __init__(
        self,
        hidden_dim: int = 3072,
        bottleneck_dim: int = None,
        gate_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        if bottleneck_dim is None:
            bottleneck_dim = hidden_dim // 8  # 384 for Phi-3.5

        # --- Core MLP: noisy h -> correction delta ---
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, hidden_dim, bias=True),
        )

        # --- Input-dependent gate network ---
        # Reads the hidden state and produces a per-token gate value in (0, 1).
        # Small by design: we don't want the gate to be more expressive than
        # the correction MLP itself.
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim, gate_dim, bias=True),
            nn.GELU(),
            nn.Linear(gate_dim, 1, bias=True),
            nn.Sigmoid(),  # Output in (0, 1)
        )

        # --- Layer Norm for stability during training ---
        self.layer_norm = nn.LayerNorm(hidden_dim)

        self._init_weights()

    def _init_weights(self):
        """
        Initialize MLP as near-identity and gate network biased toward zero
        (i.e., pass-through by default) to avoid disrupting the frozen LLM
        at the start of training.
        """
        nn.init.xavier_uniform_(self.mlp[0].weight, gain=0.1)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.zeros_(self.mlp[3].weight)
        nn.init.zeros_(self.mlp[3].bias)

        # Bias the gate output layer strongly negative so sigmoid output starts ~0.05
        nn.init.xavier_uniform_(self.gate_net[0].weight, gain=0.1)
        nn.init.zeros_(self.gate_net[0].bias)
        nn.init.zeros_(self.gate_net[2].weight)
        nn.init.constant_(self.gate_net[2].bias, -3.0)  # sigmoid(-3) ≈ 0.05

    def gate(self, h: torch.Tensor) -> torch.Tensor:
        """
        Compute per-token gate values from the hidden state.

        Args:
            h: Hidden states. Shape: [B, T, D]
        Returns:
            alpha: Per-token gate. Shape: [B, T, 1], values in (0, 1)
        """
        return self.gate_net(self.layer_norm(h))  # [B, T, 1]

    def alpha(self, h: torch.Tensor) -> torch.Tensor:
        """Alias for gate(); exposed for external logging."""
        return self.gate(h)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Hidden states from the transformer layer.
               Shape: [batch_size, seq_len, hidden_dim]

        Returns:
            h_stabilized: Corrected hidden states, same shape as input.
        """
        h_norm = self.layer_norm(h)

        # Correction delta from MLP
        correction = self.mlp(h_norm)           # [B, T, D]

        # Per-token gate: how much correction to apply at each position
        alpha = self.gate_net(h_norm)           # [B, T, 1]

        # Gated residual: alpha near 0 = pass-through, near 1 = full correction
        h_stabilized = h + alpha * correction

        return h_stabilized, alpha              # Return alpha for loss computation


# ==============================================================================
# Loss Functions
# ==============================================================================

def robustness_loss(h_noisy_stabilized: torch.Tensor, h_clean: torch.Tensor) -> torch.Tensor:
    """
    Primary loss: Push the stabilized noisy representation toward the clean one.
    Uses cosine similarity (scale-invariant) rather than MSE.

    MSE is sensitive to hidden state magnitude, which varies wildly across
    batches in a frozen LLM — causing the loss explosions we observed.
    Cosine similarity only cares about directional alignment, which is what
    we actually want: the stabilized representation pointing the same way
    as the clean one, regardless of norm.

    Args:
        h_noisy_stabilized: Output of stabilizer(h_noisy). Shape: [B, T, D]
        h_clean:            Hidden states from clean forward pass (no grad). Shape: [B, T, D]
    """
    cos_sim = F.cosine_similarity(h_noisy_stabilized, h_clean, dim=-1)  # [B, T]
    return (1.0 - cos_sim).mean()


def depth_wise_stability_loss(
    h_current_layer: torch.Tensor,
    h_prev_layer: torch.Tensor
) -> torch.Tensor:
    """
    Depth-wise regularizer: penalize erratic jumps between consecutive layers
    for the SAME token. This is the core of Adjustment A.

    Unlike temporal smoothing (which blurs semantics across tokens), this
    operates in the depth dimension — a token's representation should evolve
    *smoothly* through layers. Typos cause anomalous layer-to-layer spikes
    which this loss directly penalizes.

    Args:
        h_current_layer: Hidden states after stabilizer at layer L. Shape: [B, T, D]
        h_prev_layer:    Hidden states at layer L-1 (no stabilizer). Shape: [B, T, D]

    Returns:
        Scalar loss. Higher when representations spike between layers.
    """
    # Cosine similarity per token: shape [B, T]
    # We want similarity to be HIGH (close to 1), so we penalize (1 - similarity)
    cos_sim = F.cosine_similarity(h_current_layer, h_prev_layer, dim=-1)
    return (1.0 - cos_sim).mean()


def gate_sparsity_loss(alpha: torch.Tensor) -> torch.Tensor:
    """
    L1 sparsity on per-token gate outputs.
    Encourages the gate to stay near zero (no intervention) by default,
    only opening when the robustness loss forces it to.

    Args:
        alpha: Per-token gate values. Shape: [B, T, 1], values in (0, 1).
    """
    return alpha.mean()


def compute_total_loss(
    h_noisy_stabilized: torch.Tensor,
    h_clean: torch.Tensor,
    h_prev_layer: torch.Tensor,
    alpha: torch.Tensor,
    lambda_robustness: float = 1.0,
    lambda_depth: float = 0.5,
    lambda_gate: float = 0.0001,
) -> tuple[torch.Tensor, dict]:
    """
    Combine all three loss components with configurable weights.

    Args:
        h_noisy_stabilized: stabilizer(h_noisy) output
        h_clean:            Clean reference hidden states (detached)
        h_prev_layer:       Layer L-1 hidden states for depth-wise loss
        alpha:              Per-token gate values from the gate network [B, T, 1]
        lambda_*:           Loss weights

    Returns:
        total_loss (Tensor), loss_dict (dict of individual components for logging)
    """
    l_rob  = robustness_loss(h_noisy_stabilized, h_clean)
    l_dep  = depth_wise_stability_loss(h_noisy_stabilized, h_prev_layer)
    l_gate = gate_sparsity_loss(alpha)

    total = (lambda_robustness * l_rob) + (lambda_depth * l_dep) + (lambda_gate * l_gate)

    loss_dict = {
        "loss_total":      total.item(),
        "loss_robustness": l_rob.item(),
        "loss_depth":      l_dep.item(),
        "loss_gate":       l_gate.item(),
        "gate_alpha_mean": alpha.mean().item(),
        "gate_alpha_max":  alpha.max().item(),   # Watch this: should rise on noisy steps
    }

    return total, loss_dict
