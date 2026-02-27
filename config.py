"""
DWR-Transformer Phase 1 Configuration.

All architectural parameters for the static MoE prototype.
Values sourced from design.md Section 11 (Prototype Configuration v1).

Active parameter count (~50M per token) vs total parameter count (~241M)
is expected for MoE: only top_k / num_experts fraction of expert params
are active per forward pass.
"""

from dataclasses import dataclass


@dataclass
class DWRConfig:
    """Phase 1 prototype configuration."""

    # --- Model Dimensions ---
    d_model: int = 512
    d_ff: int = 2048  # Per-expert FFN hidden dimension

    # --- Transformer Structure ---
    num_layers: int = 6
    num_heads: int = 8  # d_model / num_heads = 64 per head

    # --- MoE Parameters ---
    num_experts: int = 16
    top_k: int = 2

    # --- Vocabulary and Sequence ---
    vocab_size: int = 32000
    max_seq_len: int = 512

    # --- Regularization ---
    dropout: float = 0.1

    # --- Load Balancing (design.md Section 7.2) ---
    # Coefficient λ for auxiliary balance loss: L_total = L_task + λ * L_balance
    balance_loss_coeff: float = 0.01

    def __post_init__(self) -> None:
        assert self.d_model % self.num_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"
        )
        assert self.top_k <= self.num_experts, (
            f"top_k ({self.top_k}) must be <= num_experts ({self.num_experts})"
        )
