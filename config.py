"""
DWR-Transformer Configuration.

Phase 1: Model architecture parameters (design.md Section 11).
Phase 2: Training hyperparameters added.

Active parameter count (~50M per token) vs total parameter count (~241M)
is expected for MoE: only top_k / num_experts fraction of expert params
are active per forward pass.
"""

from dataclasses import dataclass


@dataclass
class DWRConfig:
    """Model architecture + training configuration."""

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
    # GPT-2 BPE has 50257 tokens; padded to 50304 (next multiple of 64)
    # for GPU kernel alignment efficiency.
    vocab_size: int = 50304
    max_seq_len: int = 512

    # --- Regularization ---
    dropout: float = 0.1

    # --- Load Balancing (design.md Section 7.2) ---
    # Coefficient λ for auxiliary balance loss: L_total = L_task + λ * L_balance
    balance_loss_coeff: float = 0.01

    # --- Training Hyperparameters (Phase 2) ---
    # Tuned for WikiText-103 (~103M tokens) on single T4 GPU.
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_epochs: int = 3            # 103M tokens × 3 = ~309M token budget
    batch_size: int = 8            # Micro-batch (fits T4 VRAM)
    grad_accum_steps: int = 4      # Effective batch = 8 × 4 = 32
    grad_clip: float = 1.0
    warmup_steps: int = 500        # ~2.5% of epoch (longer warmup for larger data)
    eval_interval: int = 500       # Steps between validation runs
    log_interval: int = 50         # Steps between loss logging
    checkpoint_interval: int = 1   # Epochs between checkpoints

    # --- Paths ---
    checkpoint_dir: str = "checkpoints"
    data_cache_dir: str = "data_cache"

    def __post_init__(self) -> None:
        assert self.d_model % self.num_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"
        )
        assert self.top_k <= self.num_experts, (
            f"top_k ({self.top_k}) must be <= num_experts ({self.num_experts})"
        )
