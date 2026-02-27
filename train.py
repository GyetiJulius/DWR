"""
DWR-Transformer Training Loop.

Phase 1: STUB — no training implemented yet.
Phase 2+ will implement:
    - Standard cross-entropy loss + auxiliary balance loss
    - L_total = L_task + λ * L_balance (design.md Section 7.2)
    - AdamW optimizer with warmup + cosine decay
    - Gradient clipping
    - Checkpointing (individual expert serialization)

Training strategy (design.md Section 10):
    Phase 1: Train normally with all experts loaded (no streaming).
    Phase 2: Deploy streaming only during inference.
"""

import torch
from config import DWRConfig
from models.transformer import DWRTransformer


def build_model(config: DWRConfig) -> DWRTransformer:
    """Construct a DWR-Transformer from config."""
    return DWRTransformer(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        d_ff=config.d_ff,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        num_experts=config.num_experts,
        top_k=config.top_k,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout,
    )


def train() -> None:
    """Training entry point — Phase 2+ implementation."""
    raise NotImplementedError(
        "Training loop not yet implemented. Phase 1 covers forward-pass only. "
        "See design.md Section 10 for planned training strategy."
    )


if __name__ == "__main__":
    train()
