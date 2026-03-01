"""
Dense Baseline Configurations for DWR comparison.

Two baselines are defined to isolate different axes of comparison:

Dense-Small (compute-matched):
    Same per-token FLOPs as DWR-Transformer.
    DWR activates top-2 experts with d_ff=2048 → effective FFN width = 4096.
    Dense-Small uses d_ff=4096 with same d_model=512, num_layers=6.
    Question answered: Does MoE routing improve quality at equal compute?

Dense-Large (param-matched):
    Same total parameter count as DWR (~260M).
    Uses d_model=1024, d_ff=4096, num_layers=12 → ~261M params.
    Question answered: Can DWR match dense quality with far less compute?

Both share:
    - Same vocabulary (GPT-2 BPE, 50304)
    - Same max_seq_len (512)
    - Same training recipe (AdamW, warmup+cosine, gradient accumulation)
    - Same dataset (WikiText-103)
    - Same initialization scheme (N(0, 0.02))
"""

from dataclasses import dataclass


@dataclass
class DenseConfig:
    """Configuration for dense Transformer baselines."""

    # --- Model Dimensions ---
    d_model: int = 512
    d_ff: int = 4096
    num_layers: int = 6
    num_heads: int = 8

    # --- Vocabulary and Sequence ---
    vocab_size: int = 50304
    max_seq_len: int = 512

    # --- Regularization ---
    dropout: float = 0.1

    # --- Training Hyperparameters ---
    # Identical to DWR training for fair comparison.
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_epochs: int = 3
    batch_size: int = 8
    grad_accum_steps: int = 4     # Effective batch = 8 × 4 = 32
    grad_clip: float = 1.0
    warmup_steps: int = 500
    eval_interval: int = 500
    log_interval: int = 50
    checkpoint_interval: int = 1

    # --- Paths ---
    checkpoint_dir: str = "checkpoints_dense"
    data_cache_dir: str = "data_cache"

    # --- Model variant name (set by presets) ---
    model_name: str = "dense-small"

    def __post_init__(self) -> None:
        assert self.d_model % self.num_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"
        )


def dense_small_config() -> DenseConfig:
    """
    Dense-Small: Compute-matched baseline.

    Same per-token FLOPs as DWR (top-2 × d_ff=2048 = 4096 effective).
    d_model=512, d_ff=4096, 6 layers, 8 heads.
    """
    return DenseConfig(
        d_model=512,
        d_ff=4096,
        num_layers=6,
        num_heads=8,
        batch_size=8,
        grad_accum_steps=4,
        checkpoint_dir="checkpoints_dense_small",
        model_name="dense-small",
    )


def dense_large_config() -> DenseConfig:
    """
    Dense-Large: Parameter-matched baseline.

    Same total param count as DWR (~260M).
    d_model=1024, d_ff=4096, 12 layers, 16 heads → ~261M params.

    Note: batch_size may need reduction due to larger model.
    grad_accum_steps increased to maintain effective batch=32.
    """
    return DenseConfig(
        d_model=1024,
        d_ff=4096,
        num_layers=12,
        num_heads=16,
        batch_size=4,             # Larger model → smaller micro-batch
        grad_accum_steps=8,       # Maintain effective batch=32
        checkpoint_dir="checkpoints_dense_large",
        model_name="dense-large",
    )
