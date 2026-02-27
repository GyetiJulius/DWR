"""
DWR-Transformer Inference.

Phase 1: STUB — no inference pipeline implemented yet.
Phase 2+ will implement:
    - Autoregressive token generation
    - Expert weight streaming from disk (design.md Section 8)
    - GPU cache manager with LRU eviction (design.md Section 9.2)
    - Weight prefetch engine (design.md Section 9.3)
    - Greedy / top-p / temperature sampling

Inference architecture (design.md Section 9):
    - Expert Store: disk/CPU memory, memory-mapped, async loading
    - GPU Cache Manager: LRU policy, on-demand expert loading
    - Prefetch Strategy: frequency-based caching in v1
"""

import torch
from config import DWRConfig
from train import build_model


def inference() -> None:
    """Inference entry point — Phase 2+ implementation."""
    raise NotImplementedError(
        "Inference pipeline not yet implemented. Phase 1 covers forward-pass only. "
        "See design.md Section 9 for planned runtime architecture."
    )


if __name__ == "__main__":
    inference()
