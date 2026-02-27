"""
Expert module: Standard Transformer FFN used as one expert in DWR-Transformer.

Each expert is a two-layer feed-forward network:
    x → Linear(d_model, d_ff) → GELU → Dropout → Linear(d_ff, d_model)

This is architecturally identical to a standard Transformer FFN block.
In Phase 2+, each expert will be individually serializable for disk-backed
loading. The module boundary is intentionally kept clean to enable this.

Design doc reference: Section 4.1 (Expert Definition)
"""

import torch
import torch.nn as nn


class Expert(nn.Module):
    """
    Single expert FFN block.

    Parameters
    ----------
    d_model : int
        Input and output dimensionality.
    d_ff : int
        Hidden layer dimensionality (expansion factor).
    dropout : float
        Dropout probability applied after activation.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through expert FFN.

        Args:
            x: (..., d_model) — arbitrary leading dims supported,
               typically (batch, seq_len, d_model) or (num_tokens, d_model)
               when called from DWRBlock dispatch.

        Returns:
            (..., d_model) — same shape as input.
        """
        x = self.fc1(x)        # (..., d_ff)
        x = self.activation(x)  # (..., d_ff)
        x = self.dropout(x)    # (..., d_ff)
        x = self.fc2(x)        # (..., d_model)
        return x
