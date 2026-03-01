"""
Dense Transformer Baseline for DWR comparison.

A standard transformer with dense FFN layers (no Mixture of Experts).
Reuses MultiHeadSelfAttention from the DWR model to ensure identical
attention computation. Only the FFN sub-layer differs:

    DWR:   Router → top-k expert dispatch → weighted sum
    Dense: Single FFN (Linear → GELU → Dropout → Linear)

Two baseline configurations:
    Dense-Small (compute-matched):
        Same per-token FLOPs as DWR. d_ff = top_k × expert_d_ff = 4096.
        Answers: "Does MoE routing improve quality over dense at same compute?"

    Dense-Large (param-matched):
        Same total parameter count as DWR (~260M).
        Answers: "Does DWR match dense quality with far less compute per token?"
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer import MultiHeadSelfAttention


class DenseFFN(nn.Module):
    """
    Standard Transformer Feed-Forward Network.

    Linear → GELU → Dropout → Linear

    Identical activation function and structure to Expert but used as
    a single FFN per layer (no routing, no gating).

    Parameters
    ----------
    d_model : int
        Input/output dimensionality.
    d_ff : int
        Hidden dimensionality.
    dropout : float
        Dropout probability.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model) → (batch, seq_len, d_model)"""
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class DenseTransformerBlock(nn.Module):
    """
    Standard Transformer block: Self-Attention + Dense FFN.

    Structure mirrors DWRTransformerBlock exactly, but replaces
    the MoE FFN sub-layer (DWRBlock) with a single DenseFFN.

    Post-norm architecture (matching DWR):
        x → Self-Attention → Add & Norm → Dense FFN → Add & Norm

    Parameters
    ----------
    d_model : int
        Token representation dimensionality.
    d_ff : int
        FFN hidden dimensionality.
    num_heads : int
        Number of attention heads.
    dropout : float
        Dropout for attention, FFN, and residual connections.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.self_attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.attn_norm = nn.LayerNorm(d_model)

        self.ffn = DenseFFN(d_model, d_ff, dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through one dense transformer block.

        Args:
            x:    (batch, seq_len, d_model)
            mask: Optional causal attention mask.

        Returns:
            output: (batch, seq_len, d_model)
        """
        # Self-attention with residual + post-norm
        attn_output = self.self_attn(x, mask)
        x = self.attn_norm(x + attn_output)

        # Dense FFN with residual + post-norm
        ffn_output = self.ffn(x)
        x = self.ffn_norm(x + self.ffn_dropout(ffn_output))

        return x


class DenseTransformer(nn.Module):
    """
    Standard dense Transformer for language modeling.

    Architecture:
        Token Embedding + Positional Embedding
        → Dropout
        → [DenseTransformerBlock × num_layers]
        → Final LayerNorm
        → Output Projection (d_model → vocab_size)

    Mirrors DWRTransformer exactly except FFN sub-layers are dense.
    No auxiliary loss — just cross-entropy.

    Parameters
    ----------
    vocab_size : int
        Vocabulary size.
    d_model : int
        Token representation dimensionality.
    d_ff : int
        FFN hidden dimensionality.
    num_layers : int
        Number of transformer blocks.
    num_heads : int
        Number of attention heads.
    max_seq_len : int
        Maximum sequence length.
    dropout : float
        Global dropout probability.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        d_ff: int,
        num_layers: int,
        num_heads: int,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Embeddings (same as DWRTransformer)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.embed_dropout = nn.Dropout(dropout)

        # Dense transformer blocks
        self.blocks = nn.ModuleList([
            DenseTransformerBlock(d_model, d_ff, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

        # Output projection (no bias, same as DWRTransformer)
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)

        # Initialize weights (same scheme as DWRTransformer)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize parameters with N(0, 0.02): same as DWRTransformer."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @staticmethod
    def generate_causal_mask(
        seq_len: int, device: torch.device
    ) -> torch.Tensor:
        """Generate additive causal mask. Same as DWRTransformer."""
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device), diagonal=1
        )
        mask = mask.masked_fill(mask == 1.0, float("-inf"))
        return mask

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through dense Transformer.

        Args:
            input_ids: (batch, seq_len) integer token indices.
            mask:      Optional causal attention mask.

        Returns:
            logits: (batch, seq_len, vocab_size)

        Note: Returns only logits (no auxiliary loss since no MoE).
        """
        B, S = input_ids.shape

        assert S <= self.max_seq_len, (
            f"Sequence length {S} exceeds max_seq_len {self.max_seq_len}"
        )

        # Token + positional embeddings
        positions = torch.arange(S, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.embed_dropout(x)

        # Pass through transformer blocks (no aux loss)
        for block in self.blocks:
            x = block(x, mask)

        x = self.final_norm(x)
        logits = self.output_proj(x)

        return logits
