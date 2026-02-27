"""
DWR-Transformer: Full model combining self-attention with MoE FFN blocks.

Architecture per block (design.md Section 3.2):
    x → Self-Attention → Add & Norm → DWRBlock (MoE FFN) → Add & Norm

This module provides:
    - MultiHeadSelfAttention: Standard scaled dot-product attention
    - DWRTransformerBlock: Single transformer layer (attention + MoE FFN)
    - DWRTransformer: Full model (embeddings + stacked blocks + output)

Self-attention is standard and unmodified from the original Transformer.
Only the FFN sub-layer is replaced by the MoE routing mechanism.
This separation is key for Phase 2: attention stays fixed while
expert weights become externalized.

Design doc reference: Sections 2, 3, 11
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from models.dwr_block import DWRBlock


class MultiHeadSelfAttention(nn.Module):
    """
    Standard multi-head self-attention with optional masking.

    Implements Q/K/V projections, scaled dot-product attention,
    and output projection. No modifications for MoE — this is
    vanilla attention as specified in design.md Section 2.1.

    Parameters
    ----------
    d_model : int
        Token representation dimensionality.
    num_heads : int
        Number of attention heads. Must divide d_model evenly.
    dropout : float
        Dropout applied to attention weights and output.
    """

    def __init__(
        self, d_model: int, num_heads: int, dropout: float = 0.1
    ) -> None:
        super().__init__()

        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        )

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.scale = math.sqrt(self.d_head)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Multi-head self-attention forward pass.

        Args:
            x:    (batch, seq_len, d_model)
            mask: Optional attention mask. Supported shapes:
                  - (seq_len, seq_len): broadcast across batch and heads
                  - (batch, 1, seq_len, seq_len): per-sample mask
                  Additive mask convention: 0 = attend, -inf = mask out.

        Returns:
            (batch, seq_len, d_model)
        """
        B, S, D = x.shape
        H = self.num_heads
        DH = self.d_head

        # Project to Q, K, V
        q = self.q_proj(x)  # (B, S, D)
        k = self.k_proj(x)  # (B, S, D)
        v = self.v_proj(x)  # (B, S, D)

        # Reshape to multi-head: (B, H, S, DH)
        q = q.view(B, S, H, DH).transpose(1, 2)  # (B, H, S, DH)
        k = k.view(B, S, H, DH).transpose(1, 2)  # (B, H, S, DH)
        v = v.view(B, S, H, DH).transpose(1, 2)  # (B, H, S, DH)

        # Scaled dot-product attention
        # (B, H, S, DH) @ (B, H, DH, S) → (B, H, S, S)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        if mask is not None:
            attn_weights = attn_weights + mask

        attn_weights = F.softmax(attn_weights, dim=-1)  # (B, H, S, S)
        attn_weights = self.attn_dropout(attn_weights)

        # (B, H, S, S) @ (B, H, S, DH) → (B, H, S, DH)
        attn_output = torch.matmul(attn_weights, v)

        # Merge heads: (B, H, S, DH) → (B, S, H, DH) → (B, S, D)
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(B, S, D)
        )

        output = self.out_proj(attn_output)  # (B, S, D)
        output = self.resid_dropout(output)

        return output


class DWRTransformerBlock(nn.Module):
    """
    Single DWR-Transformer block.

    Structure (design.md Section 3.2):
        x → Self-Attention → Add & Norm → DWRBlock (MoE FFN) → Add & Norm

    Both sub-layers use post-norm (LayerNorm after residual addition),
    matching the "Add & Norm" notation in the design document.

    Parameters
    ----------
    d_model : int
        Token representation dimensionality.
    d_ff : int
        Expert FFN hidden dimensionality.
    num_heads : int
        Number of attention heads.
    num_experts : int
        Number of experts for MoE FFN sub-layer.
    top_k : int
        Experts selected per token.
    dropout : float
        Dropout for attention, FFN, and residual connections.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_heads: int,
        num_experts: int,
        top_k: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Self-attention sub-layer
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.attn_norm = nn.LayerNorm(d_model)

        # MoE FFN sub-layer (includes its own residual + LayerNorm)
        self.moe_ffn = DWRBlock(d_model, d_ff, num_experts, top_k, dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through one transformer block.

        Args:
            x:    (batch, seq_len, d_model)
            mask: Optional attention mask (see MultiHeadSelfAttention)

        Returns:
            output:   (batch, seq_len, d_model)
            aux_loss: scalar — load balancing loss from this block's router
        """
        # Self-attention with residual + post-norm
        attn_output = self.self_attn(x, mask)      # (B, S, D)
        x = self.attn_norm(x + attn_output)         # (B, S, D)

        # MoE FFN sub-layer (DWRBlock handles residual + norm internally)
        x, aux_loss = self.moe_ffn(x)               # (B, S, D)

        return x, aux_loss


class DWRTransformer(nn.Module):
    """
    Full DWR-Transformer model.

    Architecture:
        Token Embedding + Positional Embedding
        → Dropout
        → [DWRTransformerBlock × num_layers]
        → Final LayerNorm
        → Output Projection (d_model → vocab_size)

    Positional encoding uses learned embeddings (standard for modern transformers).
    Output projection does NOT share weights with token embedding to avoid
    introducing unspecified coupling; this can be added later if needed.

    Parameters
    ----------
    vocab_size : int
        Vocabulary size for token embedding and output projection.
    d_model : int
        Token representation dimensionality.
    d_ff : int
        Expert FFN hidden dimensionality.
    num_layers : int
        Number of stacked transformer blocks.
    num_heads : int
        Number of attention heads per block.
    num_experts : int
        Number of experts per MoE layer.
    top_k : int
        Experts selected per token.
    max_seq_len : int
        Maximum sequence length for positional embeddings.
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
        num_experts: int,
        top_k: int = 2,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.embed_dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DWRTransformerBlock(
                d_model, d_ff, num_heads, num_experts, top_k, dropout
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

        # Output projection to vocabulary logits
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initialize parameters with small normal distribution.

        Follows standard transformer initialization practices:
        - Linear weights: N(0, 0.02)
        - Biases: zeros
        - Embeddings: N(0, 0.02)
        """
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
        """
        Generate additive causal (autoregressive) attention mask.

        Returns:
            (seq_len, seq_len) tensor where position (i, j) is:
                0.0    if j <= i  (attend)
                -inf   if j > i  (mask out)
        """
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device), diagonal=1
        )
        mask = mask.masked_fill(mask == 1.0, float("-inf"))
        return mask

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass through DWR-Transformer.

        Args:
            input_ids: (batch, seq_len) — integer token indices
            mask:      Optional attention mask. If None, no masking applied.
                       Use generate_causal_mask() for autoregressive tasks.

        Returns:
            logits:         (batch, seq_len, vocab_size)
            total_aux_loss: scalar — sum of load balancing losses from all layers
        """
        B, S = input_ids.shape

        assert S <= self.max_seq_len, (
            f"Sequence length {S} exceeds max_seq_len {self.max_seq_len}"
        )

        # Token + positional embeddings
        positions = torch.arange(S, device=input_ids.device).unsqueeze(0)  # (1, S)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.embed_dropout(x)  # (B, S, d_model)

        # Pass through transformer blocks, accumulating auxiliary losses
        total_aux_loss = torch.tensor(0.0, device=x.device)

        for block in self.blocks:
            x, aux_loss = block(x, mask)          # (B, S, d_model)
            total_aux_loss = total_aux_loss + aux_loss

        # Final normalization
        x = self.final_norm(x)  # (B, S, d_model)

        # Project to vocabulary
        logits = self.output_proj(x)  # (B, S, vocab_size)

        return logits, total_aux_loss
