"""
DWR Block: Mixture-of-Experts FFN sub-layer.

Replaces the dense FFN in a standard transformer block with routed experts:
    Router(x) → Select top-k experts → Dispatch tokens → Weighted aggregation
    → Residual + Dropout → LayerNorm

Token dispatch strategy (Phase 1):
    Iterate over all experts, gather assigned tokens, run expert forward,
    scatter weighted outputs back. This is O(num_experts) in loop overhead
    but straightforward. Phase 2+ can replace this with batched dispatch
    or block-sparse matmuls (MegaBlocks-style).

Routing weight fix (vs. practical_implementation.md reference):
    The reference code incorrectly sums ALL top-k scores per token when
    weighting each expert's output. The correct behavior extracts only the
    score corresponding to the current expert. This implementation fixes that.

Design doc reference: Sections 3.2, 4, 6
"""

import torch
import torch.nn as nn
from typing import Tuple

from models.expert import Expert
from models.router import Router


class DWRBlock(nn.Module):
    """
    MoE FFN sub-layer with top-k routing.

    Each forward pass:
    1. Routes every token to top_k experts via the Router.
    2. Dispatches token representations to selected experts.
    3. Aggregates expert outputs weighted by routing scores.
    4. Applies residual connection + post-layer-norm.

    Parameters
    ----------
    d_model : int
        Token representation dimensionality.
    d_ff : int
        Expert FFN hidden dimensionality.
    num_experts : int
        Total number of experts in this layer.
    top_k : int
        Number of experts selected per token.
    dropout : float
        Dropout probability for residual path.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        top_k: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k

        self.router = Router(d_model, num_experts, top_k)

        # Each expert is a separate nn.Module for future individual serialization.
        # Phase 2 will extract these into independently loadable weight files.
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff, dropout) for _ in range(num_experts)
        ])

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through MoE FFN sub-layer.

        Args:
            x: (batch, seq_len, d_model)

        Returns:
            output:   (batch, seq_len, d_model)
            aux_loss: scalar — load balancing loss from router
        """
        residual = x  # (B, S, D)

        # --- Routing ---
        topk_indices, topk_scores, aux_loss = self.router(x)
        # topk_indices: (B, S, top_k)
        # topk_scores:  (B, S, top_k)

        B, S, D = x.shape

        # Flatten batch and sequence dims for per-token dispatch.
        # After flattening, each row is one token.
        x_flat = x.reshape(B * S, D)                                 # (T, D) where T = B*S
        topk_indices_flat = topk_indices.reshape(B * S, self.top_k)   # (T, K)
        topk_scores_flat = topk_scores.reshape(B * S, self.top_k)    # (T, K)

        # Accumulator for weighted expert outputs
        output = torch.zeros_like(x_flat)  # (T, D)

        # --- Expert dispatch loop ---
        # For each expert, find tokens that selected it and compute expert(x).
        # This loop is O(num_experts) but each iteration only processes
        # the subset of tokens assigned to that expert.
        for expert_id, expert in enumerate(self.experts):

            # Boolean mask: True at position (token, k) where topk_indices == expert_id
            mask = (topk_indices_flat == expert_id)  # (T, K)

            # Which tokens selected this expert in any top-k slot
            token_mask = mask.any(dim=-1)  # (T,)
            token_indices = token_mask.nonzero(as_tuple=True)[0]  # (num_assigned,)

            if token_indices.numel() == 0:
                continue

            # Gather tokens assigned to this expert
            expert_input = x_flat[token_indices]    # (num_assigned, D)
            expert_output = expert(expert_input)    # (num_assigned, D)

            # Extract routing weight for THIS expert only.
            #
            # mask[token_indices] is (num_assigned, K) — True only at positions
            # where topk_indices == expert_id for each selected token.
            # Multiplying with scores zeros out scores from other experts.
            # Summing across K gives the single score for this expert.
            #
            # (Fixes the bug in reference implementation that summed ALL top-k scores.)
            expert_weights = (
                topk_scores_flat[token_indices] * mask[token_indices].float()
            ).sum(dim=-1, keepdim=True)  # (num_assigned, 1)

            # Scatter weighted output back to accumulator
            output[token_indices] += expert_output * expert_weights

        # Reshape back to (B, S, D)
        output = output.reshape(B, S, D)

        # Residual connection + post-layer-norm (design.md Section 3.2: "Add & Norm")
        x = self.layer_norm(residual + self.dropout(output))

        return x, aux_loss
