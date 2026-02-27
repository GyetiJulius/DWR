"""
Streaming DWR Block: MoE FFN sub-layer with on-demand expert loading.

Replaces the static DWRBlock's nn.ModuleList of resident experts with
dynamic retrieval through the GPUCacheManager. Only the top-k experts
selected by the router are loaded into GPU memory for each forward pass.

Architecture difference from Phase 1/2 DWRBlock:
    Static:    self.experts = ModuleList([Expert() × 16])  → all in VRAM
    Streaming: self.cache_manager.get_expert(layer, id)    → on-demand from disk

The router and layer norm remain identical — only the expert dispatch changes.
This preserves mathematical equivalence with the static model (bit-for-bit
identical outputs given the same weights).

Design doc reference: Sections 3.2, 9
"""

import torch
import torch.nn as nn
from typing import Tuple, Set

from models.router import Router
from runtime.cache_manager import GPUCacheManager


class StreamingDWRBlock(nn.Module):
    """
    MoE FFN sub-layer with streaming expert retrieval.

    Unlike the static DWRBlock which holds all experts in a ModuleList,
    this block retrieves only the needed experts from the GPUCacheManager
    during forward pass. The router and norm weights are part of this module;
    expert weights live in the ExpertStore on disk.

    Parameters
    ----------
    d_model : int
        Token representation dimensionality.
    num_experts : int
        Total number of experts available (for router).
    top_k : int
        Experts selected per token.
    layer_idx : int
        Index of this block's layer (for cache key construction).
    cache_manager : GPUCacheManager
        Shared cache manager for expert retrieval.
    dropout : float
        Dropout for residual path.
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        top_k: int,
        layer_idx: int,
        cache_manager: GPUCacheManager,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.layer_idx = layer_idx
        self.cache_manager = cache_manager

        # Router is part of the backbone (loaded from backbone.pt)
        self.router = Router(d_model, num_experts, top_k)

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with dynamic expert retrieval.

        Flow:
        1. Router selects top-k experts per token.
        2. Determine unique expert IDs needed for this batch.
        3. Prefetch those experts via cache manager.
        4. Dispatch tokens to experts (only loaded ones).
        5. Weighted aggregation + residual + norm.

        Args:
            x: (batch, seq_len, d_model)

        Returns:
            output: (batch, seq_len, d_model)
        """
        residual = x  # (B, S, D)

        # --- Routing ---
        topk_indices, topk_scores, _ = self.router(x)
        # topk_indices: (B, S, top_k)
        # topk_scores:  (B, S, top_k)

        B, S, D = x.shape

        # Flatten for per-token dispatch
        x_flat = x.reshape(B * S, D)                                  # (T, D)
        topk_indices_flat = topk_indices.reshape(B * S, self.top_k)    # (T, K)
        topk_scores_flat = topk_scores.reshape(B * S, self.top_k)     # (T, K)

        # --- Determine which experts are needed ---
        # Only load the unique experts actually selected by the router.
        # This is the key optimization: instead of iterating all 16 experts,
        # we only touch the ones the router selected.
        unique_expert_ids: Set[int] = set(topk_indices_flat.unique().tolist())

        # --- Prefetch needed experts into GPU cache ---
        prefetch_keys = [
            (self.layer_idx, eid) for eid in unique_expert_ids
        ]
        self.cache_manager.prefetch(prefetch_keys)

        # --- Expert dispatch (only selected experts) ---
        output = torch.zeros_like(x_flat)  # (T, D)

        for expert_id in unique_expert_ids:
            # Retrieve expert from cache (guaranteed hit after prefetch)
            expert = self.cache_manager.get_expert(self.layer_idx, expert_id)

            # Find tokens assigned to this expert
            mask = (topk_indices_flat == expert_id)  # (T, K)
            token_mask = mask.any(dim=-1)             # (T,)
            token_indices = token_mask.nonzero(as_tuple=True)[0]

            if token_indices.numel() == 0:
                continue

            # Run expert forward
            expert_input = x_flat[token_indices]     # (num_assigned, D)
            expert_output = expert(expert_input)     # (num_assigned, D)

            # Extract routing weight for THIS expert only
            # (Same logic as static DWRBlock — mathematically identical)
            expert_weights = (
                topk_scores_flat[token_indices] * mask[token_indices].float()
            ).sum(dim=-1, keepdim=True)  # (num_assigned, 1)

            output[token_indices] += expert_output * expert_weights

        # Reshape back to (B, S, D)
        output = output.reshape(B, S, D)

        # Residual + post-layer-norm
        x = self.layer_norm(residual + self.dropout(output))

        return x
