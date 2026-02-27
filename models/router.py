"""
Router module: Top-k gating for token-level expert selection.

Computes routing logits via a single linear projection, applies softmax,
and selects top-k experts per token. Also computes Switch Transformer-style
load balancing auxiliary loss for training stability.

Mathematical forward pass (design.md Section 6):
    1. r = Linear(x)            — shape: (B, S, num_experts)
    2. g = Softmax(r, dim=-1)   — routing probabilities
    3. indices = TopK(g, k)     — selected expert indices
    4. scores = g[indices]      — routing weights (NOT renormalized after top-k)

Load balancing loss (design.md Section 7.2):
    L_balance = num_experts * Σ_i (f_i * p_i)
    where f_i = fraction of tokens routed to expert i
          p_i = mean routing probability for expert i

Design doc reference: Sections 6, 7
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class Router(nn.Module):
    """
    Top-k expert router with load balancing loss.

    Parameters
    ----------
    d_model : int
        Token representation dimensionality.
    num_experts : int
        Total number of experts available for routing.
    top_k : int
        Number of experts selected per token.
    """

    def __init__(self, d_model: int, num_experts: int, top_k: int = 2) -> None:
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k

        # Single linear projection for routing logits.
        # Bias included to match reference implementation.
        self.gate = nn.Linear(d_model, num_experts)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route each token to its top-k experts.

        Args:
            x: (batch, seq_len, d_model)

        Returns:
            topk_indices: (batch, seq_len, top_k) — selected expert indices
            topk_scores:  (batch, seq_len, top_k) — routing weights from softmax
                          (NOT renormalized; raw softmax values for selected experts)
            aux_loss:     scalar tensor — load balancing auxiliary loss
        """
        logits = self.gate(x)                               # (B, S, num_experts)
        probs = F.softmax(logits, dim=-1)                   # (B, S, num_experts)

        # Select top-k experts per token.
        # torch.topk returns distinct indices (the k largest values).
        topk_scores, topk_indices = torch.topk(
            probs, self.top_k, dim=-1
        )
        # topk_scores:  (B, S, top_k)
        # topk_indices: (B, S, top_k)

        # Compute load balancing loss for training.
        # Even in Phase 1 (no training), we compute this so the API is stable.
        aux_loss = self._load_balance_loss(probs, topk_indices)

        return topk_indices, topk_scores, aux_loss

    def _load_balance_loss(
        self, probs: torch.Tensor, topk_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Switch Transformer load balancing loss.

        Encourages uniform expert utilization by penalizing correlation
        between token dispatch fraction (f_i) and mean routing probability (p_i).

        Gradient flows through p_i only — f_i is treated as a non-differentiable
        scaling factor (computed from discrete top-k selection).

        Args:
            probs:        (B, S, num_experts) — full softmax routing probabilities
            topk_indices: (B, S, top_k) — selected expert indices

        Returns:
            Scalar loss tensor.
        """
        B, S, _ = probs.shape
        num_tokens = B * S

        # Flatten batch and sequence dimensions
        probs_flat = probs.reshape(num_tokens, self.num_experts)      # (B*S, E)
        indices_flat = topk_indices.reshape(num_tokens, self.top_k)   # (B*S, K)

        # f_i: fraction of tokens dispatched to expert i.
        # One-hot encode selected experts, then count total assignments per expert.
        # A token with top_k=2 contributes to 2 experts.
        one_hot = F.one_hot(
            indices_flat, self.num_experts
        )                                                    # (B*S, K, E)
        tokens_per_expert = one_hot.sum(dim=1).float().sum(dim=0)  # (E,)
        f = tokens_per_expert / num_tokens                   # (E,)

        # p_i: mean routing probability assigned to expert i across all tokens.
        # This term carries gradients back to the gate parameters.
        p = probs_flat.mean(dim=0)                           # (E,)

        # L_balance = N_experts * Σ_i (f_i * p_i)
        # Ideal: if both f and p are uniform (1/E), loss = 1.0.
        # Deviation from uniform increases loss.
        aux_loss = self.num_experts * (f * p).sum()

        return aux_loss
