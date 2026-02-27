"""
Streaming DWR-Transformer: Full inference model with dynamic expert loading.

Reconstructs the DWR-Transformer from:
    - backbone.pt: embeddings, attention, norms, router, output projection
    - expert_store/: individual expert .pt files loaded on demand

Architecture is mathematically identical to the static DWRTransformer.
The only difference is HOW expert weights reach the GPU:
    Static:    All experts pre-loaded in nn.ModuleList
    Streaming: Experts fetched from disk via GPUCacheManager on demand

This module handles the non-trivial task of remapping backbone state_dict
keys to the streaming model's structure, since StreamingDWRBlock has a
different module hierarchy than the static DWRBlock.

Design doc reference: Sections 2, 3, 9
"""

import math
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import DWRConfig
from models.transformer import MultiHeadSelfAttention
from runtime.cache_manager import GPUCacheManager
from runtime.expert_store import ExpertStore
from runtime.streaming_block import StreamingDWRBlock


class StreamingTransformerBlock(nn.Module):
    """
    Single transformer block with streaming MoE FFN.

    Structure identical to DWRTransformerBlock:
        x → Self-Attention → Add & Norm → StreamingDWRBlock → output

    Parameters
    ----------
    d_model : int
        Token representation dimensionality.
    num_heads : int
        Number of attention heads.
    num_experts : int
        Total experts per layer.
    top_k : int
        Experts selected per token.
    layer_idx : int
        Layer index (for cache key construction).
    cache_manager : GPUCacheManager
        Shared cache manager for expert loading.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_experts: int,
        top_k: int,
        layer_idx: int,
        cache_manager: GPUCacheManager,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.self_attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.attn_norm = nn.LayerNorm(d_model)

        self.moe_ffn = StreamingDWRBlock(
            d_model=d_model,
            num_experts=num_experts,
            top_k=top_k,
            layer_idx=layer_idx,
            cache_manager=cache_manager,
            dropout=dropout,
        )

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward through attention + streaming MoE FFN.

        Args:
            x:    (batch, seq_len, d_model)
            mask: Optional causal attention mask.

        Returns:
            (batch, seq_len, d_model)
        """
        attn_output = self.self_attn(x, mask)
        x = self.attn_norm(x + attn_output)

        x = self.moe_ffn(x)

        return x


class StreamingDWRTransformer(nn.Module):
    """
    Full DWR-Transformer with streaming expert retrieval.

    Loads backbone weights once (embeddings, attention, norms, routers,
    output projection) and retrieves expert weights on demand from disk.

    Parameters
    ----------
    config : DWRConfig
        Model configuration.
    expert_store : ExpertStore
        Disk-backed expert loader.
    cache_manager : GPUCacheManager
        LRU GPU cache for experts.
    device : torch.device
        Target computation device.
    """

    def __init__(
        self,
        config: DWRConfig,
        expert_store: ExpertStore,
        cache_manager: GPUCacheManager,
        device: torch.device,
    ) -> None:
        super().__init__()

        self.config = config
        self.device = device
        self.cache_manager = cache_manager

        # Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)

        # Transformer blocks with streaming MoE
        self.blocks = nn.ModuleList([
            StreamingTransformerBlock(
                d_model=config.d_model,
                num_heads=config.num_heads,
                num_experts=config.num_experts,
                top_k=config.top_k,
                layer_idx=layer_idx,
                cache_manager=cache_manager,
                dropout=0.0,  # No dropout during inference
            )
            for layer_idx in range(config.num_layers)
        ])

        self.final_norm = nn.LayerNorm(config.d_model)
        self.output_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Load backbone weights
        self._load_backbone(expert_store)

        self.to(device)
        self.eval()

    def _load_backbone(self, expert_store: ExpertStore) -> None:
        """
        Load backbone state dict and map keys to streaming model structure.

        The backbone was exported from the static DWRTransformer, which has
        a different module structure. Key remapping:

        Static model key                              → Streaming model key
        ─────────────────────────────────────────────────────────────────────
        token_embedding.weight                        → token_embedding.weight
        position_embedding.weight                     → position_embedding.weight
        blocks.{L}.self_attn.*                        → blocks.{L}.self_attn.*
        blocks.{L}.attn_norm.*                        → blocks.{L}.attn_norm.*
        blocks.{L}.moe_ffn.router.*                   → blocks.{L}.moe_ffn.router.*
        blocks.{L}.moe_ffn.layer_norm.*               → blocks.{L}.moe_ffn.layer_norm.*
        blocks.{L}.moe_ffn.experts.{I}.*              → (excluded — in expert files)
        final_norm.*                                  → final_norm.*
        output_proj.*                                 → output_proj.*
        """
        backbone_state = expert_store.load_backbone()

        # Filter out any expert keys that might have leaked into backbone
        # (shouldn't happen, but defensive)
        filtered = {}
        for key, value in backbone_state.items():
            if ".moe_ffn.experts." in key:
                continue  # Skip expert weights
            filtered[key] = value

        # Load with strict=False because the streaming model doesn't have
        # expert submodules in its state dict
        missing, unexpected = self.load_state_dict(filtered, strict=False)

        # Verify: missing keys should only be expert-related (which we load on demand)
        # In streaming model there are no expert submodules, so nothing should be missing
        if unexpected:
            print(f"[StreamingModel] Warning: {len(unexpected)} unexpected keys in backbone")

        print(f"[StreamingModel] Backbone loaded: {len(filtered)} parameters")

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

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with streaming expert retrieval.

        Args:
            input_ids: (batch, seq_len) integer token indices.
            mask:      Optional causal attention mask.

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        B, S = input_ids.shape

        positions = torch.arange(S, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        # (B, S, d_model)

        for block in self.blocks:
            x = block(x, mask)

        x = self.final_norm(x)
        logits = self.output_proj(x)  # (B, S, vocab_size)

        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """
        Autoregressive token generation with streaming experts.

        Uses nucleus (top-p) sampling with temperature.

        Args:
            input_ids:      (1, prompt_len) — single sequence prompt.
            max_new_tokens: Maximum tokens to generate.
            temperature:    Sampling temperature (lower = more deterministic).
            top_p:          Nucleus sampling threshold.

        Returns:
            (1, prompt_len + generated_len) — full sequence including prompt.
        """
        self.eval()

        for _ in range(max_new_tokens):
            # Truncate to max_seq_len if needed
            ctx = input_ids[:, -self.config.max_seq_len:]
            S = ctx.shape[1]

            # Causal mask for current context
            mask = self.generate_causal_mask(S, ctx.device)

            # Forward pass (only need logits at last position)
            logits = self.forward(ctx, mask=mask)       # (1, S, V)
            next_logits = logits[:, -1, :]              # (1, V)

            # Temperature scaling
            next_logits = next_logits / temperature

            # Nucleus (top-p) sampling
            sorted_logits, sorted_indices = torch.sort(
                next_logits, descending=True
            )
            probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(probs, dim=-1)

            # Remove tokens with cumulative prob above threshold
            # Keep at least one token
            sorted_mask = cumulative_probs - probs > top_p
            sorted_logits[sorted_mask] = float("-inf")

            # Sample from filtered distribution
            probs = F.softmax(sorted_logits, dim=-1)
            next_token_sorted = torch.multinomial(probs, num_samples=1)
            next_token = sorted_indices.gather(-1, next_token_sorted)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


def build_streaming_model(
    config: DWRConfig,
    expert_store_dir: str,
    device: torch.device,
    cache_capacity: int = 32,
) -> StreamingDWRTransformer:
    """
    Build a streaming DWR-Transformer from exported expert store.

    Convenience function that wires up ExpertStore → GPUCacheManager →
    StreamingDWRTransformer.

    Args:
        config:           Model configuration.
        expert_store_dir: Path to directory with exported expert files.
        device:           Target GPU device.
        cache_capacity:   Number of expert slots in GPU cache.

    Returns:
        StreamingDWRTransformer ready for inference.
    """
    expert_store = ExpertStore(
        store_dir=expert_store_dir,
        d_model=config.d_model,
        d_ff=config.d_ff,
        num_layers=config.num_layers,
        num_experts=config.num_experts,
        device=device,
    )

    cache_manager = GPUCacheManager(
        expert_store=expert_store,
        capacity=cache_capacity,
    )

    model = StreamingDWRTransformer(
        config=config,
        expert_store=expert_store,
        cache_manager=cache_manager,
        device=device,
    )

    return model
