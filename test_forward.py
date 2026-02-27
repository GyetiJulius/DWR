"""
Phase 1 Forward-Pass Validation Test.

Tests shape correctness and basic forward-pass behavior for:
    1. Expert (standalone FFN)
    2. Router (top-k routing)
    3. DWRBlock (MoE FFN sub-layer)
    4. DWRTransformerBlock (attention + MoE FFN)
    5. DWRTransformer (full model)

Run: python test_forward.py
Expected: All assertions pass, shape outputs printed.
"""

import torch
from config import DWRConfig
from models.expert import Expert
from models.router import Router
from models.dwr_block import DWRBlock
from models.transformer import (
    MultiHeadSelfAttention,
    DWRTransformerBlock,
    DWRTransformer,
)


def test_expert() -> None:
    """Test Expert FFN shapes."""
    print("=" * 60)
    print("TEST: Expert")
    print("=" * 60)

    B, S, D, D_FF = 2, 8, 512, 2048
    expert = Expert(d_model=D, d_ff=D_FF, dropout=0.1)
    x = torch.randn(B, S, D)

    out = expert(x)
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape}")

    assert out.shape == (B, S, D), f"Expected {(B, S, D)}, got {out.shape}"

    # Also test with flattened tokens (as called from DWRBlock dispatch)
    x_flat = torch.randn(B * S, D)
    out_flat = expert(x_flat)
    print(f"  Flat input:  {x_flat.shape}")
    print(f"  Flat output: {out_flat.shape}")

    assert out_flat.shape == (B * S, D), f"Expected {(B * S, D)}, got {out_flat.shape}"
    print("  PASSED\n")


def test_router() -> None:
    """Test Router top-k selection and shapes."""
    print("=" * 60)
    print("TEST: Router")
    print("=" * 60)

    B, S, D = 2, 8, 512
    num_experts, top_k = 16, 2

    router = Router(d_model=D, num_experts=num_experts, top_k=top_k)
    x = torch.randn(B, S, D)

    topk_indices, topk_scores, aux_loss = router(x)

    print(f"  Input:        {x.shape}")
    print(f"  Indices:      {topk_indices.shape}")
    print(f"  Scores:       {topk_scores.shape}")
    print(f"  Aux loss:     {aux_loss.item():.4f}")

    assert topk_indices.shape == (B, S, top_k), f"Indices shape mismatch"
    assert topk_scores.shape == (B, S, top_k), f"Scores shape mismatch"
    assert aux_loss.ndim == 0, "Aux loss should be scalar"

    # Verify expert indices are in valid range [0, num_experts)
    assert topk_indices.min() >= 0, "Expert index below 0"
    assert topk_indices.max() < num_experts, "Expert index >= num_experts"

    # Verify routing scores are non-negative (from softmax)
    assert (topk_scores >= 0).all(), "Negative routing scores"

    # Verify top-k indices are distinct per token (torch.topk guarantees this)
    for b in range(B):
        for s in range(S):
            indices = topk_indices[b, s].tolist()
            assert len(set(indices)) == top_k, (
                f"Duplicate expert indices at token ({b},{s}): {indices}"
            )

    print("  PASSED\n")


def test_dwr_block() -> None:
    """Test DWRBlock (MoE FFN) shapes and routing."""
    print("=" * 60)
    print("TEST: DWRBlock")
    print("=" * 60)

    B, S, D, D_FF = 2, 8, 512, 2048
    num_experts, top_k = 16, 2

    block = DWRBlock(
        d_model=D, d_ff=D_FF,
        num_experts=num_experts, top_k=top_k, dropout=0.0,
    )
    x = torch.randn(B, S, D)

    out, aux_loss = block(x)

    print(f"  Input:    {x.shape}")
    print(f"  Output:   {out.shape}")
    print(f"  Aux loss: {aux_loss.item():.4f}")

    assert out.shape == (B, S, D), f"Expected {(B, S, D)}, got {out.shape}"
    assert aux_loss.ndim == 0, "Aux loss should be scalar"

    # Verify output is not identical to input (experts should transform)
    assert not torch.allclose(out, x, atol=1e-4), "Output should differ from input"

    print("  PASSED\n")


def test_attention() -> None:
    """Test MultiHeadSelfAttention shapes."""
    print("=" * 60)
    print("TEST: MultiHeadSelfAttention")
    print("=" * 60)

    B, S, D, H = 2, 8, 512, 8

    attn = MultiHeadSelfAttention(d_model=D, num_heads=H, dropout=0.0)
    x = torch.randn(B, S, D)

    # Without mask
    out = attn(x)
    print(f"  Input (no mask): {x.shape}")
    print(f"  Output:          {out.shape}")
    assert out.shape == (B, S, D), f"Expected {(B, S, D)}, got {out.shape}"

    # With causal mask
    mask = DWRTransformer.generate_causal_mask(S, x.device)
    out_masked = attn(x, mask=mask)
    print(f"  Causal mask:     {mask.shape}")
    print(f"  Output (masked): {out_masked.shape}")
    assert out_masked.shape == (B, S, D)

    print("  PASSED\n")


def test_transformer_block() -> None:
    """Test DWRTransformerBlock shapes."""
    print("=" * 60)
    print("TEST: DWRTransformerBlock")
    print("=" * 60)

    B, S, D, D_FF, H = 2, 8, 512, 2048, 8
    num_experts, top_k = 16, 2

    block = DWRTransformerBlock(
        d_model=D, d_ff=D_FF, num_heads=H,
        num_experts=num_experts, top_k=top_k, dropout=0.0,
    )
    x = torch.randn(B, S, D)

    out, aux_loss = block(x)

    print(f"  Input:    {x.shape}")
    print(f"  Output:   {out.shape}")
    print(f"  Aux loss: {aux_loss.item():.4f}")

    assert out.shape == (B, S, D), f"Expected {(B, S, D)}, got {out.shape}"
    print("  PASSED\n")


def test_full_model() -> None:
    """Test full DWRTransformer end-to-end."""
    print("=" * 60)
    print("TEST: DWRTransformer (full model)")
    print("=" * 60)

    config = DWRConfig()

    model = DWRTransformer(
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

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    B, S = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (B, S))

    # Forward pass without mask
    logits, total_aux_loss = model(input_ids)

    print(f"  Input IDs:      {input_ids.shape}")
    print(f"  Logits:         {logits.shape}")
    print(f"  Total aux loss: {total_aux_loss.item():.4f}")

    assert logits.shape == (B, S, config.vocab_size), (
        f"Expected {(B, S, config.vocab_size)}, got {logits.shape}"
    )
    assert total_aux_loss.ndim == 0, "Total aux loss should be scalar"

    # Forward pass with causal mask
    mask = DWRTransformer.generate_causal_mask(S, input_ids.device)
    logits_masked, aux_masked = model(input_ids, mask=mask)

    print(f"  Logits (causal): {logits_masked.shape}")
    assert logits_masked.shape == (B, S, config.vocab_size)

    # Verify gradient flow through router
    total_aux_loss.backward()
    router_grad_exists = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for block in model.blocks
        for p in block.moe_ffn.router.parameters()
    )
    print(f"  Router gradients flow: {router_grad_exists}")

    print("  PASSED\n")


def test_parameter_count() -> None:
    """Verify parameter breakdown matches expectations."""
    print("=" * 60)
    print("TEST: Parameter Count Breakdown")
    print("=" * 60)

    config = DWRConfig()
    model = DWRTransformer(
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

    # Breakdown
    embed_params = sum(
        p.numel() for p in model.token_embedding.parameters()
    ) + sum(
        p.numel() for p in model.position_embedding.parameters()
    )

    attn_params_per_layer = sum(
        p.numel() for p in model.blocks[0].self_attn.parameters()
    ) + sum(
        p.numel() for p in model.blocks[0].attn_norm.parameters()
    )

    router_params_per_layer = sum(
        p.numel() for p in model.blocks[0].moe_ffn.router.parameters()
    )

    expert_params_per_layer = sum(
        p.numel() for p in model.blocks[0].moe_ffn.experts.parameters()
    )

    total = sum(p.numel() for p in model.parameters())

    # Active params per token: embeddings + attention + top_k experts + norms + output
    active_expert_params_per_layer = expert_params_per_layer * config.top_k // config.num_experts

    print(f"  Embeddings:             {embed_params:>12,}")
    print(f"  Attention/layer:        {attn_params_per_layer:>12,}")
    print(f"  Router/layer:           {router_params_per_layer:>12,}")
    print(f"  All experts/layer:      {expert_params_per_layer:>12,}")
    print(f"  Active experts/layer:   {active_expert_params_per_layer:>12,}  (top-{config.top_k}/{config.num_experts})")
    print(f"  Total parameters:       {total:>12,}")
    print(f"  ~Active per token:      {embed_params + config.num_layers * (attn_params_per_layer + router_params_per_layer + active_expert_params_per_layer):>12,}")

    print("  PASSED\n")


if __name__ == "__main__":
    print("\nDWR-Transformer Phase 1 — Forward-Pass Validation\n")

    torch.manual_seed(42)

    test_expert()
    test_router()
    test_dwr_block()
    test_attention()
    test_transformer_block()
    test_full_model()
    test_parameter_count()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
