"""
Phase 3 Streaming Inference Validation Test.

Tests the core contribution of DWR-Transformer:
    1. Expert Store: disk-backed expert loading
    2. GPU Cache Manager: LRU eviction policy
    3. Streaming DWR Block: on-demand expert dispatch
    4. Streaming Model: mathematical equivalence with static model
    5. End-to-end generation: autoregressive text generation

Key validation: The streaming model MUST produce numerically identical
outputs to the static model given the same weights and inputs.

Run: python test_phase3.py
Requires: No trained checkpoint (creates a random model, exports it,
          then rebuilds as streaming model).
"""

import os
import shutil
import tempfile

import torch
from config import DWRConfig
from models.transformer import DWRTransformer
from runtime.expert_store import ExpertStore
from runtime.cache_manager import GPUCacheManager
from runtime.streaming_block import StreamingDWRBlock
from runtime.streaming_model import StreamingDWRTransformer, build_streaming_model
from utils.checkpoint import export_experts


# Use a smaller config for fast testing
TEST_CONFIG = DWRConfig(
    d_model=128,
    d_ff=256,
    num_layers=2,
    num_heads=4,
    num_experts=8,
    top_k=2,
    vocab_size=1024,
    max_seq_len=64,
    dropout=0.0,  # Zero dropout for deterministic comparison
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_and_export_static_model(export_dir: str) -> DWRTransformer:
    """Build a random static model and export its experts to disk."""
    model = DWRTransformer(
        vocab_size=TEST_CONFIG.vocab_size,
        d_model=TEST_CONFIG.d_model,
        d_ff=TEST_CONFIG.d_ff,
        num_layers=TEST_CONFIG.num_layers,
        num_heads=TEST_CONFIG.num_heads,
        num_experts=TEST_CONFIG.num_experts,
        top_k=TEST_CONFIG.top_k,
        max_seq_len=TEST_CONFIG.max_seq_len,
        dropout=0.0,
    ).to(DEVICE)
    model.eval()

    export_experts(model, export_dir)
    return model


def test_expert_store() -> None:
    """Test ExpertStore: file validation, loading, statistics."""
    print("=" * 60)
    print("TEST: Expert Store")
    print("=" * 60)

    export_dir = tempfile.mkdtemp(prefix="dwr_test_store_")
    try:
        model = _build_and_export_static_model(export_dir)

        store = ExpertStore(
            store_dir=export_dir,
            d_model=TEST_CONFIG.d_model,
            d_ff=TEST_CONFIG.d_ff,
            num_layers=TEST_CONFIG.num_layers,
            num_experts=TEST_CONFIG.num_experts,
            device=DEVICE,
        )

        # 1. All files should exist (validation passed in constructor)
        num_files = len([f for f in os.listdir(export_dir) if f.startswith("expert_")])
        expected = TEST_CONFIG.num_layers * TEST_CONFIG.num_experts
        assert num_files == expected, f"Expected {expected} expert files, got {num_files}"
        print(f"  Files validated: {num_files} expert files + backbone.pt")

        # 2. Load a specific expert and verify shapes
        expert = store.load_expert(0, 0)
        x = torch.randn(4, TEST_CONFIG.d_model, device=DEVICE)
        out = expert(x)
        assert out.shape == (4, TEST_CONFIG.d_model), f"Expert output shape mismatch: {out.shape}"
        print(f"  Expert(0,0) loaded: output shape {out.shape}")

        # 3. Verify loaded expert matches static model's expert
        static_expert = model.blocks[0].moe_ffn.experts[0]
        static_out = static_expert(x)
        assert torch.allclose(out, static_out, atol=1e-5), (
            f"Expert output mismatch! Max diff: {(out - static_out).abs().max().item():.6e}"
        )
        print(f"  Expert(0,0) output matches static model: max diff = "
              f"{(out - static_out).abs().max().item():.2e}")

        # 4. Check statistics
        stats = store.get_stats()
        assert stats["load_count"] >= 1, "Load count should be >= 1"
        assert stats["total_mb_loaded"] > 0, "Should have loaded > 0 bytes"
        print(f"  Stats: {stats['load_count']} loads, {stats['total_mb_loaded']:.3f} MB")

        # 5. Backbone loading
        backbone = store.load_backbone()
        assert "token_embedding.weight" in backbone, "Backbone missing token_embedding"
        assert "output_proj.weight" in backbone, "Backbone missing output_proj"
        # Backbone should NOT contain expert keys
        expert_keys = [k for k in backbone if ".moe_ffn.experts." in k]
        assert len(expert_keys) == 0, f"Backbone should not contain expert keys: {expert_keys[:3]}"
        print(f"  Backbone loaded: {len(backbone)} keys, no expert weights")

        print("  PASSED\n")
    finally:
        shutil.rmtree(export_dir)


def test_cache_manager() -> None:
    """Test GPUCacheManager: LRU eviction, hit/miss, statistics."""
    print("=" * 60)
    print("TEST: GPU Cache Manager")
    print("=" * 60)

    export_dir = tempfile.mkdtemp(prefix="dwr_test_cache_")
    try:
        _build_and_export_static_model(export_dir)

        store = ExpertStore(
            store_dir=export_dir,
            d_model=TEST_CONFIG.d_model,
            d_ff=TEST_CONFIG.d_ff,
            num_layers=TEST_CONFIG.num_layers,
            num_experts=TEST_CONFIG.num_experts,
            device=DEVICE,
        )

        # Small capacity to test eviction behavior
        cache = GPUCacheManager(expert_store=store, capacity=4)

        # 1. Cache miss → load from disk
        e0 = cache.get_expert(0, 0)
        assert cache.misses == 1 and cache.hits == 0
        print(f"  After get(0,0): misses=1, hits=0, cached={cache.cache_contents()}")

        # 2. Cache hit
        e0_again = cache.get_expert(0, 0)
        assert cache.hits == 1
        print(f"  After get(0,0) again: misses=1, hits=1")

        # 3. Fill to capacity
        cache.get_expert(0, 1)  # miss
        cache.get_expert(0, 2)  # miss
        cache.get_expert(0, 3)  # miss
        assert len(cache.cache_contents()) == 4
        assert cache.evictions == 0
        print(f"  Filled to capacity: {cache.cache_contents()}, evictions=0")

        # 4. Trigger eviction
        cache.get_expert(1, 0)  # miss → should evict LRU
        # After step 2, (0,0) was moved to end, but steps 3-5 appended
        # (0,1), (0,2), (0,3) AFTER (0,0), so final LRU order is:
        #   front (LRU) → (0,0), (0,1), (0,2), (0,3) ← back (MRU)
        # Eviction removes (0,0) from front.
        assert cache.evictions == 1
        assert not cache.is_cached(0, 0), "Expert (0,0) should have been evicted (LRU)"
        assert cache.is_cached(1, 0), "Expert (1,0) should be cached"
        print(f"  After eviction: {cache.cache_contents()}, evictions=1")

        # 5. Prefetch
        cache.prefetch([(0, 4), (0, 5)])
        assert cache.is_cached(0, 4) and cache.is_cached(0, 5)
        print(f"  After prefetch: {cache.cache_contents()}")

        # 6. Statistics
        stats = cache.get_stats()
        assert stats["hit_rate"] > 0, "Hit rate should be > 0"
        print(f"  Final stats: hits={stats['hits']}, misses={stats['misses']}, "
              f"evictions={stats['evictions']}, hit_rate={stats['hit_rate']:.1%}")

        # 7. Clear cache
        cache.clear()
        assert len(cache.cache_contents()) == 0
        print(f"  After clear: cache empty")

        print("  PASSED\n")
    finally:
        shutil.rmtree(export_dir)


def test_streaming_equivalence() -> None:
    """
    Test mathematical equivalence: static model == streaming model.

    This is the CRITICAL test for Phase 3. The streaming model must produce
    numerically identical outputs to the static model. Any difference means
    the expert dispatch logic or weight loading is wrong.
    """
    print("=" * 60)
    print("TEST: Static ↔ Streaming Equivalence")
    print("=" * 60)

    export_dir = tempfile.mkdtemp(prefix="dwr_test_equiv_")
    try:
        static_model = _build_and_export_static_model(export_dir)

        # Build streaming model from same exported weights
        streaming_model = build_streaming_model(
            config=TEST_CONFIG,
            expert_store_dir=export_dir,
            device=DEVICE,
            cache_capacity=TEST_CONFIG.num_layers * TEST_CONFIG.num_experts,  # all fit
        )

        # Create deterministic test input
        torch.manual_seed(42)
        input_ids = torch.randint(0, TEST_CONFIG.vocab_size, (2, 16), device=DEVICE)
        mask = DWRTransformer.generate_causal_mask(16, DEVICE)

        # Static model forward
        with torch.no_grad():
            static_logits, _ = static_model(input_ids, mask=mask)

        # Streaming model forward
        streaming_logits = streaming_model(input_ids, mask=mask)

        # Compare
        max_diff = (static_logits - streaming_logits).abs().max().item()
        mean_diff = (static_logits - streaming_logits).abs().mean().item()

        print(f"  Input shape:      {input_ids.shape}")
        print(f"  Static logits:    {static_logits.shape}")
        print(f"  Streaming logits: {streaming_logits.shape}")
        print(f"  Max abs diff:     {max_diff:.2e}")
        print(f"  Mean abs diff:    {mean_diff:.2e}")

        # Tolerance: fp32 accumulation may differ in dispatch order.
        # Both models route the same way but iterate experts in different order.
        # Static: iterates all 8 experts sequentially.
        # Streaming: iterates only unique selected experts (unordered set).
        # FP32 addition is not associative, so allow small tolerance.
        TOLERANCE = 1e-4
        assert max_diff < TOLERANCE, (
            f"Streaming model differs from static by {max_diff:.2e} (tolerance {TOLERANCE:.0e}). "
            f"Expert dispatch logic or weight loading is incorrect."
        )

        # Verify shapes match exactly
        assert static_logits.shape == streaming_logits.shape, (
            f"Shape mismatch: static {static_logits.shape} vs streaming {streaming_logits.shape}"
        )

        print(f"  Equivalence verified (tolerance {TOLERANCE:.0e})")
        print("  PASSED\n")
    finally:
        shutil.rmtree(export_dir)


def test_streaming_generation() -> None:
    """Test autoregressive generation with streaming model."""
    print("=" * 60)
    print("TEST: Streaming Generation")
    print("=" * 60)

    export_dir = tempfile.mkdtemp(prefix="dwr_test_gen_")
    try:
        _build_and_export_static_model(export_dir)

        model = build_streaming_model(
            config=TEST_CONFIG,
            expert_store_dir=export_dir,
            device=DEVICE,
            cache_capacity=8,
        )

        # Generate from a short prompt
        prompt = torch.tensor([[1, 2, 3, 4, 5]], device=DEVICE)
        max_new_tokens = 20

        output = model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            top_p=0.9,
        )

        # Verify output shape
        expected_len = prompt.shape[1] + max_new_tokens
        assert output.shape == (1, expected_len), (
            f"Expected (1, {expected_len}), got {output.shape}"
        )
        print(f"  Prompt length:     {prompt.shape[1]}")
        print(f"  Generated length:  {output.shape[1] - prompt.shape[1]}")
        print(f"  Total sequence:    {output.shape[1]}")

        # Verify prompt is preserved
        assert torch.equal(output[:, :prompt.shape[1]], prompt), "Prompt should be unchanged"
        print(f"  Prompt preserved:  True")

        # All tokens should be in valid range
        assert output.min() >= 0 and output.max() < TEST_CONFIG.vocab_size, (
            f"Token out of range: [{output.min()}, {output.max()}]"
        )
        print(f"  Token range:       [0, {output.max().item()}] (vocab={TEST_CONFIG.vocab_size})")

        # Cache should have been used during generation
        stats = model.cache_manager.get_stats()
        assert stats["hits"] + stats["misses"] > 0, "Cache should have been used"
        print(f"  Cache stats:       hits={stats['hits']}, misses={stats['misses']}, "
              f"evictions={stats['evictions']}, hit_rate={stats['hit_rate']:.1%}")

        print("  PASSED\n")
    finally:
        shutil.rmtree(export_dir)


def test_cache_eviction_under_pressure() -> None:
    """
    Test GPU cache under extreme pressure: capacity=1.

    With only 1 cache slot, every expert access after the first should
    trigger an eviction. This tests the worst-case eviction path.
    """
    print("=" * 60)
    print("TEST: Cache Eviction Under Pressure (capacity=1)")
    print("=" * 60)

    export_dir = tempfile.mkdtemp(prefix="dwr_test_pressure_")
    try:
        _build_and_export_static_model(export_dir)

        model = build_streaming_model(
            config=TEST_CONFIG,
            expert_store_dir=export_dir,
            device=DEVICE,
            cache_capacity=1,  # Extreme: only 1 slot
        )

        # Run a forward pass — will cause many evictions
        input_ids = torch.randint(0, TEST_CONFIG.vocab_size, (1, 8), device=DEVICE)
        mask = model.generate_causal_mask(8, DEVICE)
        logits = model(input_ids, mask=mask)

        # Should succeed without errors
        assert logits.shape == (1, 8, TEST_CONFIG.vocab_size)

        stats = model.cache_manager.get_stats()
        print(f"  Logits shape:  {logits.shape}")
        print(f"  Cache stats:   hits={stats['hits']}, misses={stats['misses']}, "
              f"evictions={stats['evictions']}")
        print(f"  Hit rate:      {stats['hit_rate']:.1%}")

        # With capacity=1, evictions should be frequent
        assert stats["evictions"] > 0, "Capacity=1 should trigger evictions"
        print(f"  Evictions confirmed: {stats['evictions']}")

        print("  PASSED\n")
    finally:
        shutil.rmtree(export_dir)


def test_memory_efficiency() -> None:
    """
    Verify that streaming model uses less GPU memory than static model.

    With a small cache, the streaming model should hold far fewer expert
    parameters in GPU memory compared to the static model.
    """
    print("=" * 60)
    print("TEST: Memory Efficiency")
    print("=" * 60)

    export_dir = tempfile.mkdtemp(prefix="dwr_test_mem_")
    try:
        static_model = _build_and_export_static_model(export_dir)

        # Count parameters in static model (all experts in GPU)
        static_total = sum(p.numel() for p in static_model.parameters())

        # Count backbone-only parameters (no experts)
        static_expert_params = 0
        for block in static_model.blocks:
            for expert in block.moe_ffn.experts:
                static_expert_params += sum(p.numel() for p in expert.parameters())

        backbone_params = static_total - static_expert_params

        # Streaming model with small cache
        streaming_model = build_streaming_model(
            config=TEST_CONFIG,
            expert_store_dir=export_dir,
            device=DEVICE,
            cache_capacity=4,  # Only 4 of 16 total experts cached
        )

        # Resident parameters = backbone + cached experts
        # At init time, cache is empty, so only backbone is on GPU
        streaming_params = sum(p.numel() for p in streaming_model.parameters())

        print(f"  Static model:    {static_total:,} params (all on GPU)")
        print(f"  Expert params:   {static_expert_params:,} ({static_expert_params/static_total*100:.0f}%)")
        print(f"  Backbone params: {backbone_params:,}")
        print(f"  Streaming model: {streaming_params:,} params (backbone only at init)")

        # The streaming model should have fewer parameters on GPU than static
        assert streaming_params < static_total, (
            f"Streaming model ({streaming_params}) should have fewer GPU params "
            f"than static model ({static_total})"
        )
        reduction = (1 - streaming_params / static_total) * 100
        print(f"  GPU param reduction: {reduction:.0f}%")

        print("  PASSED\n")
    finally:
        shutil.rmtree(export_dir)


def main() -> None:
    """Run all Phase 3 tests."""
    print(f"\nPhase 3: Streaming Inference Tests")
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    test_expert_store()
    test_cache_manager()
    test_streaming_equivalence()
    test_streaming_generation()
    test_cache_eviction_under_pressure()
    test_memory_efficiency()

    print("=" * 60)
    print("ALL PHASE 3 TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
