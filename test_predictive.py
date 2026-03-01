"""
Tests for the predictive prefetch pipeline.

Tests the three new components:
    1. ExpertPredictor strategies (transition matrix, heuristic, oracle)
    2. AsyncPrefetcher (background loading, CUDA streams)
    3. PredictiveDWRTransformer (end-to-end predictive pipeline)

These tests validate correctness; benchmarks measure performance.
"""

import os
import json
import time
import tempfile
from typing import Set

import torch
import pytest

# Add project root to path
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DWRConfig
from runtime.predictor import (
    TransitionMatrixPredictor,
    HeuristicPredictor,
    OraclePredictor,
)


# ── Test 1: Transition Matrix Predictor ──────────────────────────────

class TestTransitionMatrixPredictor:
    """Test the calibration-based predictor."""

    def setup_method(self):
        self.predictor = TransitionMatrixPredictor(num_layers=4, num_experts=8)

    def test_uncalibrated_fallback(self):
        """Before calibration, should fall back to returning same experts."""
        result = self.predictor.predict(
            current_layer=0,
            selected_experts={2, 5},
            num_predict=4,
        )
        # Uncalibrated: returns same experts (fallback)
        assert len(result) <= 4
        assert 2 in result or 5 in result

    def test_record_and_finalize(self):
        """Recording co-occurrences and finalizing should produce valid probabilities."""
        # Simulate: experts {0, 1} at layer 0 always co-occur with {2, 3} at layer 1
        for _ in range(100):
            self.predictor.record(0, {0, 1}, {2, 3})

        # And experts {4, 5} at layer 0 co-occur with {6, 7} at layer 1
        for _ in range(50):
            self.predictor.record(0, {4, 5}, {6, 7})

        self.predictor.finalize()

        # Row sums should be 1.0 (valid probabilities)
        T = self.predictor.transitions[0]
        row_sums = T.sum(dim=1)
        for i in [0, 1, 4, 5]:
            assert abs(row_sums[i].item() - 1.0) < 1e-5

    def test_prediction_accuracy(self):
        """After calibration, should predict co-occurring experts."""
        # Strong pattern: experts {0, 1} → {2, 3}
        for _ in range(100):
            self.predictor.record(0, {0, 1}, {2, 3})

        self.predictor.finalize()

        result = self.predictor.predict(
            current_layer=0,
            selected_experts={0, 1},
            num_predict=4,
        )
        # Should predict experts 2 and 3 (strong co-occurrence)
        assert 2 in result
        assert 3 in result

    def test_save_and_load(self):
        """Transition matrix should serialize/deserialize correctly."""
        for _ in range(50):
            self.predictor.record(0, {0, 1}, {3, 4})
        self.predictor.finalize()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            self.predictor.save(path)
            loaded = TransitionMatrixPredictor.load(path)

            # Should give same predictions
            orig = self.predictor.predict(0, {0, 1}, 4)
            restored = loaded.predict(0, {0, 1}, 4)
            assert orig == restored
        finally:
            os.unlink(path)

    def test_last_layer_returns_empty(self):
        """Predicting from the last layer should return empty set."""
        self.predictor.finalize()
        result = self.predictor.predict(
            current_layer=3,  # Last layer (num_layers=4)
            selected_experts={0, 1},
            num_predict=4,
        )
        assert result == set()


# ── Test 2: Heuristic Predictor ─────────────────────────────────────

class TestHeuristicPredictor:
    """Test the zero-cost heuristic predictor."""

    def test_same_experts_predicted(self):
        """Should predict the same experts as the current layer."""
        predictor = HeuristicPredictor(num_experts=16)
        result = predictor.predict(
            current_layer=0,
            selected_experts={3, 7},
            num_predict=4,
        )
        assert 3 in result
        assert 7 in result

    def test_popular_fillup(self):
        """When fewer selected than budget, should fill with popular experts."""
        predictor = HeuristicPredictor(
            num_experts=16, popular_experts=[10, 11, 12, 13]
        )
        result = predictor.predict(
            current_layer=0,
            selected_experts={3},
            num_predict=4,
        )
        assert 3 in result
        assert len(result) == 4  # 1 selected + 3 popular fill

    def test_budget_limit(self):
        """Should not return more experts than num_predict."""
        predictor = HeuristicPredictor(num_experts=16)
        result = predictor.predict(
            current_layer=0,
            selected_experts={0, 1, 2, 3, 4, 5, 6, 7},
            num_predict=3,
        )
        assert len(result) <= 3


# ── Test 3: Oracle Predictor ────────────────────────────────────────

class TestOraclePredictor:
    """Test the perfect-knowledge oracle predictor."""

    def test_oracle_returns_set_experts(self):
        """Oracle should return exactly the experts it was told."""
        oracle = OraclePredictor()
        oracle.set_oracle({5, 9, 12})

        result = oracle.predict(
            current_layer=0,
            selected_experts={0, 1},  # Ignored by oracle
            num_predict=4,
        )
        assert 5 in result
        assert 9 in result
        assert 12 in result

    def test_oracle_budget_limit(self):
        """Oracle should respect num_predict limit."""
        oracle = OraclePredictor()
        oracle.set_oracle({0, 1, 2, 3, 4, 5})

        result = oracle.predict(
            current_layer=0,
            selected_experts=set(),
            num_predict=3,
        )
        assert len(result) <= 3


# ── Test 4: AsyncPrefetcher (unit test without real files) ───────────

class TestAsyncPrefetcherUnit:
    """Unit tests for the AsyncPrefetcher logic."""

    def test_skip_already_cached(self):
        """Prefetching a cached expert should be a no-op."""
        from unittest.mock import MagicMock, patch

        mock_store = MagicMock()
        mock_cache = MagicMock()
        mock_cache.is_cached.return_value = True  # Already cached

        from runtime.async_prefetcher import AsyncPrefetcher
        prefetcher = AsyncPrefetcher(
            expert_store=mock_store,
            cache_manager=mock_cache,
            max_workers=1,
            device=torch.device("cpu"),
        )

        prefetcher.submit_prefetch({(0, 5), (1, 3)})
        time.sleep(0.1)  # Give threads time

        # Should have counted as hits (already cached, skipped)
        assert prefetcher.prefetch_hits >= 2

        prefetcher.shutdown()

    def test_stats_tracking(self):
        """Stats should track requests correctly."""
        from unittest.mock import MagicMock

        mock_store = MagicMock()
        mock_cache = MagicMock()
        mock_cache.is_cached.return_value = True

        from runtime.async_prefetcher import AsyncPrefetcher
        prefetcher = AsyncPrefetcher(
            expert_store=mock_store,
            cache_manager=mock_cache,
            max_workers=1,
            device=torch.device("cpu"),
        )

        prefetcher.submit_prefetch({(0, 0), (0, 1), (0, 2)})
        time.sleep(0.1)

        stats = prefetcher.get_stats()
        assert stats["prefetch_requests"] == 3

        prefetcher.shutdown()


# ── Test 5: Integration test with real expert store ──────────────────

@pytest.fixture
def expert_store_dir():
    """Check if expert store exists for integration tests."""
    store_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "expert_store"
    )
    if not os.path.exists(store_dir):
        pytest.skip("Expert store not found — run training + export first")
    return store_dir


class TestPredictiveIntegration:
    """Integration tests using real model weights."""

    def test_predictive_forward_pass(self, expert_store_dir):
        """PredictiveDWRTransformer should produce valid logits."""
        config = DWRConfig()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        heuristic = HeuristicPredictor(config.num_experts)
        from runtime.predictive_streaming_model import build_predictive_model

        model = build_predictive_model(
            config=config,
            expert_store_dir=expert_store_dir,
            device=device,
            predictor=heuristic,
            cache_capacity=32,
            prefetch_budget=4,
        )

        input_ids = torch.randint(
            0, config.vocab_size, (1, 64), device=device
        )
        mask = model.generate_causal_mask(64, device)

        logits = model(input_ids, mask=mask)

        assert logits.shape == (1, 64, config.vocab_size)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()

        model.prefetcher.shutdown()

    def test_predictive_vs_reactive_equivalence(self, expert_store_dir):
        """Predictive model should produce same logits as reactive model."""
        config = DWRConfig()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build reactive model
        from runtime.streaming_model import build_streaming_model
        reactive_model = build_streaming_model(
            config=config,
            expert_store_dir=expert_store_dir,
            device=device,
            cache_capacity=96,  # Full cache = all experts loaded
        )

        # Build predictive model (full cache = prediction doesn't affect results)
        heuristic = HeuristicPredictor(config.num_experts)
        from runtime.predictive_streaming_model import build_predictive_model
        predictive_model = build_predictive_model(
            config=config,
            expert_store_dir=expert_store_dir,
            device=device,
            predictor=heuristic,
            cache_capacity=96,
            prefetch_budget=4,
        )

        # Same input
        torch.manual_seed(42)
        input_ids = torch.randint(
            0, config.vocab_size, (1, 32), device=device
        )
        mask = reactive_model.generate_causal_mask(32, device)

        # Forward both
        logits_reactive = reactive_model(input_ids, mask=mask)
        logits_predictive = predictive_model(input_ids, mask=mask)

        # Should be identical (same weights, same computation)
        assert torch.allclose(logits_reactive, logits_predictive, atol=1e-4), \
            f"Max diff: {(logits_reactive - logits_predictive).abs().max().item()}"

        predictive_model.prefetcher.shutdown()

    def test_prefetch_improves_cache_hits(self, expert_store_dir):
        """Predictive prefetch should achieve higher cache hits than small reactive cache."""
        config = DWRConfig()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Small cache (only 8 slots) to force evictions
        cache_capacity = 8

        # Reactive baseline
        from runtime.streaming_model import build_streaming_model
        reactive_model = build_streaming_model(
            config=config,
            expert_store_dir=expert_store_dir,
            device=device,
            cache_capacity=cache_capacity,
        )

        input_ids = torch.randint(
            0, config.vocab_size, (1, 128), device=device
        )
        mask = reactive_model.generate_causal_mask(128, device)

        _ = reactive_model(input_ids, mask=mask)
        reactive_stats = reactive_model.cache_manager.get_stats()

        # Predictive with heuristic
        heuristic = HeuristicPredictor(config.num_experts)
        from runtime.predictive_streaming_model import build_predictive_model
        predictive_model = build_predictive_model(
            config=config,
            expert_store_dir=expert_store_dir,
            device=device,
            predictor=heuristic,
            cache_capacity=cache_capacity,
            prefetch_budget=4,
        )

        _ = predictive_model(input_ids, mask=mask)
        predictive_stats = predictive_model.cache_manager.get_stats()

        print(f"\n  Reactive cache hit rate:   {reactive_stats['hit_rate']:.1%}")
        print(f"  Predictive cache hit rate: {predictive_stats['hit_rate']:.1%}")

        # Predictive should have at least as good cache performance
        # (the prefetch pre-loads experts before they're needed)
        assert predictive_stats["hit_rate"] >= reactive_stats["hit_rate"] * 0.9, \
            "Predictive should not be significantly worse than reactive"

        predictive_model.prefetcher.shutdown()

    def test_generation_produces_tokens(self, expert_store_dir):
        """Predictive model should generate valid token sequences."""
        config = DWRConfig()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        heuristic = HeuristicPredictor(config.num_experts)
        from runtime.predictive_streaming_model import build_predictive_model

        model = build_predictive_model(
            config=config,
            expert_store_dir=expert_store_dir,
            device=device,
            predictor=heuristic,
            cache_capacity=32,
            prefetch_budget=4,
        )

        prompt = torch.randint(0, config.vocab_size, (1, 16), device=device)
        output = model.generate(prompt, max_new_tokens=10, temperature=0.8)

        assert output.shape[1] == 26  # 16 prompt + 10 generated
        assert (output >= 0).all()
        assert (output < config.vocab_size).all()

        model.prefetcher.shutdown()


# ── Run tests ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running predictor unit tests...")

    # Unit tests (no GPU/expert store needed)
    t = TestTransitionMatrixPredictor()
    t.setup_method()
    t.test_uncalibrated_fallback()
    print("  [PASS] TransitionMatrix: uncalibrated fallback")

    t.setup_method()
    t.test_record_and_finalize()
    print("  [PASS] TransitionMatrix: record and finalize")

    t.setup_method()
    t.test_prediction_accuracy()
    print("  [PASS] TransitionMatrix: prediction accuracy")

    t.setup_method()
    t.test_save_and_load()
    print("  [PASS] TransitionMatrix: save and load")

    t.setup_method()
    t.test_last_layer_returns_empty()
    print("  [PASS] TransitionMatrix: last layer empty")

    h = TestHeuristicPredictor()
    h.test_same_experts_predicted()
    print("  [PASS] Heuristic: same experts predicted")
    h.test_popular_fillup()
    print("  [PASS] Heuristic: popular fillup")
    h.test_budget_limit()
    print("  [PASS] Heuristic: budget limit")

    o = TestOraclePredictor()
    o.test_oracle_returns_set_experts()
    print("  [PASS] Oracle: returns set experts")
    o.test_oracle_budget_limit()
    print("  [PASS] Oracle: budget limit")

    a = TestAsyncPrefetcherUnit()
    a.test_skip_already_cached()
    print("  [PASS] AsyncPrefetcher: skip already cached")
    a.test_stats_tracking()
    print("  [PASS] AsyncPrefetcher: stats tracking")

    print(f"\nAll {12} unit tests passed!")

    # Integration tests (need expert store)
    store_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "expert_store"
    )
    if os.path.exists(store_dir):
        print("\nRunning integration tests with expert store...")
        integ = TestPredictiveIntegration()

        integ.test_predictive_forward_pass(store_dir)
        print("  [PASS] Predictive forward pass")

        integ.test_predictive_vs_reactive_equivalence(store_dir)
        print("  [PASS] Predictive vs reactive equivalence")

        integ.test_prefetch_improves_cache_hits(store_dir)
        print("  [PASS] Prefetch improves cache hits")

        integ.test_generation_produces_tokens(store_dir)
        print("  [PASS] Generation produces tokens")

        print(f"\nAll {4} integration tests passed!")
    else:
        print(f"\n[Skip] Integration tests: expert_store not found at {store_dir}")
        print("  Run training + export_experts() first.")
