"""
GPU Cache Manager: LRU cache for expert modules in GPU memory.

Keeps the most recently used experts resident on GPU. When a requested
expert is not in cache (cache miss), the least recently used expert
is evicted and the requested expert is loaded from the ExpertStore.

Design doc reference: Section 9.2
    "Keeps active experts in GPU memory.
     Policy: LRU (Least Recently Used) for v1.
     IF expert in GPU cache: Use directly
     ELSE: Evict least recently used expert; Load requested expert"

Cache key: (layer_idx, expert_id) tuple — globally unique identifier.

Memory budget:
    Each expert is ~8.1 MB (from Phase 2 export).
    With capacity=32 slots: 32 × 8.1 ≈ 260 MB GPU memory for cache.
    Total experts: 96 (6 layers × 16 experts) = ~780 MB if all loaded.
    Cache holds ~33% of total experts — meaningful sparsity.

Eviction:
    Pure LRU — the expert that was accessed least recently is evicted.
    The evicted expert module is deleted and its GPU memory freed.
    No write-back needed (experts are read-only during inference).
"""

from collections import OrderedDict
from typing import Tuple, Dict, Optional

import torch

from models.expert import Expert
from runtime.expert_store import ExpertStore


# Cache key type: (layer_index, expert_id)
CacheKey = Tuple[int, int]


class GPUCacheManager:
    """
    LRU cache for Expert modules in GPU memory.

    Parameters
    ----------
    expert_store : ExpertStore
        Backing store for loading experts from disk.
    capacity : int
        Maximum number of experts held in GPU memory simultaneously.
        When exceeded, the least recently used expert is evicted.
    """

    def __init__(
        self,
        expert_store: ExpertStore,
        capacity: int = 32,
    ) -> None:
        self.expert_store = expert_store
        self.capacity = capacity

        # OrderedDict maintains insertion/access order for LRU.
        # Most recently used is at the end; least recently used at the front.
        self._cache: OrderedDict[CacheKey, Expert] = OrderedDict()

        # Statistics
        self.hits: int = 0
        self.misses: int = 0
        self.evictions: int = 0

    def get_expert(self, layer_idx: int, expert_id: int) -> Expert:
        """
        Retrieve an expert, loading from disk if not cached.

        This is the primary interface used by StreamingDWRBlock.
        Implements the cache logic from design.md Section 9.2.

        Args:
            layer_idx: Transformer layer index.
            expert_id: Expert index within layer.

        Returns:
            Expert module on GPU, in eval mode, ready for forward pass.
        """
        key: CacheKey = (layer_idx, expert_id)

        if key in self._cache:
            # Cache hit: move to end (most recently used)
            self._cache.move_to_end(key)
            self.hits += 1
            return self._cache[key]

        # Cache miss: load from disk
        self.misses += 1
        expert = self.expert_store.load_expert(layer_idx, expert_id)

        # Evict if at capacity
        if len(self._cache) >= self.capacity:
            self._evict_lru()

        # Insert into cache (at end = most recently used)
        self._cache[key] = expert

        return expert

    def _evict_lru(self) -> None:
        """
        Evict the least recently used expert from GPU cache.

        The evicted Expert module is explicitly deleted and GPU memory
        is freed. No write-back is needed because experts are read-only
        during inference (weights don't change).
        """
        if not self._cache:
            return

        # Pop from front = least recently used
        evicted_key, evicted_expert = self._cache.popitem(last=False)
        self.evictions += 1

        # Explicitly free GPU memory
        del evicted_expert

    def prefetch(self, keys: list) -> None:
        """
        Pre-load a batch of experts into cache.

        Simple frequency-based prefetch (design.md Section 9.3):
        Load experts that are likely to be needed soon.
        Called before the forward pass when routing decisions are known.

        Args:
            keys: List of (layer_idx, expert_id) tuples to prefetch.
        """
        for layer_idx, expert_id in keys:
            # get_expert handles cache hit/miss/eviction
            self.get_expert(layer_idx, expert_id)

    def is_cached(self, layer_idx: int, expert_id: int) -> bool:
        """Check if an expert is currently in GPU cache."""
        return (layer_idx, expert_id) in self._cache

    def cache_contents(self) -> list:
        """Return list of cached expert keys in LRU order (oldest first)."""
        return list(self._cache.keys())

    def clear(self) -> None:
        """Evict all experts from cache and free GPU memory."""
        self._cache.clear()

    def get_stats(self) -> Dict[str, float]:
        """Return cache performance statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        return {
            "capacity": self.capacity,
            "current_size": len(self._cache),
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": hit_rate,
        }

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"GPUCacheManager(capacity={stats['capacity']}, "
            f"used={stats['current_size']}, "
            f"hit_rate={stats['hit_rate']:.1%})"
        )
