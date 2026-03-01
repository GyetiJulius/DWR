"""
Async Expert Prefetcher: Overlap disk I/O with GPU compute.

The key to hiding storage latency on consumer hardware. While the GPU
computes layer L, this engine loads layer L+1's predicted experts in
the background using:

    Thread pool (disk I/O)  →  Pinned CPU memory (staging)  →  CUDA stream (GPU transfer)

Three-stage pipeline:
    Stage 1: ThreadPoolExecutor reads expert .pt files from SSD to CPU
    Stage 2: Weights staged in pinned (page-locked) CPU memory
    Stage 3: Dedicated CUDA stream copies to GPU asynchronously

The main GPU stream continues computing while the prefetch stream
transfers expert weights. By the time layer L+1 starts, its experts
are already in the GPU cache — turning a cache miss into a cache hit.

Why threading (not multiprocessing):
    - Expert loading is I/O-bound (SSD read), not CPU-bound
    - torch.load() releases the GIL during file I/O
    - Shared memory space with GPUCacheManager — no serialization needed
    - Lower overhead than multiprocessing

Design principles:
    - Non-blocking: prefetch requests return immediately
    - Cancellable: stale predictions don't block fresh ones
    - Idempotent: prefetching an already-cached expert is a no-op
    - Bounded: respects cache capacity, won't over-evict
"""

import time
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Set, Dict, List, Tuple, Optional

import torch

from models.expert import Expert
from runtime.expert_store import ExpertStore
from runtime.cache_manager import GPUCacheManager, CacheKey


class AsyncPrefetcher:
    """
    Asynchronous expert prefetch engine.

    Loads predicted experts from disk in background threads and
    transfers them to GPU via a dedicated CUDA stream, overlapping
    with the main computation stream.

    Parameters
    ----------
    expert_store : ExpertStore
        Disk-backed expert loader.
    cache_manager : GPUCacheManager
        GPU cache to insert prefetched experts into.
    max_workers : int
        Thread pool size for concurrent disk reads.
        Recommended: 2-4 for SSD, 1 for HDD.
    device : torch.device
        Target GPU device for CUDA stream.
    """

    def __init__(
        self,
        expert_store: ExpertStore,
        cache_manager: GPUCacheManager,
        max_workers: int = 4,
        device: torch.device = None,
    ) -> None:
        self.expert_store = expert_store
        self.cache_manager = cache_manager
        self.device = device or torch.device("cuda")

        # Thread pool for I/O-bound disk reads
        self._pool = ThreadPoolExecutor(max_workers=max_workers)

        # Dedicated CUDA stream for async GPU transfers
        # (non-blocking w.r.t. the default/compute stream)
        self._prefetch_stream = (
            torch.cuda.Stream(device=self.device)
            if self.device.type == "cuda"
            else None
        )

        # Track in-flight prefetch requests
        self._pending: Dict[CacheKey, Future] = {}
        self._lock = threading.Lock()

        # Statistics
        self.prefetch_requests: int = 0
        self.prefetch_hits: int = 0      # Already cached, skipped
        self.prefetch_loads: int = 0     # Actually loaded from disk
        self.prefetch_cancels: int = 0   # Cancelled (stale prediction)
        self.total_prefetch_time: float = 0.0

    def submit_prefetch(self, keys: Set[CacheKey]) -> None:
        """
        Submit a batch of experts for background prefetching.

        Non-blocking: returns immediately. Experts are loaded in
        background threads and transferred to GPU cache.

        Skips experts that are:
        - Already in GPU cache (no-op)
        - Already being prefetched (dedup)

        Args:
            keys: Set of (layer_idx, expert_id) tuples to prefetch.
        """
        self.prefetch_requests += len(keys)

        for key in keys:
            layer_idx, expert_id = key

            # Skip if already cached
            if self.cache_manager.is_cached(layer_idx, expert_id):
                self.prefetch_hits += 1
                continue

            # Skip if already in-flight
            with self._lock:
                if key in self._pending:
                    continue
                # Submit to thread pool
                future = self._pool.submit(
                    self._load_and_cache, layer_idx, expert_id
                )
                self._pending[key] = future

    def _load_and_cache(self, layer_idx: int, expert_id: int) -> None:
        """
        Load expert from disk and insert into GPU cache.

        Runs in a background thread. Uses the prefetch CUDA stream
        for the GPU transfer to avoid blocking the compute stream.
        """
        key: CacheKey = (layer_idx, expert_id)
        start = time.perf_counter()

        try:
            # Check again (another thread may have loaded it)
            if self.cache_manager.is_cached(layer_idx, expert_id):
                self.prefetch_hits += 1
                return

            # Stage 1: Load from disk to CPU (I/O bound, GIL released)
            path = self.expert_store._expert_path(layer_idx, expert_id)
            state_dict = torch.load(
                path, map_location="cpu", weights_only=True
            )

            # Create expert module on CPU
            expert = Expert(
                self.expert_store.d_model,
                self.expert_store.d_ff,
                dropout=0.0,
            )
            expert.load_state_dict(state_dict)

            # Stage 2 & 3: Transfer to GPU via prefetch stream
            if self._prefetch_stream is not None:
                with torch.cuda.stream(self._prefetch_stream):
                    expert = expert.to(self.device, non_blocking=True)
                    expert.eval()
                # Synchronize the prefetch stream so data is ready
                self._prefetch_stream.synchronize()
            else:
                expert = expert.to(self.device)
                expert.eval()

            # Insert into cache (thread-safe via GIL for OrderedDict ops)
            # We directly insert to avoid double-loading via get_expert()
            with self._lock:
                if not self.cache_manager.is_cached(layer_idx, expert_id):
                    # Evict if at capacity
                    if len(self.cache_manager._cache) >= self.cache_manager.capacity:
                        self.cache_manager._evict_lru()
                    self.cache_manager._cache[key] = expert
                    self.cache_manager._cache.move_to_end(key)

            self.prefetch_loads += 1

        except Exception as e:
            # Don't crash on prefetch failure — the synchronous path
            # in get_expert() will handle it as a cache miss
            print(f"[AsyncPrefetch] Warning: failed to prefetch {key}: {e}")

        finally:
            elapsed = time.perf_counter() - start
            self.total_prefetch_time += elapsed

            with self._lock:
                self._pending.pop(key, None)

    def wait_pending(self, keys: Optional[Set[CacheKey]] = None) -> None:
        """
        Block until specified prefetch requests complete.

        Call this before a layer needs its experts to ensure they're
        ready. If the prediction was accurate, this should return
        almost instantly (experts already loaded).

        Args:
            keys: Specific keys to wait for. If None, waits for all pending.
        """
        with self._lock:
            if keys is None:
                futures = list(self._pending.values())
            else:
                futures = [
                    self._pending[k] for k in keys if k in self._pending
                ]

        for future in futures:
            future.result()  # Blocks until complete

    def cancel_all(self) -> None:
        """Cancel all pending prefetch requests."""
        with self._lock:
            for key, future in self._pending.items():
                future.cancel()
                self.prefetch_cancels += 1
            self._pending.clear()

    def get_stats(self) -> Dict[str, float]:
        """Return prefetch performance statistics."""
        total = self.prefetch_requests
        return {
            "prefetch_requests": self.prefetch_requests,
            "prefetch_hits": self.prefetch_hits,
            "prefetch_loads": self.prefetch_loads,
            "prefetch_cancels": self.prefetch_cancels,
            "avg_prefetch_ms": (
                (self.total_prefetch_time / max(self.prefetch_loads, 1)) * 1000
            ),
            "prediction_cache_rate": (
                self.prefetch_hits / max(total, 1)
            ),
        }

    def shutdown(self) -> None:
        """Shutdown the thread pool. Call when done with inference."""
        self._pool.shutdown(wait=True)

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"AsyncPrefetcher(workers={self._pool._max_workers}, "
            f"loads={stats['prefetch_loads']}, "
            f"avg={stats['avg_prefetch_ms']:.1f}ms)"
        )
