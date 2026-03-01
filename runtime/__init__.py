"""
runtime package — Dynamic weight retrieval system for DWR-Transformer.

Phase 3: Disk-backed expert loading with GPU LRU caching.
Phase 4: Router-informed predictive prefetching.

Implements design.md Sections 8, 9.1, 9.2, 9.3.
"""

from runtime.expert_store import ExpertStore
from runtime.cache_manager import GPUCacheManager
from runtime.predictor import (
    ExpertPredictor,
    TransitionMatrixPredictor,
    HeuristicPredictor,
    OraclePredictor,
)
from runtime.async_prefetcher import AsyncPrefetcher
from runtime.predictive_streaming_model import (
    PredictiveDWRTransformer,
    build_predictive_model,
)

__all__ = [
    "ExpertStore",
    "GPUCacheManager",
    "ExpertPredictor",
    "TransitionMatrixPredictor",
    "HeuristicPredictor",
    "OraclePredictor",
    "AsyncPrefetcher",
    "PredictiveDWRTransformer",
    "build_predictive_model",
]
