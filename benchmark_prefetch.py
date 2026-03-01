"""
Prefetch Benchmark: Compare reactive vs predictive inference.

Measures the impact of router-informed predictive prefetching on:
    1. Cache hit rate (reactive baseline vs heuristic vs transition matrix vs oracle)
    2. Throughput (tokens/second)
    3. Latency breakdown (per-layer compute vs I/O)
    4. Prediction accuracy (how often we predict the right experts)

Usage:
    python benchmark_prefetch.py --expert_dir expert_store --predictor_path predictor.json

Output:
    Comparison table + per-layer analysis + JSON results file.

This is the CORE EXPERIMENT for the paper: demonstrating that routing-
informed prediction significantly reduces cache misses and improves
throughput compared to reactive loading.

Expected results (hypothesis):
    Reactive:    ~50-70% cache hit rate (LRU only, no prediction)
    Heuristic:   ~70-85% (same experts + popular fallback)
    Transition:  ~85-95% (learned co-occurrence patterns)
    Oracle:      ~98-100% (perfect prediction upper bound)
"""

import argparse
import json
import os
import time
from typing import Dict, List

import torch

from config import DWRConfig
from data.dataset import build_dataloaders
from runtime.expert_store import ExpertStore
from runtime.cache_manager import GPUCacheManager
from runtime.streaming_model import build_streaming_model
from runtime.predictor import (
    TransitionMatrixPredictor,
    HeuristicPredictor,
    OraclePredictor,
)
from runtime.predictive_streaming_model import build_predictive_model


def benchmark_reactive(
    config: DWRConfig,
    expert_store_dir: str,
    device: torch.device,
    input_ids: torch.Tensor,
    cache_capacity: int = 32,
    num_runs: int = 5,
) -> Dict:
    """Benchmark the reactive (no prediction) baseline."""
    print("\n--- Reactive Baseline (no prediction) ---")

    model = build_streaming_model(
        config=config,
        expert_store_dir=expert_store_dir,
        device=device,
        cache_capacity=cache_capacity,
    )

    # Warmup
    mask = model.generate_causal_mask(input_ids.shape[1], device)
    _ = model(input_ids, mask=mask)

    # Reset stats after warmup
    model.cache_manager.hits = 0
    model.cache_manager.misses = 0
    model.cache_manager.evictions = 0

    # Benchmark
    times = []
    total_tokens = 0
    for _ in range(num_runs):
        model.cache_manager.clear()
        model.cache_manager.hits = 0
        model.cache_manager.misses = 0
        model.cache_manager.evictions = 0

        start = time.perf_counter()
        _ = model(input_ids, mask=mask)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        total_tokens += input_ids.numel()

    stats = model.cache_manager.get_stats()
    avg_time = sum(times) / len(times)
    tokens_per_sec = input_ids.numel() / avg_time

    result = {
        "method": "reactive",
        "cache_hit_rate": stats["hit_rate"],
        "cache_hits": stats["hits"],
        "cache_misses": stats["misses"],
        "evictions": stats["evictions"],
        "avg_latency_ms": avg_time * 1000,
        "throughput_tok_s": tokens_per_sec,
    }

    print(f"  Cache hit rate: {stats['hit_rate']:.1%}")
    print(f"  Avg latency:    {avg_time*1000:.1f}ms")
    print(f"  Throughput:     {tokens_per_sec:.0f} tok/s")

    del model
    return result


def benchmark_predictive(
    config: DWRConfig,
    expert_store_dir: str,
    device: torch.device,
    input_ids: torch.Tensor,
    predictor,
    predictor_name: str,
    cache_capacity: int = 32,
    prefetch_budget: int = 8,
    num_runs: int = 5,
) -> Dict:
    """Benchmark a predictive strategy."""
    print(f"\n--- Predictive: {predictor_name} ---")

    model = build_predictive_model(
        config=config,
        expert_store_dir=expert_store_dir,
        device=device,
        predictor=predictor,
        cache_capacity=cache_capacity,
        prefetch_budget=prefetch_budget,
    )

    # Warmup
    mask = model.generate_causal_mask(input_ids.shape[1], device)
    _ = model(input_ids, mask=mask)

    # Benchmark
    times = []
    for _ in range(num_runs):
        # Reset stats
        model.cache_manager.clear()
        model.cache_manager.hits = 0
        model.cache_manager.misses = 0
        model.cache_manager.evictions = 0
        model.prefetcher.prefetch_requests = 0
        model.prefetcher.prefetch_hits = 0
        model.prefetcher.prefetch_loads = 0
        model.prefetcher.total_prefetch_time = 0.0
        model.layer_timings = {i: [] for i in range(config.num_layers)}

        start = time.perf_counter()
        _ = model(input_ids, mask=mask)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    cache_stats = model.cache_manager.get_stats()
    prefetch_stats = model.prefetcher.get_stats()
    avg_time = sum(times) / len(times)
    tokens_per_sec = input_ids.numel() / avg_time

    # Per-layer timing
    layer_times = {}
    for l, t_list in model.layer_timings.items():
        if t_list:
            layer_times[f"layer_{l}_ms"] = sum(t_list) / len(t_list) * 1000

    result = {
        "method": predictor_name,
        "cache_hit_rate": cache_stats["hit_rate"],
        "cache_hits": cache_stats["hits"],
        "cache_misses": cache_stats["misses"],
        "evictions": cache_stats["evictions"],
        "prefetch_loads": prefetch_stats["prefetch_loads"],
        "prefetch_skip": prefetch_stats["prefetch_hits"],
        "avg_prefetch_ms": prefetch_stats["avg_prefetch_ms"],
        "avg_latency_ms": avg_time * 1000,
        "throughput_tok_s": tokens_per_sec,
        **layer_times,
    }

    print(f"  Cache hit rate:  {cache_stats['hit_rate']:.1%}")
    print(f"  Prefetch loads:  {prefetch_stats['prefetch_loads']}")
    print(f"  Avg prefetch:    {prefetch_stats['avg_prefetch_ms']:.1f}ms")
    print(f"  Avg latency:     {avg_time*1000:.1f}ms")
    print(f"  Throughput:      {tokens_per_sec:.0f} tok/s")

    # Cleanup
    model.prefetcher.shutdown()
    del model
    return result


def benchmark_generation(
    config: DWRConfig,
    expert_store_dir: str,
    device: torch.device,
    prompt_ids: torch.Tensor,
    predictor,
    predictor_name: str,
    max_new_tokens: int = 50,
    cache_capacity: int = 32,
    prefetch_budget: int = 8,
) -> Dict:
    """Benchmark autoregressive generation with prediction."""
    print(f"\n--- Generation: {predictor_name} ---")

    model = build_predictive_model(
        config=config,
        expert_store_dir=expert_store_dir,
        device=device,
        predictor=predictor,
        cache_capacity=cache_capacity,
        prefetch_budget=prefetch_budget,
    )

    start = time.perf_counter()
    output = model.generate(
        prompt_ids, max_new_tokens=max_new_tokens, temperature=0.8
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    generated = output.shape[1] - prompt_ids.shape[1]
    tok_per_sec = generated / elapsed

    cache_stats = model.cache_manager.get_stats()

    result = {
        "method": f"generate-{predictor_name}",
        "generated_tokens": generated,
        "total_time_s": elapsed,
        "tok_per_sec": tok_per_sec,
        "cache_hit_rate": cache_stats["hit_rate"],
    }

    print(f"  Generated:     {generated} tokens")
    print(f"  Time:          {elapsed:.2f}s")
    print(f"  Speed:         {tok_per_sec:.1f} tok/s")
    print(f"  Cache hit rate: {cache_stats['hit_rate']:.1%}")

    model.prefetcher.shutdown()
    del model
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark reactive vs predictive expert prefetching"
    )
    parser.add_argument(
        "--expert_dir", type=str, default="checkpoints/expert_store",
        help="Path to exported expert store directory",
    )
    parser.add_argument(
        "--predictor_path", type=str, default="predictor.json",
        help="Path to calibrated transition matrix",
    )
    parser.add_argument(
        "--cache_capacity", type=int, default=32,
        help="GPU cache capacity (expert slots)",
    )
    parser.add_argument(
        "--prefetch_budget", type=int, default=8,
        help="Max experts to prefetch per prediction",
    )
    parser.add_argument(
        "--seq_len", type=int, default=512,
        help="Sequence length for forward pass benchmark",
    )
    parser.add_argument(
        "--num_runs", type=int, default=5,
        help="Number of benchmark runs per method",
    )
    parser.add_argument(
        "--output", type=str, default="prefetch_benchmark_results.json",
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    config = DWRConfig()
    device = torch.device(args.device)

    print("=" * 60)
    print("PREFETCH BENCHMARK: Reactive vs Predictive")
    print("=" * 60)
    print(f"Config: {config.num_layers} layers × {config.num_experts} experts")
    print(f"Cache capacity: {args.cache_capacity} slots")
    print(f"Prefetch budget: {args.prefetch_budget} experts/prediction")
    print(f"Sequence length: {args.seq_len}")
    print(f"Device: {device}")

    # Create test input
    input_ids = torch.randint(
        0, config.vocab_size, (1, args.seq_len), device=device
    )
    prompt_ids = input_ids[:, :32]  # Short prompt for generation

    results = []

    # 1. Reactive baseline
    reactive_result = benchmark_reactive(
        config, args.expert_dir, device, input_ids,
        cache_capacity=args.cache_capacity,
        num_runs=args.num_runs,
    )
    results.append(reactive_result)

    # 2. Heuristic predictor
    heuristic = HeuristicPredictor(config.num_experts)
    if os.path.exists(args.predictor_path.replace(".json", "_heuristic.json")):
        with open(args.predictor_path.replace(".json", "_heuristic.json")) as f:
            data = json.load(f)
            heuristic.set_popular_experts(data["popular_experts"])

    heuristic_result = benchmark_predictive(
        config, args.expert_dir, device, input_ids,
        predictor=heuristic,
        predictor_name="heuristic",
        cache_capacity=args.cache_capacity,
        prefetch_budget=args.prefetch_budget,
        num_runs=args.num_runs,
    )
    results.append(heuristic_result)

    # 3. Transition matrix predictor (if calibrated)
    if os.path.exists(args.predictor_path):
        tm_predictor = TransitionMatrixPredictor.load(args.predictor_path)
        tm_result = benchmark_predictive(
            config, args.expert_dir, device, input_ids,
            predictor=tm_predictor,
            predictor_name="transition-matrix",
            cache_capacity=args.cache_capacity,
            prefetch_budget=args.prefetch_budget,
            num_runs=args.num_runs,
        )
        results.append(tm_result)
    else:
        print(f"\n[Skip] No calibrated predictor at {args.predictor_path}")
        print("  Run calibrate_predictor.py first.")

    # 4. Generation benchmark (with best predictor)
    best_predictor = (
        tm_predictor if os.path.exists(args.predictor_path) else heuristic
    )
    best_name = (
        "transition-matrix" if os.path.exists(args.predictor_path) else "heuristic"
    )

    gen_reactive = benchmark_generation(
        config, args.expert_dir, device, prompt_ids,
        predictor=HeuristicPredictor(config.num_experts),
        predictor_name="reactive-gen",
        max_new_tokens=50,
        cache_capacity=args.cache_capacity,
        prefetch_budget=0,  # No prefetch = reactive
    )
    results.append(gen_reactive)

    gen_predictive = benchmark_generation(
        config, args.expert_dir, device, prompt_ids,
        predictor=best_predictor,
        predictor_name=f"predictive-gen-{best_name}",
        max_new_tokens=50,
        cache_capacity=args.cache_capacity,
        prefetch_budget=args.prefetch_budget,
    )
    results.append(gen_predictive)

    # Summary table
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Method':<25} {'Hit Rate':>10} {'Latency':>10} {'Throughput':>12}")
    print("-" * 60)
    for r in results:
        method = r.get("method", "?")
        hit_rate = r.get("cache_hit_rate", 0)
        latency = r.get("avg_latency_ms", r.get("total_time_s", 0) * 1000)
        throughput = r.get("throughput_tok_s", r.get("tok_per_sec", 0))
        print(f"{method:<25} {hit_rate:>9.1%} {latency:>9.1f}ms {throughput:>11.0f}")

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
