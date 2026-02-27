"""
DWR-Transformer Streaming Inference (Phase 3).

Demonstrates dynamic weight retrieval:
    - Loads backbone once (~222 MB)
    - Retrieves expert weights from disk on demand (~8.1 MB each)
    - Uses LRU GPU cache to keep frequently used experts resident
    - Generates text autoregressively with top-p sampling

Runtime architecture (design.md Section 9):
    ExpertStore (disk) → GPUCacheManager (LRU) → StreamingDWRBlock → output

Usage:
    python inference.py                              # default prompt
    python inference.py --prompt "The meaning of"    # custom prompt
    python inference.py --cache-capacity 8           # small cache (more evictions)
"""

import argparse
import time

import tiktoken
import torch

from config import DWRConfig
from runtime.streaming_model import build_streaming_model


def inference(
    prompt: str = "The meaning of life is",
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.9,
    cache_capacity: int = 96,
    expert_store_dir: str = "checkpoints/expert_store",
) -> None:
    """
    Run streaming inference with dynamic expert loading.

    Args:
        prompt:           Text prompt to complete.
        max_new_tokens:   Number of tokens to generate.
        temperature:      Sampling temperature.
        top_p:            Nucleus sampling threshold.
        cache_capacity:   Max experts in GPU cache.
        expert_store_dir: Path to exported expert files.
    """
    config = DWRConfig()

    # --- Device ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Inference] GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[Inference] Using CPU")

    # --- Tokenizer ---
    enc = tiktoken.get_encoding("gpt2")

    # --- Build streaming model ---
    print(f"\n[Inference] Loading streaming model...")
    print(f"  Expert store: {expert_store_dir}")
    print(f"  Cache capacity: {cache_capacity} experts")

    t0 = time.time()
    model = build_streaming_model(
        config=config,
        expert_store_dir=expert_store_dir,
        device=device,
        cache_capacity=cache_capacity,
    )
    load_time = time.time() - t0
    print(f"  Backbone loaded in {load_time:.2f}s")

    # --- Tokenize prompt ---
    prompt_ids = enc.encode(prompt, allowed_special="all")
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    print(f"\n[Inference] Prompt ({len(prompt_ids)} tokens): \"{prompt}\"")
    print(f"[Inference] Generating {max_new_tokens} tokens...\n")

    # --- Generate ---
    cache = model.cache_manager
    cache_stats_before = cache.get_stats()

    t0 = time.time()
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    gen_time = time.time() - t0

    # --- Decode output ---
    generated_ids = output_ids[0].tolist()
    text = enc.decode(generated_ids)

    print("=" * 60)
    print("GENERATED TEXT")
    print("=" * 60)
    print(text)
    print("=" * 60)

    # --- Statistics ---
    total_tokens = len(generated_ids) - len(prompt_ids)
    tokens_per_sec = total_tokens / gen_time if gen_time > 0 else 0

    cache_stats = cache.get_stats()
    store_stats = model.cache_manager.expert_store.get_stats()

    print(f"\n[Inference] Generation Statistics:")
    print(f"  Tokens generated: {total_tokens}")
    print(f"  Generation time:  {gen_time:.2f}s")
    print(f"  Tokens/sec:       {tokens_per_sec:.1f}")
    print(f"  Temperature:      {temperature}")
    print(f"  Top-p:            {top_p}")

    print(f"\n[Inference] Cache Statistics:")
    print(f"  Capacity:     {cache_stats['capacity']} experts")
    print(f"  Final usage:  {cache_stats['current_size']} experts")
    print(f"  Hits:         {cache_stats['hits']}")
    print(f"  Misses:       {cache_stats['misses']}")
    print(f"  Evictions:    {cache_stats['evictions']}")
    print(f"  Hit rate:     {cache_stats['hit_rate']:.1%}")

    print(f"\n[Inference] Expert Store Statistics:")
    print(f"  Disk loads:     {store_stats['load_count']}")
    print(f"  Total loaded:   {store_stats['total_mb_loaded']:.1f} MB")

    # Report VRAM savings
    total_expert_mb = config.num_layers * config.num_experts * 8.1  # ~8.1 MB each
    cached_expert_mb = cache_stats['current_size'] * 8.1
    print(f"\n[Inference] Memory Efficiency:")
    print(f"  Total expert weight:   {total_expert_mb:.0f} MB (if all loaded)")
    print(f"  Cached in GPU:         {cached_expert_mb:.0f} MB")
    print(f"  VRAM reduction:        {(1 - cached_expert_mb / total_expert_mb) * 100:.0f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="DWR-Transformer Streaming Inference")
    parser.add_argument("--prompt", type=str, default="The meaning of life is",
                        help="Text prompt to complete")
    parser.add_argument("--max-tokens", type=int, default=100,
                        help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Nucleus sampling threshold")
    parser.add_argument("--cache-capacity", type=int, default=96,
                        help="Max experts in GPU cache (96 = all experts)")
    parser.add_argument("--expert-store", type=str, default="checkpoints/expert_store",
                        help="Path to expert store directory")
    args = parser.parse_args()

    inference(
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        cache_capacity=args.cache_capacity,
        expert_store_dir=args.expert_store,
    )


if __name__ == "__main__":
    main()
