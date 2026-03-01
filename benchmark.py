"""
Benchmark Comparison: DWR-Transformer vs Dense Baselines.

Loads trained checkpoints for all three models and compares:
    1. Perplexity (quality)
    2. Parameter count (total vs active)
    3. Throughput (tokens/sec on forward pass)
    4. Peak GPU memory during inference

Usage:
    python benchmark.py

Expects trained checkpoints in:
    checkpoints/           (DWR-Transformer)
    checkpoints_dense_small/  (Dense-Small, compute-matched)
    checkpoints_dense_large/  (Dense-Large, param-matched)
"""

import argparse
import math
import os
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import DWRConfig
from configs.dense_config import DenseConfig, dense_small_config, dense_large_config
from models.transformer import DWRTransformer
from models.dense_transformer import DenseTransformer
from data.dataset import build_dataloaders


@dataclass
class BenchmarkResult:
    """Results from benchmarking a single model."""
    model_name: str
    total_params: int
    active_params: int       # For dense, same as total; for MoE, per-token
    val_loss: float
    val_ppl: float
    throughput_tok_per_sec: float
    peak_gpu_mb: float
    inference_gpu_mb: float  # GPU memory during inference forward pass


def count_dwr_active_params(model: DWRTransformer, top_k: int) -> int:
    """
    Count active parameters per forward pass for DWR model.

    Active = everything except (num_experts - top_k) × expert_params per layer.
    """
    total = sum(p.numel() for p in model.parameters())

    expert_params_per_layer = 0
    for expert in model.blocks[0].moe_ffn.experts:
        expert_params_per_layer += sum(p.numel() for p in expert.parameters())

    num_experts = len(model.blocks[0].moe_ffn.experts)
    num_layers = len(model.blocks)

    # Inactive expert params: (num_experts - top_k) experts per layer
    inactive_expert_params = (num_experts - top_k) * expert_params_per_layer * num_layers

    return total - inactive_expert_params


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    max_seq_len: int,
    is_dwr: bool = False,
) -> dict:
    """
    Evaluate model on validation set.

    Returns:
        Dict with val_loss and val_ppl.
    """
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    mask = DWRTransformer.generate_causal_mask(max_seq_len, device)

    for x, y in val_loader:
        x = x.to(device)
        y = y.to(device)

        if is_dwr:
            logits, _ = model(x, mask=mask)
        else:
            logits = model(x, mask=mask)

        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
            reduction="sum",
        )

        total_loss += loss.item()
        total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(min(avg_loss, 20.0))

    return {"val_loss": avg_loss, "val_ppl": ppl}


@torch.no_grad()
def measure_throughput(
    model: nn.Module,
    device: torch.device,
    max_seq_len: int,
    batch_size: int = 8,
    num_warmup: int = 5,
    num_measure: int = 20,
    is_dwr: bool = False,
) -> float:
    """
    Measure inference throughput in tokens/sec.

    Uses dummy data to isolate model computation time.
    """
    model.eval()

    input_ids = torch.randint(0, 1000, (batch_size, max_seq_len), device=device)
    mask = DWRTransformer.generate_causal_mask(max_seq_len, device)

    # Warmup
    for _ in range(num_warmup):
        if is_dwr:
            model(input_ids, mask=mask)
        else:
            model(input_ids, mask=mask)

    torch.cuda.synchronize()

    # Measure
    start = time.time()
    for _ in range(num_measure):
        if is_dwr:
            model(input_ids, mask=mask)
        else:
            model(input_ids, mask=mask)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    total_tokens = batch_size * max_seq_len * num_measure
    return total_tokens / elapsed


@torch.no_grad()
def measure_peak_memory(
    model: nn.Module,
    device: torch.device,
    max_seq_len: int,
    batch_size: int = 8,
    is_dwr: bool = False,
) -> float:
    """
    Measure peak GPU memory during a forward pass (MB).
    """
    model.eval()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    input_ids = torch.randint(0, 1000, (batch_size, max_seq_len), device=device)
    mask = DWRTransformer.generate_causal_mask(max_seq_len, device)

    if is_dwr:
        model(input_ids, mask=mask)
    else:
        model(input_ids, mask=mask)

    torch.cuda.synchronize()
    peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    return peak_mb


def load_dwr_model(device: torch.device) -> Optional[DWRTransformer]:
    """Load trained DWR-Transformer from checkpoint."""
    config = DWRConfig()
    ckpt_path = os.path.join(config.checkpoint_dir, "checkpoint_best.pt")

    if not os.path.exists(ckpt_path):
        print(f"  [SKIP] DWR checkpoint not found: {ckpt_path}")
        return None

    model = DWRTransformer(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        d_ff=config.d_ff,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        num_experts=config.num_experts,
        top_k=config.top_k,
        max_seq_len=config.max_seq_len,
        dropout=0.0,  # No dropout during eval
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"  [DWR] Loaded from {ckpt_path} "
          f"(epoch {ckpt.get('epoch', '?')}, val_loss {ckpt.get('val_loss', '?')})")
    return model


def load_dense_model(
    config: DenseConfig, device: torch.device
) -> Optional[DenseTransformer]:
    """Load trained Dense Transformer from checkpoint."""
    ckpt_path = os.path.join(config.checkpoint_dir, "checkpoint_best.pt")

    if not os.path.exists(ckpt_path):
        print(f"  [SKIP] {config.model_name} checkpoint not found: {ckpt_path}")
        return None

    model = DenseTransformer(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        d_ff=config.d_ff,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        max_seq_len=config.max_seq_len,
        dropout=0.0,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"  [{config.model_name}] Loaded from {ckpt_path}")
    return model


def benchmark() -> None:
    """Run full benchmark comparison."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # --- Load validation data ---
    dwr_config = DWRConfig()
    print("\nLoading validation data...")
    _, val_loader, _ = build_dataloaders(
        max_seq_len=dwr_config.max_seq_len,
        batch_size=8,
        data_cache_dir=dwr_config.data_cache_dir,
    )

    results: list[BenchmarkResult] = []

    # ===== DWR-Transformer =====
    print("\n--- DWR-Transformer ---")
    dwr_model = load_dwr_model(device)
    if dwr_model is not None:
        total_params = sum(p.numel() for p in dwr_model.parameters())
        active_params = count_dwr_active_params(dwr_model, dwr_config.top_k)

        print(f"  Evaluating perplexity...")
        val_metrics = evaluate_model(dwr_model, val_loader, device, dwr_config.max_seq_len, is_dwr=True)

        print(f"  Measuring throughput...")
        throughput = measure_throughput(dwr_model, device, dwr_config.max_seq_len, is_dwr=True)

        print(f"  Measuring memory...")
        peak_mem = measure_peak_memory(dwr_model, device, dwr_config.max_seq_len, is_dwr=True)

        results.append(BenchmarkResult(
            model_name="DWR-Transformer",
            total_params=total_params,
            active_params=active_params,
            val_loss=val_metrics["val_loss"],
            val_ppl=val_metrics["val_ppl"],
            throughput_tok_per_sec=throughput,
            peak_gpu_mb=peak_mem,
            inference_gpu_mb=peak_mem,
        ))

        del dwr_model
        torch.cuda.empty_cache()

    # ===== Dense-Small =====
    print("\n--- Dense-Small (compute-matched) ---")
    ds_config = dense_small_config()
    ds_model = load_dense_model(ds_config, device)
    if ds_model is not None:
        total_params = sum(p.numel() for p in ds_model.parameters())

        print(f"  Evaluating perplexity...")
        val_metrics = evaluate_model(ds_model, val_loader, device, ds_config.max_seq_len)

        print(f"  Measuring throughput...")
        throughput = measure_throughput(ds_model, device, ds_config.max_seq_len)

        print(f"  Measuring memory...")
        peak_mem = measure_peak_memory(ds_model, device, ds_config.max_seq_len)

        results.append(BenchmarkResult(
            model_name="Dense-Small",
            total_params=total_params,
            active_params=total_params,  # Dense: all params active
            val_loss=val_metrics["val_loss"],
            val_ppl=val_metrics["val_ppl"],
            throughput_tok_per_sec=throughput,
            peak_gpu_mb=peak_mem,
            inference_gpu_mb=peak_mem,
        ))

        del ds_model
        torch.cuda.empty_cache()

    # ===== Dense-Large =====
    print("\n--- Dense-Large (param-matched) ---")
    dl_config = dense_large_config()
    dl_model = load_dense_model(dl_config, device)
    if dl_model is not None:
        total_params = sum(p.numel() for p in dl_model.parameters())

        print(f"  Evaluating perplexity...")
        val_metrics = evaluate_model(dl_model, val_loader, device, dl_config.max_seq_len)

        print(f"  Measuring throughput...")
        throughput = measure_throughput(dl_model, device, dl_config.max_seq_len)

        print(f"  Measuring memory...")
        peak_mem = measure_peak_memory(dl_model, device, dl_config.max_seq_len)

        results.append(BenchmarkResult(
            model_name="Dense-Large",
            total_params=total_params,
            active_params=total_params,
            val_loss=val_metrics["val_loss"],
            val_ppl=val_metrics["val_ppl"],
            throughput_tok_per_sec=throughput,
            peak_gpu_mb=peak_mem,
            inference_gpu_mb=peak_mem,
        ))

        del dl_model
        torch.cuda.empty_cache()

    # ===== Print Results Table =====
    if not results:
        print("\nNo models found. Train models first:")
        print("  python train.py                           # DWR")
        print("  python train_dense.py --model dense-small # Dense-Small")
        print("  python train_dense.py --model dense-large # Dense-Large")
        return

    print(f"\n{'='*90}")
    print(f"BENCHMARK RESULTS")
    print(f"{'='*90}")
    print(f"{'Model':<20} {'Total Params':>14} {'Active Params':>14} "
          f"{'Val PPL':>10} {'Tok/s':>10} {'GPU MB':>10}")
    print(f"{'-'*90}")

    for r in results:
        print(f"{r.model_name:<20} {r.total_params:>14,} {r.active_params:>14,} "
              f"{r.val_ppl:>10.1f} {r.throughput_tok_per_sec:>10.0f} "
              f"{r.peak_gpu_mb:>10.0f}")

    print(f"{'-'*90}")

    # ===== Quality per Compute Analysis =====
    if len(results) >= 2:
        print(f"\n{'='*90}")
        print(f"ANALYSIS")
        print(f"{'='*90}")

        dwr_result = next((r for r in results if "DWR" in r.model_name), None)

        if dwr_result:
            for r in results:
                if r.model_name == dwr_result.model_name:
                    continue

                ppl_diff = r.val_ppl - dwr_result.val_ppl
                ppl_pct = (ppl_diff / r.val_ppl) * 100

                param_ratio = r.active_params / dwr_result.active_params
                speed_ratio = dwr_result.throughput_tok_per_sec / r.throughput_tok_per_sec
                mem_ratio = r.peak_gpu_mb / dwr_result.peak_gpu_mb

                print(f"\n  DWR vs {r.model_name}:")
                print(f"    PPL:        DWR={dwr_result.val_ppl:.1f} vs {r.model_name}={r.val_ppl:.1f} "
                      f"(DWR {'better' if ppl_diff > 0 else 'worse'} by {abs(ppl_diff):.1f})")
                print(f"    Active:     DWR uses {dwr_result.active_params/1e6:.1f}M vs "
                      f"{r.active_params/1e6:.1f}M ({param_ratio:.1f}×)")
                print(f"    Throughput: DWR={dwr_result.throughput_tok_per_sec:.0f} vs "
                      f"{r.throughput_tok_per_sec:.0f} tok/s ({speed_ratio:.2f}×)")
                print(f"    GPU Memory: DWR={dwr_result.peak_gpu_mb:.0f} vs "
                      f"{r.peak_gpu_mb:.0f} MB ({mem_ratio:.2f}×)")

    # ===== Save results =====
    out_path = "benchmark_results.txt"
    with open(out_path, "w") as f:
        f.write(f"{'Model':<20} {'Total Params':>14} {'Active Params':>14} "
                f"{'Val Loss':>10} {'Val PPL':>10} {'Tok/s':>10} {'GPU MB':>10}\n")
        for r in results:
            f.write(f"{r.model_name:<20} {r.total_params:>14,} {r.active_params:>14,} "
                    f"{r.val_loss:>10.4f} {r.val_ppl:>10.1f} "
                    f"{r.throughput_tok_per_sec:>10.0f} {r.peak_gpu_mb:>10.0f}\n")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    benchmark()
