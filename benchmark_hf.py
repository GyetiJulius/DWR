"""
Benchmark: DWR-Transformer vs Pre-trained HuggingFace Models.

Evaluates pre-trained models on our exact WikiText-103 validation set
for a direct perplexity comparison. No training required — just inference.

Models compared:
    - DWR-Transformer (ours, 260M total / 57.6M active, trained on WikiText-103)
    - EleutherAI/pythia-70m   (70M,  trained on The Pile)
    - EleutherAI/pythia-160m  (160M, trained on The Pile)
    - EleutherAI/pythia-410m  (410M, trained on The Pile)
    - openai-community/gpt2   (124M, trained on WebText)

Note on fairness:
    Our DWR model was TRAINED on WikiText-103, so it has a natural advantage
    on this eval set. The HuggingFace models were trained on different data
    (The Pile, WebText) and are evaluated zero-shot. This is standard practice
    — papers routinely report zero-shot WikiText-103 PPL for pre-trained models.
    If DWR beats these models with fewer active params, it's a strong result.
    If it doesn't, the MoE routing on WikiText-103 isn't adding enough value.

Usage:
    python benchmark_hf.py                    # run all
    python benchmark_hf.py --models gpt2 pythia-70m  # specific models
    python benchmark_hf.py --skip-dwr         # only HF models
"""

import argparse
import math
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import tiktoken

from config import DWRConfig
from models.transformer import DWRTransformer
from data.dataset import build_dataloaders


@dataclass
class ModelResult:
    name: str
    total_params: int
    active_params: int
    val_loss: float
    val_ppl: float
    throughput_tok_per_sec: float
    peak_gpu_mb: float
    trained_on: str
    notes: str = ""


# ─── HuggingFace model evaluation ───────────────────────────────────────────

HF_MODELS = {
    "pythia-70m":  "EleutherAI/pythia-70m",
    "pythia-160m": "EleutherAI/pythia-160m",
    "pythia-410m": "EleutherAI/pythia-410m",
    "gpt2":        "openai-community/gpt2",
}


@torch.no_grad()
def evaluate_hf_model(
    model_id: str,
    val_tokens: torch.Tensor,
    seq_len: int,
    device: torch.device,
    batch_size: int = 8,
) -> ModelResult:
    """
    Evaluate a HuggingFace causal LM on WikiText-103 validation tokens.

    Uses a sliding window approach with stride=seq_len (non-overlapping)
    to match our DWR evaluation protocol exactly.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    short_name = [k for k, v in HF_MODELS.items() if v == model_id][0]
    print(f"\n{'='*60}")
    print(f"  Evaluating: {model_id}")
    print(f"{'='*60}")

    # Load model
    print(f"  Loading model...")
    t0 = time.time()
    config = AutoConfig.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    ).to(device)
    model.eval()
    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.1f}s")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    # We need to re-tokenize WikiText-103 val with this model's tokenizer
    # because different models use different tokenizers.
    # However, for perplexity comparison, what matters is the model's
    # ability to predict the SAME text, so we tokenize with each model's
    # own tokenizer and report per-token perplexity.
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Get the raw validation text from our cached tokens using GPT-2 BPE decode
    # Then re-encode with the HF model's tokenizer
    enc = tiktoken.get_encoding("gpt2")
    val_text = enc.decode(val_tokens.tolist())

    print(f"  Tokenizing with {model_id} tokenizer...")
    hf_tokens = tokenizer.encode(val_text, return_tensors="pt")[0]
    print(f"  HF tokens: {len(hf_tokens):,} (GPT-2 BPE: {len(val_tokens):,})")

    # Determine model's max seq len
    model_max_len = getattr(config, "max_position_embeddings", 2048)
    eval_seq_len = min(seq_len, model_max_len)
    print(f"  Eval seq_len: {eval_seq_len} (model max: {model_max_len})")

    # Evaluate: sliding window, non-overlapping, same as our TokenDataset
    num_samples = (len(hf_tokens) - 1) // eval_seq_len
    total_nll = 0.0
    total_tokens = 0

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    print(f"  Evaluating {num_samples} windows...")
    eval_start = time.time()

    for i in range(0, num_samples * eval_seq_len, eval_seq_len * batch_size):
        batch_inputs = []
        batch_targets = []

        for b in range(batch_size):
            start = i + b * eval_seq_len
            if start + eval_seq_len + 1 > len(hf_tokens):
                break
            x = hf_tokens[start : start + eval_seq_len]
            y = hf_tokens[start + 1 : start + eval_seq_len + 1]
            batch_inputs.append(x)
            batch_targets.append(y)

        if not batch_inputs:
            break

        input_ids = torch.stack(batch_inputs).to(device)
        target_ids = torch.stack(batch_targets).to(device)

        outputs = model(input_ids)
        logits = outputs.logits  # (B, S, V)

        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_ids.reshape(-1),
            reduction="sum",
        )

        total_nll += loss.item()
        total_tokens += target_ids.numel()

    eval_time = time.time() - eval_start
    peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    avg_loss = total_nll / total_tokens
    ppl = math.exp(min(avg_loss, 20.0))
    throughput = total_tokens / eval_time

    print(f"  Val loss:    {avg_loss:.4f}")
    print(f"  Val PPL:     {ppl:.1f}")
    print(f"  Throughput:  {throughput:.0f} tok/s")
    print(f"  Peak GPU:    {peak_mb:.0f} MB")
    print(f"  Eval time:   {eval_time:.1f}s")

    # Determine training data
    trained_on = "The Pile" if "pythia" in model_id else "WebText"

    result = ModelResult(
        name=short_name,
        total_params=total_params,
        active_params=total_params,  # Dense models: all params active
        val_loss=avg_loss,
        val_ppl=ppl,
        throughput_tok_per_sec=throughput,
        peak_gpu_mb=peak_mb,
        trained_on=trained_on,
        notes=f"Zero-shot eval (trained on {trained_on})",
    )

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return result


# ─── DWR evaluation ─────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_dwr(
    val_loader,
    device: torch.device,
    config: DWRConfig,
) -> Optional[ModelResult]:
    """Evaluate our trained DWR-Transformer."""

    # Try checkpoint_latest.pt first, then checkpoint_best.pt, then epoch checkpoints
    for ckpt_name in ["checkpoint_latest.pt", "checkpoint_best.pt",
                      "checkpoint_epoch3.pt", "checkpoint_epoch2.pt", "checkpoint_epoch1.pt"]:
        ckpt_path = os.path.join(config.checkpoint_dir, ckpt_name)
        if os.path.exists(ckpt_path):
            break
    else:
        print(f"  [SKIP] No DWR checkpoint found in {config.checkpoint_dir}/")
        return None

    print(f"\n{'='*60}")
    print(f"  Evaluating: DWR-Transformer (ours)")
    print(f"{'='*60}")

    model = DWRTransformer(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        d_ff=config.d_ff,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        num_experts=config.num_experts,
        top_k=config.top_k,
        max_seq_len=config.max_seq_len,
        dropout=0.0,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())

    # Active params
    expert_params = sum(p.numel() for p in model.blocks[0].moe_ffn.experts[0].parameters())
    inactive = (config.num_experts - config.top_k) * expert_params * config.num_layers
    active_params = total_params - inactive

    print(f"  Parameters: {total_params:,} total, {active_params:,} active")

    mask = DWRTransformer.generate_causal_mask(config.max_seq_len, device)

    total_nll = 0.0
    total_tokens = 0

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    eval_start = time.time()

    for x, y in val_loader:
        x = x.to(device)
        y = y.to(device)

        logits, _ = model(x, mask=mask)

        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
            reduction="sum",
        )

        total_nll += loss.item()
        total_tokens += y.numel()

    eval_time = time.time() - eval_start
    peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    avg_loss = total_nll / total_tokens
    ppl = math.exp(min(avg_loss, 20.0))
    throughput = total_tokens / eval_time

    print(f"  Val loss:    {avg_loss:.4f}")
    print(f"  Val PPL:     {ppl:.1f}")
    print(f"  Throughput:  {throughput:.0f} tok/s")
    print(f"  Peak GPU:    {peak_mb:.0f} MB")

    del model
    torch.cuda.empty_cache()

    return ModelResult(
        name="DWR (ours)",
        total_params=total_params,
        active_params=active_params,
        val_loss=avg_loss,
        val_ppl=ppl,
        throughput_tok_per_sec=throughput,
        peak_gpu_mb=peak_mb,
        trained_on="WikiText-103",
        notes=f"Trained 3 epochs. top-2 of 16 experts/layer.",
    )


# ─── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark DWR vs pre-trained HuggingFace models on WikiText-103 val."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(HF_MODELS.keys()),
        default=list(HF_MODELS.keys()),
        help="Which HF models to evaluate (default: all)",
    )
    parser.add_argument(
        "--skip-dwr",
        action="store_true",
        help="Skip DWR evaluation (only evaluate HF models)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for HF model evaluation",
    )
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    # Load data
    config = DWRConfig()
    print("\nLoading WikiText-103 validation data...")
    train_loader, val_loader, enc = build_dataloaders(
        max_seq_len=config.max_seq_len,
        batch_size=config.batch_size,
        data_cache_dir=config.data_cache_dir,
    )

    # Load raw validation tokens for HF evaluation
    val_cache = os.path.join(config.data_cache_dir, "val_tokens.pt")
    val_tokens = torch.load(val_cache, weights_only=True)
    print(f"Val tokens: {len(val_tokens):,}")

    results: List[ModelResult] = []

    # ── DWR ──
    if not args.skip_dwr:
        dwr_result = evaluate_dwr(val_loader, device, config)
        if dwr_result:
            results.append(dwr_result)

    # ── HuggingFace models ──
    for model_key in args.models:
        model_id = HF_MODELS[model_key]
        try:
            result = evaluate_hf_model(
                model_id, val_tokens, config.max_seq_len, device, args.batch_size
            )
            results.append(result)
        except Exception as e:
            print(f"  [ERROR] {model_key}: {e}")

    # ── Results Table ──
    if not results:
        print("\nNo results to show.")
        return

    # Sort by val_ppl
    results.sort(key=lambda r: r.val_ppl)

    print(f"\n{'='*100}")
    print(f"BENCHMARK RESULTS — WikiText-103 Validation Perplexity")
    print(f"{'='*100}")
    print(f"{'Model':<16} {'Total':>12} {'Active':>12} {'Val PPL':>10} "
          f"{'Tok/s':>10} {'GPU MB':>10}  {'Trained On'}")
    print(f"{'-'*100}")

    for r in results:
        marker = " ***" if r.name == "DWR (ours)" else ""
        print(f"{r.name:<16} {r.total_params:>12,} {r.active_params:>12,} "
              f"{r.val_ppl:>10.1f} {r.throughput_tok_per_sec:>10.0f} "
              f"{r.peak_gpu_mb:>10.0f}  {r.trained_on}{marker}")

    print(f"{'-'*100}")

    # ── Analysis ──
    dwr = next((r for r in results if r.name == "DWR (ours)"), None)
    if dwr:
        print(f"\n{'='*100}")
        print(f"ANALYSIS (DWR = {dwr.val_ppl:.1f} PPL, {dwr.active_params:,} active params)")
        print(f"{'='*100}")

        for r in results:
            if r.name == dwr.name:
                continue

            ppl_diff = r.val_ppl - dwr.val_ppl
            active_ratio = r.active_params / dwr.active_params
            direction = "better" if ppl_diff > 0 else "worse"

            print(f"\n  vs {r.name} ({r.total_params:,} params, {r.trained_on}):")
            print(f"    PPL: DWR={dwr.val_ppl:.1f} vs {r.val_ppl:.1f} "
                  f"→ DWR is {abs(ppl_diff):.1f} {direction}")
            print(f"    Active params: DWR uses {active_ratio:.1f}× "
                  f"{'fewer' if active_ratio > 1 else 'more'} active params")

            if ppl_diff > 0 and active_ratio > 1:
                print(f"    ✓ DWR achieves better PPL with {active_ratio:.1f}× fewer active params")
            elif ppl_diff > 0:
                print(f"    ✓ DWR achieves better PPL (but uses more active params)")
            elif active_ratio > 1:
                print(f"    ~ DWR uses {active_ratio:.1f}× fewer active params but PPL is worse")
            else:
                print(f"    ✗ Baseline wins on both PPL and efficiency")

        print(f"\n  Note: DWR was trained on WikiText-103 (in-domain).")
        print(f"  HF models are zero-shot (out-of-domain), which is a disadvantage for them.")
        print(f"  A fair comparison would train all models on the same data.")

    # ── Save ──
    out_path = "benchmark_results.txt"
    with open(out_path, "w") as f:
        f.write(f"WikiText-103 Validation Perplexity Benchmark\n")
        f.write(f"{'='*100}\n")
        f.write(f"{'Model':<16} {'Total':>12} {'Active':>12} {'Val PPL':>10} "
                f"{'Tok/s':>10} {'GPU MB':>10}  {'Trained On'}\n")
        f.write(f"{'-'*100}\n")
        for r in results:
            f.write(f"{r.name:<16} {r.total_params:>12,} {r.active_params:>12,} "
                    f"{r.val_ppl:>10.1f} {r.throughput_tok_per_sec:>10.0f} "
                    f"{r.peak_gpu_mb:>10.0f}  {r.trained_on}\n")
        f.write(f"{'-'*100}\n")
        for r in results:
            f.write(f"\n{r.name}: {r.notes}\n")

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
