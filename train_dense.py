"""
Dense Transformer Baseline Training (Step 1: Dense Baseline Comparison).

Trains a standard dense transformer on WikiText-103 with identical
hyperparameters to the DWR-Transformer for controlled comparison.

Key differences from train.py (DWR training):
    - No auxiliary load-balancing loss (no MoE routing)
    - No expert export at end of training
    - No expert utilization tracking
    - Model returns logits only (no aux_loss tuple)

Two model variants:
    python train_dense.py --model dense-small   # compute-matched (~83M)
    python train_dense.py --model dense-large    # param-matched (~261M)

All other training aspects are identical: dataset, tokenizer, optimizer,
learning rate schedule, gradient accumulation, mixed precision.
"""

import argparse
import math
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.dense_config import DenseConfig, dense_small_config, dense_large_config
from models.dense_transformer import DenseTransformer
from data.dataset import build_dataloaders


def build_dense_model(config: DenseConfig, device: torch.device) -> DenseTransformer:
    """Construct a DenseTransformer from config and move to device."""
    model = DenseTransformer(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        d_ff=config.d_ff,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout,
    )
    return model.to(device)


def get_lr(
    step: int, warmup_steps: int, max_steps: int, max_lr: float
) -> float:
    """
    Learning rate with linear warmup + cosine decay.
    Identical to DWR train.py for fair comparison.
    """
    min_lr = max_lr * 0.1

    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps

    if step >= max_steps:
        return min_lr

    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(
    model: DenseTransformer,
    val_loader: DataLoader,
    device: torch.device,
    config: DenseConfig,
) -> dict:
    """Run validation and compute loss + perplexity."""
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    mask = DenseTransformer.generate_causal_mask(config.max_seq_len, device)

    for x, y in val_loader:
        x = x.to(device)
        y = y.to(device)

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

    model.train()

    return {"val_loss": avg_loss, "val_ppl": ppl}


def save_dense_checkpoint(
    model: DenseTransformer,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    train_loss: float,
    val_loss: float,
    config: DenseConfig,
    save_dir: str,
) -> None:
    """Save training checkpoint."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"checkpoint_best.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "config": {
            "d_model": config.d_model,
            "d_ff": config.d_ff,
            "num_layers": config.num_layers,
            "num_heads": config.num_heads,
            "model_name": config.model_name,
        },
    }, path)
    print(f"  [Checkpoint] Saved to {path} (val_loss={val_loss:.4f})")


def train_dense(config: DenseConfig) -> dict:
    """
    Full training loop for dense Transformer baseline.

    Returns:
        Dict with final training metrics.
    """

    # --- Device selection ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[{config.model_name}] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[{config.model_name}] GPU Memory: "
              f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print(f"[{config.model_name}] Using CPU")

    # --- Data ---
    print(f"\n[{config.model_name}] Preparing data...")
    train_loader, val_loader, enc = build_dataloaders(
        max_seq_len=config.max_seq_len,
        batch_size=config.batch_size,
        data_cache_dir=config.data_cache_dir,
    )

    micro_steps_per_epoch = len(train_loader)
    steps_per_epoch = micro_steps_per_epoch // config.grad_accum_steps
    max_steps = config.max_epochs * steps_per_epoch
    print(f"[{config.model_name}] Micro-batches/epoch: {micro_steps_per_epoch}  "
          f"Optimizer steps/epoch: {steps_per_epoch}  "
          f"Total optimizer steps: {max_steps}  "
          f"Grad accum: {config.grad_accum_steps}  "
          f"Effective batch: {config.batch_size * config.grad_accum_steps}  "
          f"Warmup: {config.warmup_steps}")

    # --- Model ---
    print(f"\n[{config.model_name}] Building model...")
    model = build_dense_model(config, device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[{config.model_name}] Total parameters: {total_params:,}")
    print(f"[{config.model_name}] Config: d_model={config.d_model}, d_ff={config.d_ff}, "
          f"layers={config.num_layers}, heads={config.num_heads}")

    # --- Optimizer ---
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or "norm" in name or "bias" in name or "embedding" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=config.learning_rate, betas=(0.9, 0.95))

    print(f"[{config.model_name}] Decay params: {sum(p.numel() for p in decay_params):,}  "
          f"No-decay params: {sum(p.numel() for p in no_decay_params):,}")

    # --- Mixed precision ---
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if use_amp:
        print(f"[{config.model_name}] Mixed precision: {amp_dtype}")

    # --- Causal mask ---
    causal_mask = DenseTransformer.generate_causal_mask(config.max_seq_len, device)

    # --- Training loop ---
    print(f"\n{'='*60}")
    print(f"TRAINING START: {config.model_name}")
    print(f"{'='*60}\n")

    global_step = 0
    best_val_loss = float("inf")
    lr = config.learning_rate
    model.train()

    training_start = time.time()

    for epoch in range(1, config.max_epochs + 1):
        epoch_loss = 0.0
        epoch_tokens = 0
        epoch_start = time.time()

        pbar = tqdm(
            train_loader,
            desc=f"[{config.model_name}] Epoch {epoch}/{config.max_epochs}",
            unit="batch",
        )

        for batch_idx, (x, y) in enumerate(pbar):
            x = x.to(device)
            y = y.to(device)

            is_accum_step = (batch_idx + 1) % config.grad_accum_steps != 0

            # Forward pass with mixed precision
            with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(x, mask=causal_mask)

                task_loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                )

                # Scale for gradient accumulation
                scaled_loss = task_loss / config.grad_accum_steps

            # Backward
            scaler.scale(scaled_loss).backward()

            # Accumulate stats
            batch_tokens = y.numel()
            epoch_loss += task_loss.item() * batch_tokens
            epoch_tokens += batch_tokens

            if not is_accum_step or (batch_idx + 1) == len(train_loader):
                # Optimizer step
                lr = get_lr(global_step, config.warmup_steps, max_steps, config.learning_rate)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1

            # Progress bar
            pbar.set_postfix({
                "loss": f"{task_loss.item():.3f}",
                "ppl": f"{math.exp(min(task_loss.item(), 20.0)):.1f}",
                "lr": f"{lr:.2e}" if not is_accum_step else "accum",
            })

            # Periodic logging
            if not is_accum_step and global_step > 0 and global_step % config.log_interval == 0:
                tqdm.write(
                    f"  Step {global_step:>6d} | "
                    f"loss {task_loss.item():.4f} | "
                    f"ppl {math.exp(min(task_loss.item(), 20.0)):.1f} | "
                    f"lr {lr:.2e}"
                )

            # Periodic validation
            if not is_accum_step and global_step > 0 and global_step % config.eval_interval == 0:
                val_metrics = evaluate(model, val_loader, device, config)
                tqdm.write(
                    f"\n  [Val @ step {global_step}] "
                    f"loss={val_metrics['val_loss']:.4f}  "
                    f"ppl={val_metrics['val_ppl']:.1f}\n"
                )

                if val_metrics["val_loss"] < best_val_loss:
                    best_val_loss = val_metrics["val_loss"]
                    save_dense_checkpoint(
                        model, optimizer, epoch, global_step,
                        epoch_loss / epoch_tokens, val_metrics["val_loss"],
                        config, config.checkpoint_dir,
                    )

                model.train()

        # End of epoch
        epoch_time = time.time() - epoch_start
        avg_epoch_loss = epoch_loss / epoch_tokens
        avg_epoch_ppl = math.exp(min(avg_epoch_loss, 20.0))

        print(f"\n{'='*60}")
        print(f"[{config.model_name}] Epoch {epoch} complete in {epoch_time:.1f}s")
        print(f"  Train loss: {avg_epoch_loss:.4f}  ppl: {avg_epoch_ppl:.1f}")

        # Epoch-end validation
        val_metrics = evaluate(model, val_loader, device, config)
        print(f"  Val loss:   {val_metrics['val_loss']:.4f}  ppl: {val_metrics['val_ppl']:.1f}")

        # Checkpoint
        if epoch % config.checkpoint_interval == 0:
            save_dense_checkpoint(
                model, optimizer, epoch, global_step,
                avg_epoch_loss, val_metrics["val_loss"],
                config, config.checkpoint_dir,
            )

        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]

        print(f"  Best val loss so far: {best_val_loss:.4f}")
        print(f"{'='*60}\n")

    total_time = time.time() - training_start

    print(f"\n[{config.model_name}] Training complete.")
    print(f"  Total time:     {total_time:.1f}s ({total_time/3600:.2f}h)")
    print(f"  Best val loss:  {best_val_loss:.4f}")
    print(f"  Best val PPL:   {math.exp(min(best_val_loss, 20.0)):.1f}")
    print(f"  Checkpoints:    {config.checkpoint_dir}/")

    return {
        "model_name": config.model_name,
        "total_params": total_params,
        "best_val_loss": best_val_loss,
        "best_val_ppl": math.exp(min(best_val_loss, 20.0)),
        "total_time_s": total_time,
        "epochs": config.max_epochs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train dense Transformer baseline for DWR comparison."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["dense-small", "dense-large"],
        default="dense-small",
        help="Which baseline: dense-small (compute-matched) or dense-large (param-matched)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override max_epochs (default: 3, same as DWR)",
    )
    args = parser.parse_args()

    if args.model == "dense-small":
        config = dense_small_config()
    else:
        config = dense_large_config()

    if args.epochs is not None:
        config.max_epochs = args.epochs

    print(f"{'='*60}")
    print(f"Dense Baseline Training: {config.model_name}")
    print(f"{'='*60}")
    print(f"  d_model:  {config.d_model}")
    print(f"  d_ff:     {config.d_ff}")
    print(f"  layers:   {config.num_layers}")
    print(f"  heads:    {config.num_heads}")
    print(f"  epochs:   {config.max_epochs}")
    print(f"  batch:    {config.batch_size} × {config.grad_accum_steps} = "
          f"{config.batch_size * config.grad_accum_steps}")
    print(f"{'='*60}\n")

    results = train_dense(config)

    # Write summary for benchmark comparison
    summary_path = os.path.join(config.checkpoint_dir, "results.txt")
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    with open(summary_path, "w") as f:
        for k, v in results.items():
            f.write(f"{k}: {v}\n")
    print(f"\nResults saved to {summary_path}")


if __name__ == "__main__":
    main()
