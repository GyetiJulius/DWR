"""
DWR-Transformer Training Loop (Phase 2).

Implements standard autoregressive language model training:
    L_total = CrossEntropy(logits, targets) + λ * L_balance

Training strategy (design.md Section 10, Phase 1):
    "Train normally with all experts loaded. No streaming during training."

Features:
    - AdamW optimizer with linear warmup + cosine decay
    - Gradient clipping
    - Periodic validation with perplexity reporting
    - Expert utilization logging
    - Full checkpoint saving (model + optimizer)
    - Per-expert export at end of training (Phase 3 ready)
    - Mixed precision (fp16/bf16) when CUDA is available
"""

import math
import os
import time
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import DWRConfig
from models.transformer import DWRTransformer
from data.dataset import build_dataloaders
from utils.checkpoint import save_checkpoint, export_experts


def build_model(config: DWRConfig, device: torch.device) -> DWRTransformer:
    """Construct a DWR-Transformer from config and move to device."""
    model = DWRTransformer(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        d_ff=config.d_ff,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        num_experts=config.num_experts,
        top_k=config.top_k,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout,
    )
    return model.to(device)


def get_lr(
    step: int, warmup_steps: int, max_steps: int, max_lr: float
) -> float:
    """
    Compute learning rate with linear warmup + cosine decay.

    Args:
        step:         Current training step (0-indexed).
        warmup_steps: Number of linear warmup steps.
        max_steps:    Total training steps.
        max_lr:       Peak learning rate.

    Returns:
        Learning rate for this step.
    """
    min_lr = max_lr * 0.1  # Decay to 10% of max

    # Linear warmup
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps

    # Cosine decay after warmup
    if step >= max_steps:
        return min_lr

    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(
    model: DWRTransformer,
    val_loader: DataLoader,
    device: torch.device,
    config: DWRConfig,
) -> dict:
    """
    Run validation and compute loss + perplexity.

    Returns:
        Dict with val_loss, val_ppl, avg_aux_loss.
    """
    model.eval()

    total_loss = 0.0
    total_aux = 0.0
    total_tokens = 0
    num_batches = 0

    mask = DWRTransformer.generate_causal_mask(
        config.max_seq_len, device
    )

    for x, y in val_loader:
        x = x.to(device)  # (B, S)
        y = y.to(device)  # (B, S)

        logits, aux_loss = model(x, mask=mask)  # (B, S, V)

        # Cross-entropy loss, computed per-token then averaged
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),  # (B*S, V)
            y.view(-1),                         # (B*S,)
            reduction="sum",
        )

        batch_tokens = y.numel()
        total_loss += loss.item()
        total_aux += aux_loss.item()
        total_tokens += batch_tokens
        num_batches += 1

    avg_loss = total_loss / total_tokens
    avg_aux = total_aux / max(num_batches, 1)
    ppl = math.exp(min(avg_loss, 20.0))  # Clamp to avoid overflow

    model.train()

    return {
        "val_loss": avg_loss,
        "val_ppl": ppl,
        "avg_aux_loss": avg_aux,
    }


def collect_expert_utilization(model: DWRTransformer) -> str:
    """
    Summarize router gate bias magnitudes as a rough proxy for utilization.
    (Actual utilization tracking requires counting dispatched tokens,
    which will be added in Phase 3 with proper metrics.)
    """
    lines = []
    for layer_idx, block in enumerate(model.blocks):
        gate_weight = block.moe_ffn.router.gate.weight  # (num_experts, d_model)
        norms = gate_weight.norm(dim=1)  # (num_experts,)
        min_n, max_n, mean_n = norms.min().item(), norms.max().item(), norms.mean().item()
        lines.append(
            f"  Layer {layer_idx}: gate norm min={min_n:.3f} max={max_n:.3f} "
            f"mean={mean_n:.3f} spread={max_n - min_n:.3f}"
        )
    return "\n".join(lines)


def train() -> None:
    """Full training loop for DWR-Transformer."""

    config = DWRConfig()

    # --- Device selection ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Train] Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"[Train] GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("[Train] Using CPU (no CUDA available)")

    # --- Data ---
    print("\n[Train] Preparing data...")
    train_loader, val_loader, enc = build_dataloaders(
        max_seq_len=config.max_seq_len,
        batch_size=config.batch_size,
        data_cache_dir=config.data_cache_dir,
    )

    steps_per_epoch = len(train_loader)
    max_steps = config.max_epochs * steps_per_epoch
    print(f"[Train] Steps/epoch: {steps_per_epoch}  "
          f"Total steps: {max_steps}  "
          f"Warmup: {config.warmup_steps}")

    # --- Model ---
    print("\n[Train] Building model...")
    model = build_model(config, device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Train] Total parameters: {total_params:,}")

    # --- Optimizer ---
    # Separate weight decay: don't apply to biases, LayerNorm, or embeddings
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

    print(f"[Train] Decay params: {sum(p.numel() for p in decay_params):,}  "
          f"No-decay params: {sum(p.numel() for p in no_decay_params):,}")

    # --- Mixed precision ---
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if use_amp:
        print(f"[Train] Mixed precision enabled: {amp_dtype}")

    # --- Causal mask (precomputed, reused every step) ---
    causal_mask = DWRTransformer.generate_causal_mask(
        config.max_seq_len, device
    )

    # --- Training loop ---
    print("\n" + "=" * 60)
    print("TRAINING START")
    print("=" * 60 + "\n")

    global_step = 0
    best_val_loss = float("inf")
    model.train()

    for epoch in range(1, config.max_epochs + 1):
        epoch_loss = 0.0
        epoch_aux = 0.0
        epoch_tokens = 0
        epoch_start = time.time()

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{config.max_epochs}",
            unit="batch",
        )

        for batch_idx, (x, y) in enumerate(pbar):
            x = x.to(device)  # (B, S)
            y = y.to(device)  # (B, S)

            # Update learning rate
            lr = get_lr(global_step, config.warmup_steps, max_steps, config.learning_rate)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # Forward pass with mixed precision
            with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                logits, aux_loss = model(x, mask=causal_mask)  # (B, S, V)

                task_loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),  # (B*S, V)
                    y.view(-1),                         # (B*S,)
                )

                total_loss = task_loss + config.balance_loss_coeff * aux_loss

            # Backward pass
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            # Accumulate stats
            batch_tokens = y.numel()
            epoch_loss += task_loss.item() * batch_tokens
            epoch_aux += aux_loss.item()
            epoch_tokens += batch_tokens
            global_step += 1

            # Progress bar update
            pbar.set_postfix({
                "loss": f"{task_loss.item():.3f}",
                "ppl": f"{math.exp(min(task_loss.item(), 20.0)):.1f}",
                "aux": f"{aux_loss.item():.2f}",
                "lr": f"{lr:.2e}",
            })

            # Periodic logging
            if global_step % config.log_interval == 0:
                avg_loss = epoch_loss / epoch_tokens
                tqdm.write(
                    f"  Step {global_step:>6d} | "
                    f"loss {task_loss.item():.4f} | "
                    f"ppl {math.exp(min(task_loss.item(), 20.0)):.1f} | "
                    f"aux {aux_loss.item():.3f} | "
                    f"lr {lr:.2e}"
                )

            # Periodic validation
            if global_step % config.eval_interval == 0:
                val_metrics = evaluate(model, val_loader, device, config)
                tqdm.write(
                    f"\n  [Val @ step {global_step}] "
                    f"loss={val_metrics['val_loss']:.4f}  "
                    f"ppl={val_metrics['val_ppl']:.1f}  "
                    f"aux={val_metrics['avg_aux_loss']:.3f}\n"
                )

                if val_metrics["val_loss"] < best_val_loss:
                    best_val_loss = val_metrics["val_loss"]
                    save_checkpoint(
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
        print(f"Epoch {epoch} complete in {epoch_time:.1f}s")
        print(f"  Train loss: {avg_epoch_loss:.4f}  ppl: {avg_epoch_ppl:.1f}")

        # Validation at epoch end
        val_metrics = evaluate(model, val_loader, device, config)
        print(f"  Val loss:   {val_metrics['val_loss']:.4f}  "
              f"ppl: {val_metrics['val_ppl']:.1f}  "
              f"aux: {val_metrics['avg_aux_loss']:.3f}")

        # Expert utilization diagnostics
        print(f"\n  Router diagnostics:\n{collect_expert_utilization(model)}")

        # Checkpoint
        if epoch % config.checkpoint_interval == 0:
            save_checkpoint(
                model, optimizer, epoch, global_step,
                avg_epoch_loss, val_metrics["val_loss"],
                config, config.checkpoint_dir,
            )

        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]

        print(f"  Best val loss so far: {best_val_loss:.4f}")
        print(f"{'='*60}\n")

    # --- Export experts for Phase 3 ---
    print("\n[Train] Exporting individual expert files (Phase 3 ready)...")
    export_dir = os.path.join(config.checkpoint_dir, "expert_store")
    export_experts(model, export_dir)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints: {config.checkpoint_dir}/")
    print(f"Expert store: {export_dir}/")


if __name__ == "__main__":
    train()
