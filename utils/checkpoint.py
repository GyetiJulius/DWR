"""
Checkpoint utilities: model save/load and per-expert serialization.

Supports two serialization modes:
1. Full checkpoint: entire model state_dict + optimizer + metadata
   → Used for resuming training.
2. Expert export: each expert saved as an independent .pt file
   → Used for Phase 3 disk-backed expert loading.

Expert export layout (design.md Section 8):
    export_dir/
        expert_layer0_id0.pt
        expert_layer0_id1.pt
        ...
        expert_layer5_id15.pt
        backbone.pt    (everything except experts)
"""

import os
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    global_step: int,
    train_loss: float,
    val_loss: float,
    config: Any,
    checkpoint_dir: str,
) -> str:
    """
    Save full training checkpoint.

    Args:
        model:          Model to save.
        optimizer:      Optimizer state to save.
        epoch:          Current epoch number.
        global_step:    Total training steps completed.
        train_loss:     Latest training loss.
        val_loss:       Latest validation loss.
        config:         DWRConfig instance (saved for reproducibility).
        checkpoint_dir: Directory to save checkpoint in.

    Returns:
        Path to saved checkpoint file.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "config": config,
    }

    path = os.path.join(checkpoint_dir, f"checkpoint_epoch{epoch}.pt")
    torch.save(checkpoint, path)
    print(f"[Checkpoint] Saved: {path}")

    # Also save as 'latest' for easy resume
    latest_path = os.path.join(checkpoint_dir, "checkpoint_latest.pt")
    torch.save(checkpoint, latest_path)

    return path


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """
    Load a training checkpoint.

    Args:
        path:      Path to checkpoint file.
        model:     Model to load state into.
        optimizer: Optional optimizer to restore state.
        device:    Device to map tensors to.

    Returns:
        Dict with metadata: epoch, global_step, train_loss, val_loss, config.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"[Checkpoint] Loaded: {path} (epoch {checkpoint['epoch']}, "
          f"step {checkpoint['global_step']})")

    return {
        "epoch": checkpoint["epoch"],
        "global_step": checkpoint["global_step"],
        "train_loss": checkpoint["train_loss"],
        "val_loss": checkpoint["val_loss"],
        "config": checkpoint.get("config"),
    }


def export_experts(
    model: nn.Module,
    export_dir: str,
) -> None:
    """
    Export each expert as an independent .pt file for Phase 3 streaming.

    Saves expert weights in the layout defined by design.md Section 8:
        expert_layer{L}_id{I}.pt

    Also saves the backbone (everything except expert weights) as backbone.pt
    so the full model can be reconstructed by loading backbone + selected experts.

    Args:
        model:      Trained DWRTransformer model.
        export_dir: Directory to write expert files into.
    """
    os.makedirs(export_dir, exist_ok=True)

    full_state = model.state_dict()
    backbone_state = {}
    expert_count = 0

    for key, param in full_state.items():
        # Expert keys follow pattern: blocks.{L}.moe_ffn.experts.{I}.{param}
        if ".moe_ffn.experts." in key:
            # Parse layer and expert index from key
            # Example: blocks.2.moe_ffn.experts.5.fc1.weight
            parts = key.split(".")
            block_idx = int(parts[1])
            expert_idx = int(parts[4])
            param_name = ".".join(parts[5:])  # e.g., "fc1.weight"

            expert_file = os.path.join(
                export_dir,
                f"expert_layer{block_idx}_id{expert_idx}.pt"
            )

            # Append to existing expert file or create new
            if os.path.exists(expert_file):
                expert_data = torch.load(expert_file, weights_only=True)
            else:
                expert_data = {}

            expert_data[param_name] = param
            torch.save(expert_data, expert_file)
            expert_count += 1
        else:
            backbone_state[key] = param

    # Save backbone (non-expert parameters)
    backbone_path = os.path.join(export_dir, "backbone.pt")
    torch.save(backbone_state, backbone_path)

    # Count unique expert files
    num_files = len([f for f in os.listdir(export_dir) if f.startswith("expert_")])
    print(f"[Export] {num_files} expert files + backbone.pt → {export_dir}/")
    print(f"[Export] Total parameter entries exported: {expert_count}")
