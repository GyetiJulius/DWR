"""
Expert Store: Disk-backed expert weight loader.

Loads individual expert .pt files from the export directory produced
by Phase 2's export_experts(). Each file contains a single expert's
state_dict (fc1.weight, fc1.bias, fc2.weight, fc2.bias).

Design doc reference: Section 9.1
    "Load expert weights from disk or CPU.
     Provide memory-mapped access.
     Support async loading."

Phase 3 implementation:
    - Synchronous loading from disk (simple, correct first)
    - Optional CPU staging: experts loaded to CPU first, then .to(device)
    - File layout: expert_layer{L}_id{I}.pt

Memory-mapping note:
    torch.load with mmap=True requires safetensors or compatible format.
    For v1, we use standard torch.load which reads into CPU RAM then
    transfers to target device. This is correct and sufficient for
    prototype validation. Production would use safetensors + mmap.
"""

import os
from typing import Dict, Tuple

import torch
import torch.nn as nn

from models.expert import Expert


class ExpertStore:
    """
    Disk-backed expert weight storage and retrieval.

    Manages a directory of individually serialized expert files.
    Loads expert weights into Expert modules on demand.

    Parameters
    ----------
    store_dir : str
        Path to directory containing expert_layer{L}_id{I}.pt files.
    d_model : int
        Expert input/output dimensionality.
    d_ff : int
        Expert hidden dimensionality.
    num_layers : int
        Number of transformer layers (for validation).
    num_experts : int
        Number of experts per layer (for validation).
    device : torch.device
        Target device for loaded experts (usually cuda).
    """

    def __init__(
        self,
        store_dir: str,
        d_model: int,
        d_ff: int,
        num_layers: int,
        num_experts: int,
        device: torch.device,
    ) -> None:
        self.store_dir = store_dir
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.device = device

        # Validate store directory
        self._validate_store()

        # Track load statistics for diagnostics
        self.load_count: int = 0
        self.total_bytes_loaded: int = 0

    def _validate_store(self) -> None:
        """Verify that all expected expert files exist."""
        missing = []
        for layer in range(self.num_layers):
            for expert_id in range(self.num_experts):
                path = self._expert_path(layer, expert_id)
                if not os.path.exists(path):
                    missing.append(path)

        if missing:
            raise FileNotFoundError(
                f"Expert store missing {len(missing)} files. "
                f"First missing: {missing[0]}. "
                f"Run Phase 2 training and export_experts() first."
            )

        backbone_path = os.path.join(self.store_dir, "backbone.pt")
        if not os.path.exists(backbone_path):
            raise FileNotFoundError(
                f"Backbone file not found: {backbone_path}. "
                f"Run export_experts() first."
            )

    def _expert_path(self, layer_idx: int, expert_id: int) -> str:
        """Get filesystem path for a specific expert."""
        return os.path.join(
            self.store_dir,
            f"expert_layer{layer_idx}_id{expert_id}.pt"
        )

    def load_expert(
        self, layer_idx: int, expert_id: int
    ) -> Expert:
        """
        Load a single expert from disk into an Expert module on target device.

        This is the core retrieval operation (design.md Section 9.1).
        The expert is loaded to CPU first, then transferred to the target
        device to avoid GPU memory fragmentation from direct loading.

        Args:
            layer_idx: Transformer layer index (0-based).
            expert_id: Expert index within the layer.

        Returns:
            Expert module with loaded weights, on self.device, in eval mode.
        """
        path = self._expert_path(layer_idx, expert_id)

        # Load state dict to CPU first (avoids GPU memory spikes)
        state_dict = torch.load(path, map_location="cpu", weights_only=True)

        # Create expert module and load weights
        expert = Expert(self.d_model, self.d_ff, dropout=0.0)
        expert.load_state_dict(state_dict)

        # Transfer to target device and set to eval mode
        expert = expert.to(self.device)
        expert.eval()

        # Track statistics
        self.load_count += 1
        file_size = os.path.getsize(path)
        self.total_bytes_loaded += file_size

        return expert

    def load_backbone(self) -> Dict[str, torch.Tensor]:
        """
        Load the backbone state dict (everything except expert weights).

        Returns:
            State dict for non-expert parameters (embeddings, attention,
            norms, output projection, router weights).
        """
        path = os.path.join(self.store_dir, "backbone.pt")
        return torch.load(path, map_location="cpu", weights_only=True)

    def get_expert_file_size(self, layer_idx: int, expert_id: int) -> int:
        """Get file size in bytes for a specific expert."""
        return os.path.getsize(self._expert_path(layer_idx, expert_id))

    def get_stats(self) -> Dict[str, float]:
        """Return loading statistics."""
        return {
            "load_count": self.load_count,
            "total_mb_loaded": self.total_bytes_loaded / (1024 * 1024),
        }
