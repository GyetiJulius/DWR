"""
Expert Predictor: Predict which experts the next layer will need.

The core novel contribution of the DWR pipeline. While layer L computes,
the predictor uses layer L's routing decisions to forecast which experts
layer L+1 will select, enabling asynchronous prefetch that overlaps
disk I/O with GPU compute.

Three strategies implemented:

1. TransitionMatrix (calibration-based):
   Learns P(expert_j at L+1 | expert_i at L) from a calibration pass.
   At inference, given experts selected at L, looks up most likely at L+1.
   Best accuracy, requires a calibration step.

2. Heuristic (zero-cost):
   Assumes the same experts tend to repeat across adjacent layers.
   Prefetches the same expert IDs plus top-k most common experts.
   No calibration needed, works immediately.

3. Oracle (upper bound):
   Perfect prediction — knows exactly which experts will be selected.
   Used only for benchmarking to measure the prediction accuracy gap.

Why this is novel:
   Existing offloading systems (FlexGen, DeepSpeed, Mixtral Offloading)
   treat models as black boxes and page weights generically. DWR is the
   first to feed ROUTING DECISIONS back into the STORAGE SYSTEM, creating
   a feedback loop between the model's computation graph and its memory
   manager. This is architecture-runtime co-design.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Set, Tuple
import json
import os

import torch
import torch.nn as nn


# Type aliases
ExpertKey = Tuple[int, int]  # (layer_idx, expert_id)


class ExpertPredictor(ABC):
    """Base class for expert prediction strategies."""

    @abstractmethod
    def predict(
        self,
        current_layer: int,
        selected_experts: Set[int],
        num_predict: int,
    ) -> Set[int]:
        """
        Predict which experts the next layer will need.

        Args:
            current_layer:    Index of the layer that just routed.
            selected_experts: Expert IDs selected at current_layer.
            num_predict:      How many experts to predict for next layer.

        Returns:
            Set of predicted expert IDs for layer (current_layer + 1).
        """
        ...

    @abstractmethod
    def name(self) -> str:
        """Human-readable name for logging."""
        ...


class TransitionMatrixPredictor(ExpertPredictor):
    """
    Calibration-based predictor using expert co-occurrence statistics.

    For each pair of adjacent layers (L, L+1), maintains a transition
    matrix T[L] of shape (num_experts, num_experts) where:

        T[L][i][j] = P(expert j selected at L+1 | expert i selected at L)

    Built from a single calibration pass over validation data.
    At inference, given the set of experts selected at layer L, we compute
    the aggregate prediction score for each expert at L+1:

        score[j] = sum over i in selected: T[L][i][j]

    Then pick the top-num_predict experts.

    This captures cross-layer co-occurrence patterns: if expert 3 at layer 2
    frequently co-occurs with expert 7 at layer 3, the predictor learns this.
    """

    def __init__(self, num_layers: int, num_experts: int) -> None:
        self.num_layers = num_layers
        self.num_experts = num_experts

        # Transition matrices: one per adjacent layer pair
        # Shape: (num_layers - 1, num_experts, num_experts)
        # T[l][i][j] = count of co-occurrences, normalized to probabilities
        self.transitions: List[torch.Tensor] = [
            torch.zeros(num_experts, num_experts)
            for _ in range(num_layers - 1)
        ]

        self._calibrated = False

    def record(
        self,
        layer_idx: int,
        experts_this_layer: Set[int],
        experts_next_layer: Set[int],
    ) -> None:
        """
        Record a co-occurrence observation during calibration.

        For each expert i selected at layer_idx and each expert j selected
        at layer_idx + 1, increment T[layer_idx][i][j].

        Args:
            layer_idx:          Current layer index.
            experts_this_layer: Expert IDs selected at layer_idx.
            experts_next_layer: Expert IDs selected at layer_idx + 1.
        """
        if layer_idx >= self.num_layers - 1:
            return

        T = self.transitions[layer_idx]
        for i in experts_this_layer:
            for j in experts_next_layer:
                T[i, j] += 1.0

    def record_batch(
        self,
        layer_idx: int,
        experts_this: torch.Tensor,
        experts_next: torch.Tensor,
    ) -> None:
        """
        Record per-token co-occurrences from batched routing indices.

        This is the efficient vectorized version of record() used during
        calibration. For each token t, for each expert i selected at
        layer_idx and each expert j at layer_idx+1, increment T[i][j].

        This produces SPARSE, MEANINGFUL transition matrices because each
        token only selects top_k (e.g. 2) experts — not all 16.

        Args:
            layer_idx:    Current layer index.
            experts_this: (num_tokens, top_k) expert indices at layer_idx.
            experts_next: (num_tokens, top_k) expert indices at layer_idx + 1.
        """
        if layer_idx >= self.num_layers - 1:
            return

        T = self.transitions[layer_idx]
        num_tokens, top_k = experts_this.shape

        # Vectorized: for each token, all (i, j) pairs between its L and L+1 experts
        # Move to CPU for counting (transition matrices are CPU tensors)
        this_cpu = experts_this.cpu()
        next_cpu = experts_next.cpu()
        for k1 in range(top_k):
            for k2 in range(top_k):
                i_indices = this_cpu[:, k1].long()  # (T,)
                j_indices = next_cpu[:, k2].long()   # (T,)
                # Vectorized scatter-add using flat indexing
                flat_idx = i_indices * self.num_experts + j_indices
                counts = torch.bincount(flat_idx, minlength=self.num_experts ** 2)
                T += counts.float().reshape(self.num_experts, self.num_experts)

    def finalize(self) -> None:
        """
        Normalize co-occurrence counts into transition probabilities.

        Call this after all calibration observations have been recorded.
        """
        for l in range(len(self.transitions)):
            T = self.transitions[l]
            row_sums = T.sum(dim=1, keepdim=True)
            # Avoid division by zero for unused experts
            row_sums = row_sums.clamp(min=1.0)
            self.transitions[l] = T / row_sums

        self._calibrated = True

    def predict(
        self,
        current_layer: int,
        selected_experts: Set[int],
        num_predict: int,
    ) -> Set[int]:
        """
        Predict next layer's experts using transition matrix.

        Aggregates transition probabilities across all selected experts
        at the current layer, then picks the top-scoring experts.
        """
        if not self._calibrated:
            # Fallback: return same experts (heuristic)
            return set(list(selected_experts)[:num_predict])

        if current_layer >= self.num_layers - 1:
            return set()

        T = self.transitions[current_layer]

        # Aggregate: score[j] = sum_i T[i][j] for i in selected_experts
        scores = torch.zeros(self.num_experts)
        for i in selected_experts:
            scores += T[i]

        # Pick top-num_predict
        _, top_indices = scores.topk(min(num_predict, self.num_experts))
        return set(top_indices.tolist())

    def save(self, path: str) -> None:
        """Save calibrated transition matrices to disk."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        data = {
            "num_layers": self.num_layers,
            "num_experts": self.num_experts,
            "transitions": [T.tolist() for T in self.transitions],
            "calibrated": self._calibrated,
        }
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "TransitionMatrixPredictor":
        """Load calibrated predictor from disk."""
        with open(path) as f:
            data = json.load(f)

        predictor = cls(data["num_layers"], data["num_experts"])
        predictor.transitions = [
            torch.tensor(T) for T in data["transitions"]
        ]
        predictor._calibrated = data["calibrated"]
        return predictor

    def accuracy_report(self) -> str:
        """Report transition matrix coverage statistics."""
        lines = [f"TransitionMatrixPredictor (calibrated={self._calibrated})"]
        for l, T in enumerate(self.transitions):
            nonzero = (T > 0).sum().item()
            total = T.numel()
            sparsity = 1.0 - nonzero / total
            # Entropy of each row
            entropy = -(T * T.clamp(min=1e-10).log()).sum(dim=1)
            avg_entropy = entropy.mean().item()
            lines.append(
                f"  Layer {l}→{l+1}: "
                f"nonzero={nonzero}/{total} "
                f"sparsity={sparsity:.1%} "
                f"avg_entropy={avg_entropy:.2f}"
            )
        return "\n".join(lines)

    def name(self) -> str:
        return "transition-matrix"


class HeuristicPredictor(ExpertPredictor):
    """
    Zero-cost heuristic predictor.

    Prediction rule:
        1. Start with the same experts selected at the current layer
           (expert persistence across layers is common in MoE models).
        2. If fewer than num_predict, pad with globally popular experts.

    No calibration needed. Works out of the box.
    Serves as the baseline for measuring transition matrix improvement.
    """

    def __init__(self, num_experts: int, popular_experts: List[int] = None) -> None:
        self.num_experts = num_experts
        # Default popular experts: first few (will be overridden by calibration)
        self.popular_experts = popular_experts or list(range(min(4, num_experts)))

    def set_popular_experts(self, experts: List[int]) -> None:
        """Set globally popular experts (from calibration statistics)."""
        self.popular_experts = experts

    def predict(
        self,
        current_layer: int,
        selected_experts: Set[int],
        num_predict: int,
    ) -> Set[int]:
        """
        Predict: same experts + popular fillup.
        """
        predicted = set(selected_experts)

        # Fill with popular experts if we need more
        for e in self.popular_experts:
            if len(predicted) >= num_predict:
                break
            predicted.add(e)

        # Limit to num_predict
        return set(list(predicted)[:num_predict])

    def name(self) -> str:
        return "heuristic"


class OraclePredictor(ExpertPredictor):
    """
    Perfect predictor — knows the future.

    Used ONLY for benchmarking to establish the upper bound on prediction
    accuracy and prefetch hit rate. At inference time, the oracle is fed
    the actual routing decisions from the next layer (cheating).

    This measures: "How good could we possibly be if prediction were perfect?"
    """

    def __init__(self) -> None:
        # Will be set externally before each layer's predict() call
        self._oracle_experts: Set[int] = set()

    def set_oracle(self, experts: Set[int]) -> None:
        """Set the actual experts that will be needed (cheating)."""
        self._oracle_experts = experts

    def predict(
        self,
        current_layer: int,
        selected_experts: Set[int],
        num_predict: int,
    ) -> Set[int]:
        """Return the exact experts that will be needed."""
        return set(list(self._oracle_experts)[:num_predict])

    def name(self) -> str:
        return "oracle"
