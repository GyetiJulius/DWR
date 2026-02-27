# Phase 2 Results — DWR-Transformer Training

**Date:** February 27, 2026
**Author:** Julius Gyeti
**Status:** Phase 2 Complete

---

## 1. Training Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | WikiText-2-raw-v1 (~2M tokens) |
| Tokenizer | GPT-2 BPE (tiktoken, vocab=50304) |
| d_model | 512 |
| d_ff | 2048 |
| num_layers | 6 |
| num_heads | 8 |
| num_experts | 16 per layer |
| top_k | 2 |
| Total parameters | 259,712,096 |
| Active params/token | ~57.6M |
| Optimizer | AdamW (β₁=0.9, β₂=0.95) |
| Learning rate | 3e-4 (linear warmup 200 steps + cosine decay to 3e-5) |
| Weight decay | 0.1 |
| Batch size | 16 |
| Sequence length | 512 |
| Epochs | 10 |
| Total steps | 2,930 |
| Mixed precision | bfloat16 |
| Hardware | Tesla T4 (16 GB) |
| Time per epoch | ~7 min (~440s average) |
| Total training time | ~72 min |

---

## 2. Training Curve

| Epoch | Train Loss | Train PPL | Val Loss | Val PPL | Status |
|-------|-----------|-----------|----------|---------|--------|
| 1 | — | — | — | — | Warmup |
| 5 | 4.5231 | 92.1 | 5.4439 | 231.4 | Converging |
| 6 | 4.2506 | 70.1 | **5.4181** | **225.5** | **Best val** |
| 7 | 4.0120 | 55.3 | 5.4304 | 228.2 | Val rising |
| 8 | 3.8186 | 45.5 | 5.4468 | 232.0 | Overfitting |
| 9 | 3.6771 | 39.5 | 5.4734 | 238.3 | Overfitting |
| 10 | 3.5831 | 36.0 | 5.4976 | 244.1 | Overfitting |

**Best checkpoint:** Epoch 6, val loss = 5.4181, val PPL = 225.5

---

## 3. Key Findings

### 3.1 Model Trains Successfully

- Loss decreased monotonically on training data (4.52 → 3.58).
- Train perplexity improved 2.5x (92 → 36) over 10 epochs.
- No training instabilities: no NaN gradients, no loss spikes.
- Mixed precision (bfloat16) worked without issues on T4.

### 3.2 Overfitting After Epoch 6

Validation loss diverges from training loss after epoch 6. The gap widens from 0.92 (epoch 5) to 1.91 (epoch 10). This is expected:

- **~260M parameters trained on ~2M tokens** → ~130 params per token.
- The model memorizes training data faster than it learns generalizable patterns.
- This is a data-scale problem, not an architecture defect.

**Mitigation (not applied in Phase 2, documented for future work):**
- Train on larger corpus (OpenWebText, ~9B tokens).
- Early stopping at epoch 6.
- Increase dropout from 0.1 to 0.2.
- Reduce model capacity (fewer experts or smaller d_ff).

### 3.3 Router Behavior

Auxiliary balance loss per layer remained in the 2.0–2.2 range across training (ideal uniform = 1.0). This indicates moderate expert preference but no expert collapse.

**Gate norm spread by layer (epoch 10):**

| Layer | Min Norm | Max Norm | Mean Norm | Spread |
|-------|----------|----------|-----------|--------|
| 0 | 0.942 | 1.045 | 1.004 | 0.102 |
| 1 | 0.822 | 1.015 | 0.939 | 0.193 |
| 2 | 0.646 | 0.936 | 0.817 | 0.291 |
| 3 | 0.589 | 0.929 | 0.769 | 0.340 |
| 4 | 0.665 | 0.924 | 0.787 | 0.259 |
| 5 | 0.607 | 0.896 | 0.723 | 0.289 |

**Observations:**
- Deeper layers show greater routing differentiation (spread increases with depth).
- Layer 3 has the highest spread (0.340), suggesting strongest specialization pressure.
- No expert has collapsed to zero norm — all experts receive tokens.
- Balance loss coefficient λ=0.01 effectively prevents collapse without over-regularizing.

### 3.4 Expert Export

96 individual expert files (6 layers × 16 experts) successfully serialized:
- Each expert file: **8.1 MB** (W1: 512×2048, W2: 2048×512, biases)
- Backbone file: **222 MB** (embeddings, attention, norms, output projection)
- Total export: **992 MB** (matches expected: 96 × 8.1 + 222 ≈ 1000 MB)

File layout matches design.md Section 8:
```
checkpoints/expert_store/
  backbone.pt                   222 MB
  expert_layer0_id0.pt          8.1 MB
  expert_layer0_id1.pt          8.1 MB
  ...
  expert_layer5_id15.pt         8.1 MB
```

---

## 4. Checkpoint Inventory

| File | Size | Contents |
|------|------|----------|
| checkpoint_epoch1.pt – epoch10.pt | 3.0 GB each | Full state (model + optimizer + metadata) |
| checkpoint_latest.pt | 3.0 GB | Copy of epoch 10 |
| expert_store/ | 992 MB total | 96 expert files + backbone |

Total disk usage: ~32 GB (checkpoints) + ~1 GB (expert store).

---

## 5. Validated Properties

- [x] Forward pass shape safety (all tensors verified in Phase 1 tests)
- [x] Router gradient flow (confirmed via backward pass test)
- [x] Cross-entropy + auxiliary balance loss convergence
- [x] Learning rate warmup + cosine decay schedule
- [x] Gradient clipping (max_norm=1.0)
- [x] Weight decay separation (no decay on biases, norms, embeddings)
- [x] Mixed precision training (bfloat16 on T4)
- [x] Periodic validation with perplexity reporting
- [x] Full checkpoint save/load
- [x] Individual expert serialization (Phase 3 ready)

---

## 6. Phase 3 Readiness

Phase 2 outputs everything Phase 3 needs:

1. **Independently serialized experts** — each 8.1 MB, loadable one at a time.
2. **Separate backbone** — attention, embeddings, norms in one file.
3. **Trained router weights** — included in backbone, determines which experts to load.

Phase 3 scope (design.md Sections 8, 9):
- Expert Store: disk/CPU memory-mapped access
- GPU Cache Manager: LRU eviction policy
- Weight Prefetch Engine: frequency-based caching
- Inference with dynamic expert loading (only top-k experts in VRAM)

---

## 7. Known Limitations

1. **Overfitting** — WikiText-2 is too small for 260M parameters. Expected.
2. **No capacity limiting** — no token drop if too many route to one expert.
3. **Expert dispatch loop** — O(num_experts) per layer, not parallelized.
4. **No early stopping** — all 10 epochs run; best model is at epoch 6.
5. **Checkpoint size** — 3 GB per checkpoint includes optimizer state (Adam moments double memory).

---

## 8. Conclusion

Phase 2 validates that the DWR-Transformer architecture trains correctly. The routing mechanism learns meaningful expert preferences, the balance loss prevents collapse, and all experts receive non-trivial token assignments. The model successfully converges on a language modeling task. The overfitting is wholly attributable to data scale, not architectural failure.

The individually serialized expert files confirm that the weight partitioning strategy from design.md Section 8 is viable. Each expert is independently loadable, making the Phase 3 dynamic weight retrieval system feasible.
