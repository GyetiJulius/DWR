# DWR-Transformer (Dynamic Weight Retrieval Transformer)

## Version: v1 Research Blueprint

Author: Julius Gyeti
Status: Architecture Design Phase

---

# 1. Vision

The DWR-Transformer is a sparse, modular transformer architecture designed to:

* Reduce active VRAM usage
* Enable dynamic weight streaming
* Support runtime-aware expert retrieval
* Provide a research foundation for parameter-externalized neural networks

Core Idea:

> Separate model structure from expert weight storage and retrieve only required weight blocks during inference.

---

# 2. High-Level System Design

The system is divided into two major layers:

## 2.1 Model Layer (Mathematical Graph)

Defines computation and routing behavior.

Components:

* Token Embedding
* Multi-Head Self Attention
* Router Network
* Modular Expert Layer (MEL)
* LayerNorm + Residuals

## 2.2 Runtime Layer (Execution Engine)

Handles memory and weight management.

Components:

* Expert Store (Disk / CPU Memory)
* GPU Cache Manager
* Weight Prefetch Engine
* Memory Eviction Policy
* Execution Scheduler

---

# 3. Transformer Block Redesign

## 3.1 Standard Block (Reference)

x → Self-Attention → Add & Norm
→ Dense FFN → Add & Norm

The FFN contains the majority of parameters.

## 3.2 DWR Block

x → Self-Attention → Add & Norm
→ Router(x)
→ Select Experts
→ Retrieve Experts
→ Weighted Expert Aggregation
→ Add & Norm

---

# 4. Modular Expert Layer (MEL)

## 4.1 Expert Definition

Each expert i contains:

W1_i: (d_model → d_ff)
W2_i: (d_ff → d_model)
Activation: GeLU

Expert_i(x) = W2_i(GeLU(W1_i(x)))

---

# 5. Architectural Decisions (Locked for v1)

## 5.1 Number of Experts Per Layer

16 experts per transformer layer.

Reasoning:

* Good balance between sparsity and routing flexibility
* Allows meaningful specialization
* Not too large for prototype testing

## 5.2 Top-k Selection

Top-2 experts per token.

Reasoning:

* Improves stability vs top-1
* Reduces expert collapse
* Maintains sparsity

## 5.3 Expert Size

Fixed-size experts.

Reasoning:

* Simplifies implementation
* Easier memory planning
* Stable training behavior

---

# 6. Mathematical Forward Pass

Given token representation x:

1. Compute routing logits:

r = Linear(x)  → shape: (N_experts)

2. Apply softmax:

g = Softmax(r)

3. Select top-2 experts:

Indices = TopK(g, k=2)

4. Compute output:

y = Σ (g_i * Expert_i(x)) for selected experts

---

# 7. Router Network

## 7.1 Structure

Single linear layer:

Router(x): Linear(d_model → 16)

## 7.2 Load Balancing Loss

Total loss:

L_total = L_task + λ * L_balance

Where L_balance encourages uniform expert usage.

---

# 8. Weight Partition Strategy

Instead of monolithic model weights, experts are stored independently:

/expert_store/
expert_layer1_id0.pt
expert_layer1_id1.pt
...
expert_layerL_id15.pt

Each expert is:

* Independently serialized
* Memory-mappable
* Loadable on demand

---

# 9. Runtime Architecture

## 9.1 Expert Store

Responsibilities:

* Load expert weights from disk or CPU
* Provide memory-mapped access
* Support async loading

---

## 9.2 GPU Cache Manager

Keeps active experts in GPU memory.

Policy: LRU (Least Recently Used) for v1.

Logic:

IF expert in GPU cache:
Use directly
ELSE:
Evict least recently used expert
Load requested expert

---

## 9.3 Prefetch Strategy (v1)

Simple frequency-based caching.

Future versions:

* Semantic cluster-based prefetch
* Predictive expert loading

---

# 10. Training Strategy

Phase 1:

* Train normally with all experts loaded
* No streaming during training

Phase 2:

* Deploy streaming only during inference

Future research:

* Training under simulated memory constraints

---

# 11. Prototype Configuration (v1)

Model Size Target: ~50M parameters

Suggested Setup:

* Layers: 6
* d_model: 512
* d_ff per expert: 2048
* Experts per layer: 16
* Top-k: 2

Goal:
Validate streaming feasibility before scaling.

---

# 12. Expected Benefits

* Reduced active VRAM footprint
* Conditional computation
* Modular parameter scaling
* Hardware-aware optimization potential

---

# 13. Risks

* Weight transfer latency
* Cache thrashing
* Router instability
* Memory fragmentation

---

# 14. Future Research Directions

* Block-sparse matrix slicing
* Attention-layer modularization
* Predictive routing
* Expert specialization by semantic domain
* Distributed expert storage

---

# 15. Milestones

Milestone 1:
Implement modular FFN with router.

Milestone 2:
Implement disk-backed expert loading.

Milestone 3:
Add GPU cache manager.

Milestone 4:
Benchmark against dense baseline.

Milestone 5:
Scale to 500M+ parameters.

---

# End of Architecture Plan (v1)
