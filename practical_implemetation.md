Perfect. We’re building the core correctly first.

Below is a **clean, production-structured PyTorch skeleton** for:

* `Expert`
* `Router`
* `DWRBlock` (MoE Transformer block)

This is **static MoE first** (no streaming yet). That’s intentional.

---

# 📁 `models/expert.py`

Each expert is just a standard Transformer FFN.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """
    Standard Transformer Feed-Forward Network used as one expert.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

Nothing fancy yet. Keep it simple.

---

# 📁 `models/router.py`

Top-K routing (token-level).

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Router(nn.Module):
    """
    Top-k router for Mixture-of-Experts.
    """

    def __init__(self, d_model: int, num_experts: int, top_k: int = 2):
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k

        self.gate = nn.Linear(d_model, num_experts)

    def forward(self, x: torch.Tensor):
        """
        x: (batch, seq_len, d_model)

        Returns:
            topk_indices: (batch, seq_len, top_k)
            topk_scores:  (batch, seq_len, top_k)
        """

        logits = self.gate(x)  # (B, S, num_experts)

        # Softmax over experts
        probs = F.softmax(logits, dim=-1)

        # Select top-k experts per token
        topk_scores, topk_indices = torch.topk(
            probs,
            self.top_k,
            dim=-1
        )

        return topk_indices, topk_scores
```

This gives you:

* Which experts to call
* How much to weight them

---

# 📁 `models/dwr_block.py`

This is your actual MoE Transformer block.

We will assume attention is handled elsewhere and focus only on the FFN replacement.

```python
import torch
import torch.nn as nn

from models.expert import Expert
from models.router import Router


class DWRBlock(nn.Module):
    """
    Transformer block with Mixture-of-Experts FFN.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        top_k: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        self.router = Router(d_model, num_experts, top_k)

        self.experts = nn.ModuleList([
            Expert(d_model, d_ff, dropout)
            for _ in range(num_experts)
        ])

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.num_experts = num_experts
        self.top_k = top_k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        """

        residual = x

        # Router
        topk_indices, topk_scores = self.router(x)

        B, S, D = x.shape

        # Flatten tokens
        x_flat = x.view(-1, D)  # (B*S, D)
        topk_indices = topk_indices.view(-1, self.top_k)
        topk_scores = topk_scores.view(-1, self.top_k)

        output = torch.zeros_like(x_flat)

        # For each expert, process only assigned tokens
        for expert_id, expert in enumerate(self.experts):

            # Find tokens assigned to this expert
            mask = (topk_indices == expert_id)

            if not mask.any():
                continue

            token_indices = mask.any(dim=1).nonzero(as_tuple=True)[0]

            if token_indices.numel() == 0:
                continue

            expert_input = x_flat[token_indices]

            expert_output = expert(expert_input)

            # Weight by routing probability
            weights = topk_scores[token_indices]
            weights = weights.sum(dim=1, keepdim=True)

            output[token_indices] += expert_output * weights

        output = output.view(B, S, D)

        x = residual + self.dropout(output)
        x = self.layer_norm(x)

        return x
```

---

# ⚠️ Important Notes

This implementation is:

* Correct conceptually
* Not optimized
* Not load-balanced
* Not capacity-limited
* Not distributed

But it proves routing works.

That’s what we need first.

---

# 🧪 Your Immediate Test

Create a simple test:

```python
B = 2
S = 4
D = 256

block = DWRBlock(
    d_model=256,
    d_ff=1024,
    num_experts=8,
    top_k=2
)

x = torch.randn(B, S, D)

out = block(x)

print(out.shape)
```

Expected output:

```
torch.Size([2, 4, 256])
```

If this works:
You have a working static MoE block.


