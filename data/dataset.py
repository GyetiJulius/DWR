"""Dataset module: WikiText-103 loading, tokenization, and batching.

Uses the HuggingFace `datasets` library to download WikiText-103-raw-v1
and `tiktoken` for GPT-2 BPE tokenization.

The dataset is packed into contiguous token sequences of length max_seq_len
for standard autoregressive language modeling:
    input  = tokens[i   : i + max_seq_len]
    target = tokens[i+1 : i + max_seq_len + 1]

This avoids padding and maximizes GPU utilization.

Design doc context:
    Phase 1 training (design.md Section 10):
    "Train normally with all experts loaded. No streaming during training."
"""

import os
from typing import Tuple

import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader


class TokenDataset(Dataset):
    """
    Contiguous token sequence dataset for autoregressive LM training.

    Given a flat 1-D tensor of token IDs, yields (input, target) pairs
    where target is input shifted by one position.

    Parameters
    ----------
    tokens : torch.Tensor
        1-D long tensor of token IDs.
    seq_len : int
        Sequence length for each sample.
    """

    def __init__(self, tokens: torch.Tensor, seq_len: int) -> None:
        self.tokens = tokens
        self.seq_len = seq_len
        # Number of complete sequences we can extract.
        # We need seq_len + 1 tokens per sample (input + shifted target).
        self.num_samples = (len(tokens) - 1) // seq_len

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        end = start + self.seq_len

        x = self.tokens[start:end]        # (seq_len,)
        y = self.tokens[start + 1:end + 1]  # (seq_len,)
        return x, y


def _tokenize_split(
    split_text: str,
    enc: tiktoken.Encoding,
) -> torch.Tensor:
    """
    Tokenize a text split into a flat 1-D long tensor.

    Args:
        split_text: Raw text string for one dataset split.
        enc: tiktoken encoding instance.

    Returns:
        1-D torch.long tensor of token IDs.
    """
    # tiktoken is fast — encodes ~1M tokens/sec on CPU.
    # allowed_special="all" handles any special tokens in WikiText.
    token_ids = enc.encode(split_text, allowed_special="all")
    return torch.tensor(token_ids, dtype=torch.long)


def build_dataloaders(
    max_seq_len: int,
    batch_size: int,
    data_cache_dir: str = "data_cache",
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, tiktoken.Encoding]:
    """
    Build train and validation DataLoaders from WikiText-103.

    Downloads the dataset on first run and caches tokenized tensors
    to disk for subsequent runs.

    Args:
        max_seq_len:    Sequence length per sample.
        batch_size:     Batch size for both train and val.
        data_cache_dir: Directory for caching tokenized tensors.
        num_workers:    DataLoader worker processes.

    Returns:
        train_loader: Training DataLoader
        val_loader:   Validation DataLoader
        enc:          tiktoken encoding (for decode during inference)
    """
    from datasets import load_dataset

    enc = tiktoken.get_encoding("gpt2")

    os.makedirs(data_cache_dir, exist_ok=True)

    train_cache = os.path.join(data_cache_dir, "train_tokens.pt")
    val_cache = os.path.join(data_cache_dir, "val_tokens.pt")

    if os.path.exists(train_cache) and os.path.exists(val_cache):
        print("[Data] Loading cached tokenized tensors...")
        train_tokens = torch.load(train_cache, weights_only=True)
        val_tokens = torch.load(val_cache, weights_only=True)
    else:
        print("[Data] Downloading and tokenizing WikiText-103...")
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

        # Concatenate all text in each split into one string.
        # WikiText stores one paragraph per row.
        train_text = "\n".join(dataset["train"]["text"])
        val_text = "\n".join(dataset["validation"]["text"])

        train_tokens = _tokenize_split(train_text, enc)
        val_tokens = _tokenize_split(val_text, enc)

        torch.save(train_tokens, train_cache)
        torch.save(val_tokens, val_cache)

        print(f"[Data] Train tokens: {len(train_tokens):,}")
        print(f"[Data] Val tokens:   {len(val_tokens):,}")

    train_dataset = TokenDataset(train_tokens, max_seq_len)
    val_dataset = TokenDataset(val_tokens, max_seq_len)

    print(f"[Data] Train samples: {len(train_dataset):,}  "
          f"Val samples: {len(val_dataset):,}  "
          f"Seq len: {max_seq_len}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, enc
