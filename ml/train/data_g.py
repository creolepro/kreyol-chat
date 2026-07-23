"""Workstream G dataloader + LR schedule (nanoGPT-style, seeded & resumable).

The train stream is a flat uint16 `.bin` memmap. Each optimizer step consumes
`total_batch_size` tokens as `total_batch_size / seq_len` sequences drawn at RANDOM
offsets, where the offsets for step `s` are a pure function of `(seed, s)` — so the data
order is (a) identical across depth-sweep runs (depth-invariant) and (b) exactly
reproducible on resume in a fresh container. "Effective tokens" = steps * total_batch;
"epochs" = effective_tokens / unique_train_tokens (an OUTCOME we report, not a target).
"""

from __future__ import annotations

import math

import numpy as np


class Batches:
    def __init__(self, bin_path: str, seq_len: int, seed: int):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len
        self.seed = seed
        self.n = len(self.data)

    def unique_tokens(self) -> int:
        return int(self.n)

    def step_batch(self, step: int, total_batch_size: int, device_batch_size: int):
        """Yield (x, y) int64 microbatches for optimizer `step`. Deterministic in (seed, step).
        Vectorized gather (offsets[:,None] + arange) so the tiny model isn't dataloader-bound."""
        import torch

        n_seq = total_batch_size // self.seq_len
        hi = self.n - (self.seq_len + 1)
        g = np.random.default_rng(self.seed * 1_000_003 + step)
        offsets = g.integers(0, hi, size=n_seq, dtype=np.int64)
        win = np.arange(self.seq_len + 1, dtype=np.int64)   # +1 so we can slice x/y from one gather
        for i in range(0, n_seq, device_batch_size):
            chunk = offsets[i:i + device_batch_size]
            block = self.data[(chunk[:, None] + win[None, :])].astype(np.int64)  # (B, seq_len+1)
            xb = np.ascontiguousarray(block[:, :-1])
            yb = np.ascontiguousarray(block[:, 1:])
            yield torch.from_numpy(xb), torch.from_numpy(yb)


def lr_at(step: int, num_iterations: int, peak_lr: float, min_lr_frac: float, warmup_steps: int) -> float:
    """Linear warmup then cosine decay to `peak_lr * min_lr_frac`."""
    if step < warmup_steps:
        return peak_lr * (step + 1) / max(1, warmup_steps)
    if step >= num_iterations:
        return peak_lr * min_lr_frac
    prog = (step - warmup_steps) / max(1, num_iterations - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * prog))
    return peak_lr * (min_lr_frac + (1.0 - min_lr_frac) * coeff)
