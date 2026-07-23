"""Bits-per-byte for Model C — byte-identical policy to the Workstream-D base-model
probe (`probe/modal_app._score_bpb`), so Model C and the 3-4B bases are comparable ON
THE SAME SLICES: per-doc, BOS prepended as context and never scored, docs split into
`max_len` windows each re-seeded with BOS, denominator = doc UTF-8 bytes, bits = nats/ln2.

`encode` is our tiktoken Encoding's `encode_ordinary` (the SAME tokenizer that produced
train.bin) — Model C's BPB is thus its own-tokenizer NLL over shared bytes, exactly like
the scorecard.
"""

from __future__ import annotations

import math


def score_bpb(model, encode, texts, max_len: int, bos_id: int) -> dict:
    import torch
    import torch.nn.functional as F

    device = next(model.parameters()).device
    total_bits, total_bytes, n_chunks = 0.0, 0, 0
    with torch.inference_mode():
        for text in texts:
            total_bytes += len(text.encode("utf-8"))
            ids = encode(text)
            for s in range(0, len(ids), max_len):
                window = ids[s:s + max_len]
                if not window:
                    continue
                inp = torch.tensor([[bos_id] + window], device=device)
                logits = model(inp).logits[0, :-1].float()      # predicts inp[1:]
                tgt = inp[0, 1:]
                nll_nats = F.cross_entropy(logits, tgt, reduction="sum")
                total_bits += float(nll_nats) / math.log(2)
                n_chunks += 1
    return {"nll_bits": total_bits, "bytes": total_bytes, "n_chunks": n_chunks,
            "n_docs": len(texts),
            "bpb": (total_bits / total_bytes) if total_bytes else None}
