"""Workstream I — render conversations → packed token/mask bins for midtrain + SFT.

Format (nanochat chat template over kreyol-bpe's reserved specials):
    <|bos|> [ <|user_start|> U <|user_end|> <|assistant_start|> A <|assistant_end|> ]…
BOS opens the sequence (matching the Part-0 add_bos_token=true deployment path). The
loss MASK is emitted per token:
  • midtrain ("full")     — every target contributes (learn the format + keep the LM);
  • sft ("response")      — ONLY assistant-content tokens + the closing <|assistant_end|>
                            contribute (learn to answer, and to stop), everything else masked.

Packing is flat concatenation with random-offset windows (identical to the base-model
data pipeline data_g), so a window is (token.bin, mask.bin) sliced in lockstep. Outputs
under data/train_work/chat/data/ (git-ignored), uploaded to the Modal Volume.

Run:  uv run python -m train.chat_tokenize            # build midtrain + sft bins
      uv run python -m train.chat_tokenize --layers midtrain
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from . import chat_config as C
from . import config as F
from . import llama_config as G
from . import tokenize_g as TG


def _special_ids(enc):
    return {
        "bos": enc.encode_single_token(C.BOS),
        "user_start": enc.encode_single_token(C.USER_START),
        "user_end": enc.encode_single_token(C.USER_END),
        "assistant_start": enc.encode_single_token(C.ASSISTANT_START),
        "assistant_end": enc.encode_single_token(C.ASSISTANT_END),
    }


def render_conversation(messages, enc, sp, loss_mask="response"):
    """→ (ids, mask). mask[i]=1 means token i counts as a TARGET in the loss.
    'full'     → all real tokens count (bos excluded as a target seed).
    'response' → only assistant content + <|assistant_end|> count."""
    ids = [sp["bos"]]
    mask = [0]                                  # bos is context, never a target
    for m in messages:
        content_ids = enc.encode_ordinary(m["content"])
        if m["role"] == "assistant":
            ids.append(sp["assistant_start"]); mask.append(1 if loss_mask == "full" else 0)
            ids.extend(content_ids);           mask.extend([1] * len(content_ids))
            ids.append(sp["assistant_end"]);   mask.append(1)   # learn to STOP (both modes)
        else:
            ids.append(sp["user_start"]);      mask.append(1 if loss_mask == "full" else 0)
            ids.extend(content_ids);           mask.extend([1 if loss_mask == "full" else 0] * len(content_ids))
            ids.append(sp["user_end"]);        mask.append(1 if loss_mask == "full" else 0)
    return ids, mask


def build_bins(jsonl_path, prefix, loss_mask, seq_len=2048, val_frac=0.02, seed=C.DL_SEED):
    """Pack all conversations in jsonl_path into {prefix}.bin (uint16 tokens) +
    {prefix}.mask.bin (uint8) + a small held-out {prefix}_val.* for periodic val loss."""
    enc, _ = TG._encoding()
    sp = _special_ids(enc)
    rng = np.random.default_rng(seed)

    convos = [json.loads(l) for l in open(jsonl_path, encoding="utf-8") if l.strip()]
    rng.shuffle(convos)
    n_val = max(8, int(len(convos) * val_frac))
    val_set, train_set = convos[:n_val], convos[n_val:]

    stats = {"prefix": prefix, "loss_mask": loss_mask, "n_convos": len(convos),
             "n_train_convos": len(train_set), "n_val_convos": len(val_set),
             "truncated": 0, "empty_response": 0}

    def _pack(subset):
        tok_parts, mask_parts, n_resp_tokens = [], [], 0
        for c in subset:
            ids, mask = render_conversation(c["messages"], enc, sp, loss_mask)
            if len(ids) > seq_len:                         # truncate over-long convos
                ids, mask = ids[:seq_len], mask[:seq_len]
                stats["truncated"] += 1
            if sum(mask) == 0:                             # nothing to learn from → skip
                stats["empty_response"] += 1
                continue
            tok_parts.append(np.array(ids, dtype=np.uint16))
            mask_parts.append(np.array(mask, dtype=np.uint8))
            n_resp_tokens += int(sum(mask))
        toks = np.concatenate(tok_parts) if tok_parts else np.zeros(0, np.uint16)
        masks = np.concatenate(mask_parts) if mask_parts else np.zeros(0, np.uint8)
        return toks, masks, n_resp_tokens

    os.makedirs(C.CHAT_BUNDLE, exist_ok=True)
    tr_tok, tr_mask, tr_resp = _pack(train_set)
    va_tok, va_mask, va_resp = _pack(val_set)
    tr_tok.tofile(os.path.join(C.CHAT_BUNDLE, f"{prefix}.bin"))
    tr_mask.tofile(os.path.join(C.CHAT_BUNDLE, f"{prefix}.mask.bin"))
    va_tok.tofile(os.path.join(C.CHAT_BUNDLE, f"{prefix}_val.bin"))
    va_mask.tofile(os.path.join(C.CHAT_BUNDLE, f"{prefix}_val.mask.bin"))

    stats.update({
        "train_tokens": int(tr_tok.size), "train_loss_tokens": tr_resp,
        "val_tokens": int(va_tok.size), "val_loss_tokens": va_resp,
        "loss_token_frac": round(tr_resp / max(1, tr_tok.size), 3),
        "train_mb": round(tr_tok.size * 2 / 1e6, 1),
    })
    return stats


def build_all(layers=("midtrain", "sft")):
    os.makedirs(C.CHAT_WORK, exist_ok=True)
    manifest = {"snapshot_date": C.SNAPSHOT_DATE, "seq_len": C.MIDTRAIN["max_seq_len"],
                "bins": {}}
    if "midtrain" in layers:
        manifest["bins"]["midtrain"] = build_bins(
            C.LAYER1_JSONL, "midtrain", C.MIDTRAIN["loss_mask"], C.MIDTRAIN["max_seq_len"])
        print(f"[chat_tokenize] midtrain: {json.dumps(manifest['bins']['midtrain'])}")
    if "sft" in layers:
        manifest["bins"]["sft"] = build_bins(
            C.LAYER3_JSONL, "sft", C.SFT["loss_mask"], C.SFT["max_seq_len"])
        print(f"[chat_tokenize] sft: {json.dumps(manifest['bins']['sft'])}")
    with open(os.path.join(C.CHAT_WORK, "chat_bins_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    return manifest


# --- dataloader used on Modal (token.bin + mask.bin in lockstep) ---------------

class ChatBatches:
    def __init__(self, bin_path, mask_path, seq_len, seed):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.mask = np.memmap(mask_path, dtype=np.uint8, mode="r")
        self.seq_len = seq_len
        self.seed = seed
        self.n = len(self.data)

    def unique_tokens(self):
        return int(self.n)

    def step_batch(self, step, total_batch_size, device_batch_size):
        import torch
        n_seq = total_batch_size // self.seq_len
        hi = self.n - (self.seq_len + 1)
        g = np.random.default_rng(self.seed * 1_000_003 + step)
        offsets = g.integers(0, max(1, hi), size=n_seq, dtype=np.int64)
        win = np.arange(self.seq_len + 1, dtype=np.int64)
        for i in range(0, n_seq, device_batch_size):
            chunk = offsets[i:i + device_batch_size]
            block = self.data[(chunk[:, None] + win[None, :])].astype(np.int64)
            mblock = self.mask[(chunk[:, None] + win[None, :])].astype(np.int64)
            xb = np.ascontiguousarray(block[:, :-1])
            yb = np.ascontiguousarray(block[:, 1:])
            ym = np.ascontiguousarray(mblock[:, 1:])     # target-aligned loss mask
            yield torch.from_numpy(xb), torch.from_numpy(yb), torch.from_numpy(ym)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=str, default="midtrain,sft")
    args = ap.parse_args()
    build_all(tuple(x.strip() for x in args.layers.split(",") if x.strip()))


if __name__ == "__main__":
    main()
