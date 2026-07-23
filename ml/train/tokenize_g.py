"""Workstream G, local prep — pre-tokenize corpus v0.1 into flat uint16 bins + gather
the BPB eval-slice texts. Runs locally (tiktoken only; torch/transformers stay on Modal).

Outputs under `data/train_work/g/data/` (git-ignored), uploaded to the Modal Volume:

  train.bin / val.bin  — flat uint16 token streams (nanoGPT-style). Each doc is BOS +
                         content tokens; documents are concatenated in a SEEDED order.
                         train excludes the two Workstream-E eval slices and the
                         tokenizer_eval holdout (probe proverbs are already absent).
  eval_texts.json      — raw texts for the four BPB slices (authored_eval,
                         translation_shaped_eval, flores_hat, general_holdout). BPB is
                         byte-normalized, so each model tokenizes these itself — the
                         SAME texts drive Model C and the base-model comparison.
  manifest.json        — token/doc/byte counts, seeds, exclusions.

Run:  python -m train.tokenize_g [--sample]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle

import numpy as np

from . import config as F
from . import llama_config as G


# --- tiktoken Encoding (same construction as train/prepare.py) ----------------

def _encoding():
    import tiktoken
    with open(F.KREYOL_BPE_PKL, "rb") as fh:
        d = pickle.load(fh)
    mergeable_ranks, pattern = d["mergeable_ranks"], d["pattern"]
    meta = json.load(open(F.KREYOL_BPE_META, encoding="utf-8"))
    assert pattern == meta["pattern"] and meta["pattern_name"] == "kreyol_aware"
    offset = len(mergeable_ranks)                # 24,567
    special = {name: offset + i for i, name in enumerate(F.SPECIAL_TOKENS)}
    enc = tiktoken.Encoding(name="kreyol-bpe", pat_str=pattern,
                            mergeable_ranks=mergeable_ranks, special_tokens=special)
    assert enc.n_vocab == F.VOCAB_SIZE
    return enc, enc.encode_single_token("<|bos|>")


def _u01(key: str) -> float:
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def _in_tokenizer_holdout(doc_id: str) -> bool:
    return _u01(f"holdout:{F.HOLDOUT_SPLIT_SEED}:{doc_id}") < F.HOLDOUT_FRAC


def _excluded_slice_ids() -> set:
    ids = set()
    for path in (F.AUTHORED_EVAL, F.TRANSLATION_SHAPED_EVAL):
        if os.path.exists(path):
            for line in open(path, encoding="utf-8"):
                line = line.strip()
                if line:
                    ids.add(json.loads(line)["doc_id"])
    return ids


def _slice_texts(path: str) -> list:
    return [json.loads(l)["text"] for l in open(path, encoding="utf-8") if l.strip()]


def _flores_texts() -> list:
    if not os.path.exists(G.FLORES_HAT_DEVTEST):
        return []
    return [l.rstrip("\n") for l in open(G.FLORES_HAT_DEVTEST, encoding="utf-8") if l.strip()]


def build_parity_probe() -> dict:
    """The ~1k-line source-mixed probe (the EXISTING one from tokenizer/data.probe_lines)
    plus the apostrophe/clitic fixtures — the gate-4 token-ID parity input. Written to a
    JSON that the Modal convert_gates fn re-tokenizes through tiktoken / HF / llama.cpp.
    No probe proverbs (they are absent from the corpus)."""
    import sys as _sys
    _sys.path.insert(0, F.REPO_ROOT)
    from tokenizer import data as tdata

    probe = tdata.probe_lines(per_source=350)
    payload = {"probe_lines": probe, "fixtures": G.PARITY_FIXTURES,
               "n_probe": len(probe), "n_fixtures": len(G.PARITY_FIXTURES)}
    os.makedirs(G.G_BUNDLE_DATA, exist_ok=True)
    with open(os.path.join(G.G_BUNDLE_DATA, "parity_probe.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    print(f"[parity_probe] {payload['n_probe']} probe lines + {payload['n_fixtures']} fixtures")
    return {"n_probe": payload["n_probe"], "n_fixtures": payload["n_fixtures"]}


def build(sample: bool) -> dict:
    enc, bos = _encoding()
    corpus = F.CORPUS_V0_1.format(tag="sample" if sample else "full")
    exclude = _excluded_slice_ids()

    os.makedirs(G.G_BUNDLE_DATA, exist_ok=True)
    train_ids: list[np.ndarray] = []
    holdout_docs: list[tuple[str, str]] = []   # (doc_id, text) for general-holdout + val
    n_train = n_excl_slice = n_holdout = 0

    for line in open(corpus, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        did = d["acquisition"]["doc_id"]
        text = d["text"]
        if did in exclude:
            n_excl_slice += 1
            continue
        if _in_tokenizer_holdout(did):
            holdout_docs.append((did, text))
            n_holdout += 1
            continue
        toks = enc.encode_ordinary(text)
        arr = np.fromiter([bos, *toks], dtype=np.uint16, count=len(toks) + 1)
        train_ids.append(arr)
        n_train += 1

    # seeded document shuffle, then concatenate into the flat train stream
    rng = np.random.default_rng(G.TRAIN["seed"])
    order = rng.permutation(len(train_ids))
    train_stream = np.concatenate([train_ids[i] for i in order]) if train_ids else np.zeros(0, np.uint16)

    # val.bin: a seeded sample of the holdout docs (cheap periodic val-CE during training)
    rng2 = np.random.default_rng(G.TRAIN["seed"] + 1)
    hold_order = rng2.permutation(len(holdout_docs))
    val_parts, val_budget, val_bytes = [], 3_000_000, 0
    for i in hold_order:
        did, text = holdout_docs[i]
        toks = enc.encode_ordinary(text)
        val_parts.append(np.fromiter([bos, *toks], dtype=np.uint16, count=len(toks) + 1))
        val_bytes += len(text.encode("utf-8"))
        if val_bytes >= val_budget:
            break
    val_stream = np.concatenate(val_parts) if val_parts else np.zeros(0, np.uint16)

    train_stream.tofile(os.path.join(G.G_BUNDLE_DATA, "train.bin"))
    val_stream.tofile(os.path.join(G.G_BUNDLE_DATA, "val.bin"))

    # general-holdout BPB text: seed-sample holdout docs to ~700kB (scorecard budget)
    gen_hold, gh_bytes = [], 0
    for i in hold_order:
        did, text = holdout_docs[i]
        gen_hold.append(text)
        gh_bytes += len(text.encode("utf-8"))
        if gh_bytes >= G.BPB_GENERAL_HOLDOUT_BYTES:
            break

    eval_texts = {
        "authored_eval": _slice_texts(F.AUTHORED_EVAL) if os.path.exists(F.AUTHORED_EVAL) else [],
        "translation_shaped_eval": _slice_texts(F.TRANSLATION_SHAPED_EVAL) if os.path.exists(F.TRANSLATION_SHAPED_EVAL) else [],
        "flores_hat": _flores_texts(),
        "general_holdout": gen_hold,
    }
    with open(os.path.join(G.G_BUNDLE_DATA, "eval_texts.json"), "w", encoding="utf-8") as fh:
        json.dump(eval_texts, fh, ensure_ascii=False)

    def _mb(a):
        return round(a.size * 2 / 1e6, 1)

    manifest = {
        "snapshot_date": G.SNAPSHOT_DATE, "sample": sample, "seed": G.TRAIN["seed"],
        "bos_id": int(bos), "vocab_size": F.VOCAB_SIZE,
        "train_docs": n_train, "train_tokens": int(train_stream.size), "train_mb": _mb(train_stream),
        "val_tokens": int(val_stream.size), "val_mb": _mb(val_stream),
        "excluded_eval_slices": n_excl_slice, "tokenizer_holdout_docs": n_holdout,
        "eval_slice_docs": {k: len(v) for k, v in eval_texts.items()},
        "eval_slice_bytes": {k: sum(len(t.encode("utf-8")) for t in v) for k, v in eval_texts.items()},
    }
    with open(os.path.join(G.G_WORK, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[tokenize_g] {json.dumps(manifest, indent=2)}")
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true", help="use the corpus sample (fast smoke)")
    args = ap.parse_args()
    os.makedirs(G.G_WORK, exist_ok=True)
    build(args.sample)
    build_parity_probe()


if __name__ == "__main__":
    main()
