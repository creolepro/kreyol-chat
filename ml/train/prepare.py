"""Workstream F, local prep — build the nanochat tokenizer bundle + parquet shards.

Two artifacts, written under `data/train_work/` (git-ignored) and later uploaded to
the Modal Volume at nanochat's expected base-dir layout:

  tokenizer/tokenizer.pkl   — a pickled tiktoken `Encoding` built from the committed
                              kreyol-bpe {mergeable_ranks, pattern} + the 9 special
                              tokens, EXACTLY as nanochat's RustBPETokenizer expects
                              (`from_directory` unpickles an Encoding). The kreyol_aware
                              pattern from meta.json is asserted to be the one used.
  tokenizer/token_bytes.pt  — per-token UTF-8 byte length (0 for specials), for BPB eval.
  base_data_climbmix/*.parquet — corpus v0.1 text as parquet (`text` column), EXCLUDING
                              the tokenizer_eval holdout, the two Workstream-E eval slices,
                              and (already absent) the probe proverbs. Last shard = val.

Run:  python -m train.prepare [--full]     # --full = whole corpus (Workstream G)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil

from . import config as T


# --- tokenizer bundle ---------------------------------------------------------

def build_tokenizer_bundle() -> dict:
    import tiktoken

    with open(T.KREYOL_BPE_PKL, "rb") as f:
        d = pickle.load(f)
    mergeable_ranks = d["mergeable_ranks"]       # {bytes: int}, 24,567 content ranks
    pattern = d["pattern"]
    meta = json.load(open(T.KREYOL_BPE_META, encoding="utf-8"))
    assert pattern == meta["pattern"], "kreyol-bpe pkl pattern != meta.json pattern"
    assert meta["pattern_name"] == "kreyol_aware", "expected the kreyol_aware pattern"

    offset = len(mergeable_ranks)                # 24,567
    special = {name: offset + i for i, name in enumerate(T.SPECIAL_TOKENS)}
    enc = tiktoken.Encoding(name="kreyol-bpe", pat_str=pattern,
                            mergeable_ranks=mergeable_ranks, special_tokens=special)
    assert enc.n_vocab == T.VOCAB_SIZE, f"n_vocab {enc.n_vocab} != {T.VOCAB_SIZE}"

    os.makedirs(T.BUNDLE_TOKENIZER, exist_ok=True)
    with open(os.path.join(T.BUNDLE_TOKENIZER, "tokenizer.pkl"), "wb") as f:
        pickle.dump(enc, f)

    # token_bytes — raw UTF-8 byte length per token id (specials = 0). Mirrors
    # nanochat/scripts/tok_train.py so evaluate_bpb reports vocab-invariant loss.
    # Saved as JSON here (torch is not a local dep); the Modal setup fn converts it
    # to the token_bytes.pt tensor nanochat's get_token_bytes() loads.
    special_ids = set(special.values())
    token_bytes = [0 if i in special_ids else len(enc.decode_single_token_bytes(i))
                   for i in range(enc.n_vocab)]
    with open(os.path.join(T.BUNDLE_TOKENIZER, "token_bytes.json"), "w") as f:
        json.dump(token_bytes, f)

    # round-trip sanity on Kreyòl text
    s = "Dèyè mòn gen mòn. Mwen renmen peyi mwen anpil."
    assert enc.decode(enc.encode_ordinary(s)) == s
    info = {"n_vocab": enc.n_vocab, "n_content": offset, "n_special": len(special),
            "pattern_name": meta["pattern_name"], "bos_id": enc.encode_single_token("<|bos|>"),
            "pad_multiple_64": ((T.VOCAB_SIZE + 63) // 64) * 64,
            "roundtrip_ok": True}
    print(f"[tokenizer] {info}")
    return info


# --- corpus v0.1 -> parquet ---------------------------------------------------

def _u01(key: str) -> float:
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def _in_tokenizer_holdout(doc_id: str) -> bool:
    # identical rule to ml/tokenizer/data.py in_holdout — exclude the SAME docs
    return _u01(f"holdout:{T.HOLDOUT_SPLIT_SEED}:{doc_id}") < T.HOLDOUT_FRAC


def _excluded_ids() -> set:
    ids = set()
    for path in (T.AUTHORED_EVAL, T.TRANSLATION_SHAPED_EVAL):
        if os.path.exists(path):
            for line in open(path, encoding="utf-8"):
                line = line.strip()
                if line:
                    ids.add(json.loads(line)["doc_id"])
    return ids


def build_parquet(full: bool) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq

    corpus = T.CORPUS_V0_1.format(tag="full")
    exclude = _excluded_ids()
    max_mb = -1 if full else T.SMOKE_MAX_MB
    # crude keep fraction to hit the byte cap on a mixed (not source-ordered) sample
    keep_frac = 1.0 if max_mb < 0 else min(1.0, max_mb / 400.0)
    val_frac = T.VAL_MAX_MB / max(T.SMOKE_MAX_MB, T.VAL_MAX_MB) if not full else 0.02

    if os.path.isdir(T.BUNDLE_DATA):
        shutil.rmtree(T.BUNDLE_DATA)
    os.makedirs(T.BUNDLE_DATA, exist_ok=True)

    train_texts, val_texts = [], []
    n_excl_slice = n_excl_holdout = 0
    for line in open(corpus, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        did = d["acquisition"]["doc_id"]
        if did in exclude:
            n_excl_slice += 1; continue
        if _in_tokenizer_holdout(did):
            n_excl_holdout += 1; continue
        if keep_frac < 1.0 and _u01(f"keep:{T.DATA_SEED}:{did}") >= keep_frac:
            continue
        (val_texts if _u01(f"val:{T.DATA_SEED}:{did}") < val_frac else train_texts).append(d["text"])

    def _write(path, texts):
        tbl = pa.table({"text": pa.array(texts, type=pa.string())})
        pq.write_table(tbl, path, row_group_size=T.PARQUET_ROW_GROUP, compression="zstd")

    # shard_00000..N = train; the last (highest index) = val
    n_train_shards = max(1, len(train_texts) // 20000 + (1 if len(train_texts) % 20000 else 0))
    per = (len(train_texts) + n_train_shards - 1) // max(1, n_train_shards)
    idx = 0
    for s in range(n_train_shards):
        chunk = train_texts[s * per:(s + 1) * per]
        if not chunk:
            continue
        _write(os.path.join(T.BUNDLE_DATA, f"shard_{idx:05d}.parquet"), chunk)
        idx += 1
    _write(os.path.join(T.BUNDLE_DATA, f"shard_{idx:05d}.parquet"), val_texts)  # val = last

    def _mb(texts):
        return sum(len(t.encode("utf-8")) for t in texts) / 1e6
    info = {"full": full, "train_docs": len(train_texts), "val_docs": len(val_texts),
            "train_mb": round(_mb(train_texts), 1), "val_mb": round(_mb(val_texts), 1),
            "train_shards": idx, "val_shard_index": idx,
            "excluded_eval_slices": n_excl_slice, "excluded_tokenizer_holdout": n_excl_holdout,
            "keep_frac": round(keep_frac, 4)}
    print(f"[parquet] {info}")
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="whole corpus (Workstream G); default = smoke cap")
    args = ap.parse_args()
    tok = build_tokenizer_bundle()
    data = build_parquet(args.full)
    manifest = {"snapshot_date": T.SNAPSHOT_DATE, "nanochat_commit": T.NANOCHAT_COMMIT,
                "tokenizer": tok, "data": data}
    os.makedirs(T.WORK, exist_ok=True)
    with open(os.path.join(T.WORK, "prepare_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[prepare] wrote bundle under {T.WORK}")


if __name__ == "__main__":
    main()
