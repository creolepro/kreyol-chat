"""Reusable near-dup index over the EXISTING corpus (v0.1) — Workstream J.

Computes one MinHash per existing v0.1 document (word-5-gram shingles,
num_perm=128, Jaccard threshold 0.8 — byte-identical to corpus.dedup) and pickles
them under data/interim/ (untracked). Every J source is then measured for
true net-new tokens, and deduped, against everything already in the corpus —
MADLAD (91.7%) + htwiki (8.3%) — using the same LSH the pipeline uses.

Build once:  uv run python -m corpus.corpus_index build
Then load()  -> (lsh, minhashes) for overlap/dedup queries.
"""

from __future__ import annotations

import argparse
import os
import pickle
import time

from datasketch import MinHashLSH

from . import common, config, dedup
from . import config_v0_1 as CV

INDEX_PKL = os.path.join(config.DATA, "interim", "corpus_v0_1_minhash.pkl")


def build(limit: int | None = None) -> int:
    shard = CV.CORPUS_V0_1.format(tag="full")
    os.makedirs(os.path.dirname(INDEX_PKL), exist_ok=True)
    mh: dict[str, object] = {}
    t0 = time.time()
    for i, d in enumerate(common.read_jsonl(shard)):
        did = d["acquisition"]["doc_id"]
        mh[did] = dedup._minhash(d["text"])
        if (i + 1) % 20000 == 0:
            common.log(f"  [corpus_index] {i+1:,} docs  ({time.time()-t0:.0f}s)")
        if limit and i + 1 >= limit:
            break
    with open(INDEX_PKL, "wb") as f:
        pickle.dump(mh, f, protocol=pickle.HIGHEST_PROTOCOL)
    common.log(f"  [corpus_index] built {len(mh):,} minhashes -> {INDEX_PKL} "
               f"({time.time()-t0:.0f}s)")
    return len(mh)


def load():
    """Return (lsh, minhashes dict). lsh keys are v0.1 doc_ids."""
    with open(INDEX_PKL, "rb") as f:
        mh = pickle.load(f)
    lsh = MinHashLSH(threshold=config.MINHASH_THRESHOLD,
                     num_perm=config.MINHASH_NUM_PERM)
    with lsh.insertion_session() as sess:
        for did, m in mh.items():
            sess.insert(did, m)
    return lsh, mh


def query(lsh, text: str) -> list[str]:
    """v0.1 doc_ids that near-duplicate `text` (Jaccard >= 0.8)."""
    return lsh.query(dedup._minhash(text))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    build(args.limit)


if __name__ == "__main__":
    main()
