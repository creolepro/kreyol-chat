"""Workstream J-ADDENDUM — assemble corpus v0.2.1 (incremental on top of v0.2).

Reuses J's pipeline. Design: keep corpus v0.2 FROZEN as the base, add the NET-NEW
survivors of the four sweep-4 sources (register-tagged), deduped STRICTLY (exact +
MinHash) against the FULL v0.2 corpus (v0.1 base + all J survivors) — not just
v0.1. This is an ADDENDUM, not a rebuild: the ~104.9M-token v0.2 shard is streamed
through unchanged and the sweep-4 survivors are appended.

Dedup is INVERTED for memory: there are only a few hundred sweep-4 candidate docs,
vs 315k in v0.2. So we build a tiny LSH over the CANDIDATES and stream the full v0.2
shard once through it (computing each v0.2 doc's content-hash for exact-dedup and
MinHash for near-dedup). Memory stays tiny; one pass; no 500MB index to pickle.

Per sweep-4 source (data/interim/v0_2_1_ingest/*.jsonl, written by corpus.sweep4):
  normalize → probe-proverb guard (standing rule) → langid (skipped for historical
  orthography, which langid cannot read) → [streamed] EXACT-dedup vs v0.2 (content
  hash) + MinHash-dedup vs v0.2 (Jaccard 0.8) → cross-new dedup → append.

The one held-out Cric? Crac! heritage fable never entered the ingest files, so it is
absent here by construction (leak-check asserted).

Output (untracked): data/clean/corpus_v0_2_1-full.jsonl + build_stats.json
Run:  uv run python -m corpus.build_v0_2_1
"""

from __future__ import annotations

import json
import os

from datasketch import MinHashLSH

from . import audit, common, dedup
from . import config as CFG
from . import config_v0_2 as C2
from .build_v0_2 import (LANGID_DROP_CONF, _FOREIGN, _cross_dedup, _has_probe,
                         _norm, _pnorm, _tally)

SWEEP4_SOURCES = ["cmu_haitian", "cric_crac_1901", "anthologie_1925"]
# historical orthography defeats fastText langid (docs/data.md §4) — skip it there;
# the marker-based Kreyòl extraction already isolated the language at ingest.
SKIP_LANGID = {"cric_crac_1901", "anthologie_1925"}
MIN_CHARS = 40


def _kb():
    from tokenizer.core import KreyolBPE
    return KreyolBPE.load_pkl(os.path.join(common.config.REPO_ROOT, "tokenizer",
                                            "kreyol-bpe", "tokenizer.pkl"))


def _load_candidates(stats):
    """Load + pre-filter (probe/langid) the sweep-4 ingest docs into candidates.
    Each candidate carries its MinHash + content hash for the streaming dedup."""
    cands = []
    for src in SWEEP4_SOURCES:
        path = os.path.join(C2.V0_2_1_INGEST, f"{src}.jsonl")
        s = {"in": 0, "short": 0, "probe_leak": 0, "langid": 0,
             "exact_dup_v02": 0, "near_dup_v02": 0, "kept": 0}
        if not os.path.exists(path):
            stats[src] = s
            continue
        prio = C2.survivor_priority(common.priority_class(src))
        for d in common.read_jsonl(path):
            s["in"] += 1
            text = _norm(d["text"])
            if len(text) < MIN_CHARS:
                s["short"] += 1
                continue
            if _has_probe(_pnorm(text)):        # standing rule: no probe proverb in train
                s["probe_leak"] += 1
                continue
            if src not in SKIP_LANGID:
                lang, conf = audit._lid_predict(text)
                if lang in _FOREIGN and conf >= LANGID_DROP_CONF:
                    s["langid"] += 1
                    continue
            d["text"] = text
            d["_mh"] = dedup._minhash(text)
            d["_prio"] = prio
            d["_src"] = src
            d["_chash"] = common.content_hash(text)
            cands.append(d)
        stats[src] = s
    return cands


def build() -> dict:
    kb = _kb()
    stats = {"sources": {}}
    cands = _load_candidates(stats["sources"])
    common.log(f"[v0.2.1] {len(cands)} sweep-4 candidates after probe/langid")

    # tiny LSH over the candidates; exact-hash map candidate content_hash -> [idx]
    lsh = MinHashLSH(threshold=CFG.MINHASH_THRESHOLD, num_perm=CFG.MINHASH_NUM_PERM)
    by_hash = {}
    for i, d in enumerate(cands):
        lsh.insert(i, d["_mh"])
        by_hash.setdefault(d["_chash"], []).append(i)
    exact_dup = [False] * len(cands)
    near_dup = [False] * len(cands)

    # stream v0.2 -> v0.2.1 shard; dedup candidates against every v0.2 doc
    v02_shard = C2.CORPUS_V0_2.format(tag="full")
    out = C2.CORPUS_V0_2_1.format(tag="full")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    comp = {}
    base_docs = base_tokens = 0
    with open(out, "w", encoding="utf-8") as f:
        for d in common.read_jsonl(v02_shard):
            t = d["text"]
            for i in by_hash.get(common.content_hash(t), ()):     # EXACT dedup
                exact_dup[i] = True
            for i in lsh.query(dedup._minhash(t)):                # MinHash near-dedup
                near_dup[i] = True
            reg = d.get("register", "web_crawl")
            _tally(comp, reg, kb.count(t))
            base_docs += 1
            base_tokens += kb.count(t)
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
            if base_docs % 50000 == 0:
                common.log(f"[v0.2.1] streamed {base_docs:,} v0.2 docs...")
    common.log(f"[v0.2.1] v0.2 base: {base_docs:,} docs / {base_tokens:,} kb-tok")

    # collect survivors + per-source removal tallies
    survivors = []
    for i, d in enumerate(cands):
        src = d["_src"]
        if exact_dup[i]:
            stats["sources"][src]["exact_dup_v02"] += 1
        elif near_dup[i]:
            stats["sources"][src]["near_dup_v02"] += 1
        else:
            survivors.append(d)
    for src in SWEEP4_SOURCES:
        stats["sources"][src]["kept"] = sum(1 for d in survivors if d["_src"] == src)
        s = stats["sources"][src]
        common.log(f"[v0.2.1] {src}: in={s['in']} probe={s['probe_leak']} "
                   f"langid={s['langid']} exact={s['exact_dup_v02']} "
                   f"near={s['near_dup_v02']} -> kept {s['kept']}")

    # cross-new dedup among the sweep-4 survivors (priority survivor)
    n_before = len(survivors)
    survivors = _cross_dedup(survivors)
    stats["cross_new_dedup_removed"] = n_before - len(survivors)

    # heritage-fable holdout invariant
    heritage_txt = _load_heritage_text()
    if heritage_txt:
        hnorm = _pnorm(heritage_txt)[:200]
        leaked = sum(1 for d in survivors if hnorm and hnorm in _pnorm(d["text"]))
        stats["heritage_fable_leak_check"] = leaked
        assert leaked == 0, "heritage fable leaked into training!"

    # append survivors to the v0.2.1 shard
    with open(out, "a", encoding="utf-8") as f:
        for d in survivors:
            for k in ("_mh", "_prio", "_k", "_src", "_chash"):
                d.pop(k, None)
            _tally(comp, d.get("register", "web_crawl"), kb.count(d["text"]))
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    new_tokens = sum(kb.count(d["text"]) for d in survivors)
    stats["composition_by_register"] = comp
    stats["base_v0_2_docs"] = base_docs
    stats["base_v0_2_tokens_kb"] = base_tokens
    stats["new_survivor_docs"] = len(survivors)
    stats["new_survivor_tokens_kb"] = new_tokens
    stats["total_docs"] = base_docs + len(survivors)
    stats["total_tokens_kb"] = base_tokens + new_tokens
    stats["mix_weights"] = C2.MIX_WEIGHTS
    stats["output_shard"] = os.path.basename(out)
    sw = os.path.join(common.config.DATA, "interim", "sweep4_stats.json")
    if os.path.exists(sw):
        stats["sweep4_ingest"] = json.load(open(sw, encoding="utf-8"))

    with open(C2.V0_2_1_STATS.format(tag="full"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    common.log(f"[v0.2.1] corpus v0.2.1: {stats['total_docs']:,} docs / "
               f"{stats['total_tokens_kb']:,} kb-tok (+{new_tokens:,} net-new) -> {out}")
    return stats


def _load_heritage_text() -> str | None:
    p = os.path.join(C2.V0_2_1_INGEST, "_cric_heritage_fable.json")
    if os.path.exists(p):
        return json.load(open(p, encoding="utf-8")).get("text")
    return None


if __name__ == "__main__":
    build()
