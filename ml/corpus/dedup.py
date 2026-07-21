"""Stage 4 — dedup. Exact + MinHash near-dup, document AND paragraph level, across sources.

docs/phase-0.md A2.4: exact (hash) then near-dup (MinHash/LSH, 5-gram shingles,
~0.8) at both document and paragraph level, across sources. Deterministic
survivor priority owned > wikipedia > crawl. Keep a duplicate MAP (cluster id +
why the survivor won) rather than silently discarding.

What runs here:
  * document exact-dup   — identical normalized text (content hash)
  * document near-dup    — MinHashLSH over word-5-gram shingles (Jaccard ~0.8)
  * paragraph exact-dup  — identical newline-delimited segments repeated ACROSS
                           documents are removed from all but the first (by
                           priority); this is the dominant crawl-boilerplate mode.
Paragraph-level NEAR-dup is intentionally not run at full scale (a MinHash per
paragraph is ~1–2M objects); it is covered indirectly by document near-dup +
paragraph exact-dup. The report states this explicitly.

Run:  python -m corpus.dedup [--sample]
"""

from __future__ import annotations

import argparse
import json
import os
import re

from datasketch import MinHash, MinHashLSH

from . import common, config

_WORD_RE = re.compile(r"\w+", re.UNICODE)


# --- union-find ---------------------------------------------------------------

class _UF:
    def __init__(self, keys):
        self.p = {k: k for k in keys}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


# --- helpers ------------------------------------------------------------------

def _priority(rec) -> int:
    src = rec["acquisition"]["source"]
    return config.SURVIVOR_PRIORITY.get(common.priority_class(src), 5)


def _survivor_key(rec):
    """Lower is better: (priority, -length, doc_id) — deterministic."""
    return (_priority(rec), -len(rec["text"]), rec["acquisition"]["doc_id"])


def _survivor_reason(winner, loser) -> str:
    if _priority(winner) != _priority(loser):
        return "higher_source_priority"
    if len(winner["text"]) != len(loser["text"]):
        return "longer_text"
    return "lower_doc_id"


def _shingles(text: str):
    words = [w.lower() for w in _WORD_RE.findall(text)]
    n = config.SHINGLE_N
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def _minhash(text: str) -> MinHash:
    m = MinHash(num_perm=config.MINHASH_NUM_PERM)
    for sh in _shingles(text):
        m.update(sh.encode("utf-8"))
    return m


def _segments(text: str):
    """Newline-delimited segments used as the paragraph unit for dedup."""
    return [s.strip() for s in re.split(r"\n+", text)
            if len(s.strip()) >= config.DEDUP_MIN_PARAGRAPH_CHARS]


# --- main ---------------------------------------------------------------------

def dedup_stage(sample: bool):
    tag = common.run_tag(sample)
    docs = []
    for source in config.PIPELINE_SOURCES:
        p = common.stage_path(tag, "filter", source)
        if os.path.exists(p):
            docs.extend(common.read_jsonl(p))
    by_id = {d["acquisition"]["doc_id"]: d for d in docs}
    common.log(f"  dedup input: {len(docs)} docs")

    clusters = []          # duplicate-map records
    removed_exact = 0
    removed_near = 0

    # 1) document exact-dup ----------------------------------------------------
    by_hash = {}
    for d in docs:
        h = d["acquisition"]["cleaned_content_hash"]
        by_hash.setdefault(h, []).append(d["acquisition"]["doc_id"])
    exact_survivors = []
    for h, ids in by_hash.items():
        if len(ids) == 1:
            exact_survivors.append(ids[0])
            continue
        members = [by_id[i] for i in ids]
        winner = min(members, key=_survivor_key)
        wid = winner["acquisition"]["doc_id"]
        exact_survivors.append(wid)
        removed_exact += len(ids) - 1
        clusters.append({
            "cluster_id": f"docexact:{h}", "level": "document_exact",
            "survivor_doc_id": wid,
            "survivor_reason": _survivor_reason(winner, next(m for m in members if m is not winner)),
            "size": len(ids), "members": ids,
        })

    # 2) document near-dup (MinHash LSH over exact-survivors) -------------------
    lsh = MinHashLSH(threshold=config.MINHASH_THRESHOLD, num_perm=config.MINHASH_NUM_PERM)
    mh = {}
    for i in exact_survivors:
        m = _minhash(by_id[i]["text"])
        mh[i] = m
        lsh.insert(i, m)
    uf = _UF(exact_survivors)
    for i in exact_survivors:
        for j in lsh.query(mh[i]):
            if j != i:
                uf.union(i, j)
    groups = {}
    for i in exact_survivors:
        groups.setdefault(uf.find(i), []).append(i)
    near_survivors = []
    for root, ids in groups.items():
        if len(ids) == 1:
            near_survivors.append(ids[0])
            continue
        members = [by_id[i] for i in ids]
        winner = min(members, key=_survivor_key)
        wid = winner["acquisition"]["doc_id"]
        near_survivors.append(wid)
        removed_near += len(ids) - 1
        clusters.append({
            "cluster_id": f"docnear:{wid}", "level": "document_near",
            "survivor_doc_id": wid,
            "survivor_reason": _survivor_reason(winner, min((m for m in members if m is not winner), key=_survivor_key)),
            "size": len(ids), "members": sorted(ids),
        })

    # 3) paragraph exact-dup across surviving docs (crawl boilerplate) ---------
    #    First occurrence (by survivor priority order) keeps the segment; later
    #    docs drop it. Deterministic ordering by _survivor_key.
    near_survivors.sort(key=lambda i: _survivor_key(by_id[i]))
    seen_seg = set()
    paragraphs_removed = 0
    docs_touched = 0
    final = []
    for i in near_survivors:
        d = by_id[i]
        segs = d["text"].split("\n")
        kept, removed_here = [], 0
        for seg in segs:
            s = seg.strip()
            if len(s) >= config.DEDUP_MIN_PARAGRAPH_CHARS:
                key = common.content_hash(s)
                if key in seen_seg:
                    removed_here += 1
                    continue
                seen_seg.add(key)
            kept.append(seg)
        newtext = re.sub(r"\n{3,}", config.PARAGRAPH_SEP, "\n".join(kept)).strip()
        if removed_here:
            paragraphs_removed += removed_here
            docs_touched += 1
        if not newtext.strip():
            # entire doc was duplicated boilerplate -> drop it
            clusters.append({
                "cluster_id": f"paraall:{i}", "level": "paragraph_exact_dropped_doc",
                "survivor_doc_id": None, "survivor_reason": "all_paragraphs_duplicated",
                "size": 1, "members": [i],
            })
            continue
        d = dict(d)
        d["text"] = newtext
        d["acquisition"] = dict(d["acquisition"])
        d["acquisition"]["cleaned_content_hash"] = common.content_hash(newtext)
        d["paragraphs_removed"] = removed_here
        d["n_paragraphs"] = sum(1 for p in newtext.split(config.PARAGRAPH_SEP) if p.strip())
        final.append(d)

    # --- write outputs --------------------------------------------------------
    out = os.path.join(config.CLEAN, f"corpus_v0-{tag}.jsonl")
    n = common.write_jsonl(out, final)
    dupmap = common.stage_path(tag, "dedup", "duplicate_map")
    common.write_jsonl(dupmap, clusters)

    surv_by_prio = {}
    for d in final:
        pc = common.priority_class(d["acquisition"]["source"])
        surv_by_prio[pc] = surv_by_prio.get(pc, 0) + 1
    stats = {
        "input_docs": len(docs),
        "document_exact_removed": removed_exact,
        "document_near_removed": removed_near,
        "paragraph_exact_removed": paragraphs_removed,
        "paragraph_dropped_docs": sum(1 for c in clusters if c["level"] == "paragraph_exact_dropped_doc"),
        "docs_touched_by_paragraph_dedup": docs_touched,
        "final_docs": len(final),
        "survivors_by_priority_class": surv_by_prio,
        "clusters": len(clusters),
    }
    stats_path = common.stage_path(tag, "dedup", "dedup").replace(".jsonl", ".stats.json")
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    common.log(f"  dedup -> {out}  (final={len(final)}  exact-={removed_exact} "
               f"near-={removed_near} para-segs-={paragraphs_removed})")
    return out, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    args = ap.parse_args()
    common.log(f"[dedup] sample={args.sample}")
    dedup_stage(args.sample)


if __name__ == "__main__":
    main()
