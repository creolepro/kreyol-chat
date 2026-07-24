"""J0 — MinHash overlap of the VOA sample against the existing v0.1 corpus.

MADLAD already contains partial VOA, so some sampled articles will near-duplicate
docs we already have. This measures that overlap rate on the PD-clean article
sample, which discounts the net-new token projection.

Run:  uv run python -m corpus.j0_scoping overlap   (after `voa` + corpus_index build)
"""

from __future__ import annotations

from . import common, corpus_index, j0_scoping


def run():
    data = j0_scoping._load("voa_sample.json")
    arts = [r for r in data["records"]
            if r.get("kind") == "article" and not r["wire_byline"] and r.get("clean_text")]
    common.log(f"[j0.overlap] loading corpus v0.1 minhash index...")
    lsh, mh = corpus_index.load()
    common.log(f"[j0.overlap] index has {len(mh):,} docs; querying {len(arts)} VOA articles")

    dup = 0
    dup_tokens = 0
    tot_tokens = 0
    examples = []
    for r in arts:
        hits = corpus_index.query(lsh, r["clean_text"])
        tot_tokens += r["kb_tokens"]
        if hits:
            dup += 1
            dup_tokens += r["kb_tokens"]
            if len(examples) < 8:
                examples.append({"url": r["url"], "hits": hits[:3]})

    overlap_rate = dup / len(arts) if arts else 0.0
    overlap_token_rate = dup_tokens / tot_tokens if tot_tokens else 0.0
    agg = {
        "articles_checked": len(arts),
        "articles_dup": dup,
        "overlap_rate_docs": overlap_rate,
        "overlap_rate_tokens": overlap_token_rate,
        "examples": examples,
    }
    j0_scoping._save("voa_madlad_overlap.json", agg)
    common.log(f"[j0.overlap] overlap: {dup}/{len(arts)} docs = {overlap_rate:.3f}; "
               f"token-overlap {overlap_token_rate:.3f}")
    return agg
