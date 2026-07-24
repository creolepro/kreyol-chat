"""Workstream J0 — scoping pass (measure before crawling).

Produces the numbers behind ml/reports/corpus_v0_2_scoping.md and the binding
RED-FLAG GATE (docs/data.md J0):

  STOP after J0 if any of:
    * VOA measured net-new < 5M tokens, OR
    * wire-content rate makes clean filtering infeasible (> 40% of articles), OR
    * robots.txt blocks the crawl.

Stages (each writes an untracked JSON sidecar under data/interim/j0/):
  voa      — stratified-by-year sample of ~200 VOA articles: tokens/article
             (kreyol-bpe), wire rate, is_article rate, projected net-new.
  overlap  — MinHash overlap of the VOA sample against MADLAD (already in v0.1).
  fineweb  — fineweb-2 hat_Latn: raw size + true net-new after dedup vs MADLAD.
  bloom    — sil-ai/bloom-lm hat: per-entry license split (keep BY/BY-SA).

Token unit is kreyol-bpe (the training tokenizer; matches the "112M unique
tokens" v0.1 figure). o200k is recorded alongside for cross-reference.

Run:  uv run python -m corpus.j0_scoping <voa|overlap|fineweb|bloom> [--n 200]
Nothing under data/ is committed.
"""

from __future__ import annotations

import argparse
import json
import os
import random

from . import common, config, voa

J0_DIR = os.path.join(config.DATA, "interim", "j0")


def _kb():
    from tokenizer.core import KreyolBPE
    return KreyolBPE.load_pkl(os.path.join(config.REPO_ROOT, "tokenizer",
                                            "kreyol-bpe", "tokenizer.pkl"))


def _o200k():
    return common.RefTokenizer()


def _save(name: str, obj: dict):
    os.makedirs(J0_DIR, exist_ok=True)
    path = os.path.join(J0_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    common.log(f"  wrote {path}")
    return path


def _load(name: str) -> dict:
    with open(os.path.join(J0_DIR, name), encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------
# VOA stratified sample
# --------------------------------------------------------------------------
def stratified_sample(entries: list[dict], n: int, seed: int) -> list[dict]:
    """Proportional-by-year allocation with a floor, so every year is represented
    but recent high-volume years (the corpus mass) dominate."""
    rng = random.Random(seed)
    by_year: dict[str, list] = {}
    for e in entries:
        by_year.setdefault(e["year"] or "unknown", []).append(e)
    total = len(entries)
    picked: list[dict] = []
    for year, items in by_year.items():
        share = max(2, round(n * len(items) / total)) if len(items) >= 2 else len(items)
        share = min(share, len(items))
        picked += rng.sample(items, share)
    rng.shuffle(picked)
    return picked


def run_voa(n: int, seed: int = 20260724):
    kb, o2 = _kb(), _o200k()
    s = voa.make_session()
    common.log("[j0.voa] fetching sitemap shards...")
    entries = voa.sitemap_entries(s)
    common.log(f"[j0.voa] {len(entries):,} article URLs in sitemap")
    years = sorted({e["year"] for e in entries if e["year"]})
    year_hist = {y: sum(1 for e in entries if e["year"] == y) for y in years}

    sample = stratified_sample(entries, n, seed)
    common.log(f"[j0.voa] fetching {len(sample)} sampled URLs (~1 req/s)...")

    records = []
    for i, e in enumerate(sample):
        r = voa.get(s, e["url"], allow_redirects=False)
        rec = {"url": e["url"], "sitemap_year": e["year"], "id": e["id"],
               "status": r.status_code}
        if voa.is_topic_redirect(r):
            rec["kind"] = "topic_redirect"
        elif r.status_code != 200:
            rec["kind"] = f"http_{r.status_code}"
        else:
            d = voa.extract(r.text, e["url"])
            kind = "article" if d["is_article"] else "non_article"
            rec.update({
                "kind": kind,
                "is_article": d["is_article"],
                "wire_byline": d["wire_byline"],
                "wire_para_count": d["wire_para_count"],
                "author": d["author"],
                "date_published": d["date_published"],
                "n_paragraphs": d["n_paragraphs"],
                "body_chars": d["body_chars"],
                "kb_tokens": kb.count(d["clean_text"]) if d["is_article"] else 0,
                "o200k_tokens": o2.count(d["clean_text"]) if d["is_article"] else 0,
                "pub_year": (d["date_published"] or "")[:4] or None,
            })
            # keep the clean text of articles for the overlap stage
            if d["is_article"]:
                rec["clean_text"] = d["clean_text"]
        records.append(rec)
        if (i + 1) % 25 == 0:
            common.log(f"    {i+1}/{len(sample)}")

    agg = _aggregate(records, len(entries), year_hist, len(sample), seed)
    _save("voa_sample.json", {"agg": agg, "records": records})
    _log_voa(agg)
    return agg


def _aggregate(records, sitemap_total, year_hist, sample_size, seed):
    """Build the VOA-sample aggregate. `wire_byline` on each article record is the
    authoritative wire flag (recomputed by reclassify_wire() from the author field
    via voa._WIRE_BYLINE, so a corrected regex reflows the whole aggregate)."""
    arts = [r for r in records if r.get("kind") == "article"]
    kept = [r for r in arts if not r["wire_byline"]]          # PD-clean articles
    redirects = [r for r in records if r.get("kind") == "topic_redirect"]
    nonart = [r for r in records if r.get("kind") == "non_article"]
    errs = [r for r in records if str(r.get("kind", "")).startswith("http_")]

    def _mean(xs):
        return (sum(xs) / len(xs)) if xs else 0.0

    kb_tokens = [r["kb_tokens"] for r in kept]
    o2_tokens = [r["o200k_tokens"] for r in kept]
    wire_arts = [r for r in arts if r["wire_byline"]]
    # per-sampled-URL yield of PD-clean tokens (folds in article-rate + wire-drop)
    per_url_kb = sum(kb_tokens) / sample_size if sample_size else 0.0
    per_url_o2 = sum(o2_tokens) / sample_size if sample_size else 0.0

    return {
        "sitemap_total_urls": sitemap_total,
        "sitemap_year_hist": year_hist,
        "sample_size": sample_size,
        "counts": {
            "article": len(arts),
            "article_pd_clean": len(kept),
            "article_wire_byline": len(wire_arts),
            "topic_redirect": len(redirects),
            "non_article": len(nonart),
            "http_error": len(errs),
        },
        "rates": {
            "live_article_rate": len(arts) / sample_size if sample_size else 0.0,
            "pd_clean_rate_of_urls": len(kept) / sample_size if sample_size else 0.0,
            "wire_byline_rate_of_articles": (len(wire_arts) / len(arts)) if arts else 0.0,
        },
        "tokens_per_article_kb": {
            "mean": _mean(kb_tokens),
            "median": sorted(kb_tokens)[len(kb_tokens)//2] if kb_tokens else 0,
            "min": min(kb_tokens) if kb_tokens else 0,
            "max": max(kb_tokens) if kb_tokens else 0,
        },
        "tokens_per_article_o200k": {"mean": _mean(o2_tokens)},
        "per_url_yield_kb": per_url_kb,
        "per_url_yield_o200k": per_url_o2,
        "gross_projection_kb": sitemap_total * per_url_kb,
        "gross_projection_o200k": sitemap_total * per_url_o2,
        "pub_year_hist_of_articles": _hist([r.get("pub_year") for r in arts]),
        "pd_clean_article_count": len(kept),
        "seed": seed,
    }


def reclassify_wire():
    """Recompute each article's wire_byline from its stored author via the current
    voa._WIRE_BYLINE, then rebuild the aggregate. Lets a corrected wire regex reflow
    the saved sample without re-crawling."""
    data = _load("voa_sample.json")
    recs = data["records"]
    for r in recs:
        if r.get("kind") == "article":
            r["wire_byline"] = bool(r.get("author") and voa._WIRE_BYLINE.search(r["author"]))
    agg = data["agg"]
    new = _aggregate(recs, agg["sitemap_total_urls"], agg["sitemap_year_hist"],
                     agg["sample_size"], agg.get("seed"))
    _save("voa_sample.json", {"agg": new, "records": recs})
    _log_voa(new)
    return new


def _hist(vals):
    h = {}
    for v in vals:
        h[v or "unknown"] = h.get(v or "unknown", 0) + 1
    return dict(sorted(h.items()))


def _log_voa(agg):
    c, r = agg["counts"], agg["rates"]
    common.log(f"[j0.voa] sample={agg['sample_size']}  articles={c['article']} "
               f"(pd_clean={c['article_pd_clean']}, wire={c['article_wire_byline']}) "
               f"redirect={c['topic_redirect']} non_article={c['non_article']} err={c['http_error']}")
    common.log(f"[j0.voa] live_article_rate={r['live_article_rate']:.3f}  "
               f"pd_clean_rate_of_urls={r['pd_clean_rate_of_urls']:.3f}  "
               f"wire_byline_rate={r['wire_byline_rate_of_articles']:.3f}")
    common.log(f"[j0.voa] tokens/article (kreyol-bpe): mean={agg['tokens_per_article_kb']['mean']:.0f} "
               f"median={agg['tokens_per_article_kb']['median']}")
    common.log(f"[j0.voa] GROSS projection: {agg['gross_projection_kb']/1e6:.2f}M kb-tok / "
               f"{agg['gross_projection_o200k']/1e6:.2f}M o200k-tok (before dedup)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["voa", "reclassify", "overlap", "fineweb", "bloom"])
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()
    if args.stage == "voa":
        run_voa(args.n)
    elif args.stage == "reclassify":
        reclassify_wire()
    elif args.stage == "overlap":
        from . import j0_overlap
        j0_overlap.run()
    elif args.stage == "fineweb":
        from . import j0_fineweb
        j0_fineweb.run()
    elif args.stage == "bloom":
        from . import j0_bloom
        j0_bloom.run()


if __name__ == "__main__":
    main()
