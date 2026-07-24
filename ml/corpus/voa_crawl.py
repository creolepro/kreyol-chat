"""Workstream J1 — VOA Nouvèl resumable crawler.

Newest-first (by descending numeric article id — id correlates with recency, and
this front-loads recent journalism + 2026 articles for authored_eval_v2). Safe to
interrupt/resume across sessions via a checkpoint state file. Politeness and the
wire carve-out live in corpus/voa.py.

Writes (all under ml/data/voa/, untracked):
  crawl_state.json      resumable checkpoint (cursor, counts, done ids)
  articles.jsonl        one line per KEPT PD-clean article (extracted + provenance)
  html/<id>.html.gz     raw HTML of kept articles (provenance/reproducibility)

Per-article provenance: url, id, date_published, byline (author), sections,
genre=news / register=journalism_authored. Wire-bylined articles are dropped
whole; wire-credited paragraphs are stripped from kept articles (voa.extract).

Run (background):  uv run python -m corpus.voa_crawl [--limit N] [--max-seconds S]
Resume: just run again — it continues from the checkpoint.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import time

from . import common, voa
from . import config_v0_2 as C2


def _load_state() -> dict:
    if os.path.exists(C2.VOA_STATE):
        with open(C2.VOA_STATE, encoding="utf-8") as f:
            return json.load(f)
    return {"cursor": 0, "done_ids": [],
            "counts": {"article_pd_clean": 0, "wire": 0, "non_article": 0,
                       "topic_redirect": 0, "http_error": 0, "no_id": 0},
            "pub_years": {}}


def _save_state(state: dict):
    os.makedirs(C2.VOA_DIR, exist_ok=True)
    tmp = C2.VOA_STATE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)
    os.replace(tmp, C2.VOA_STATE)


def _ordered_urls(session) -> list[dict]:
    """All sitemap entries, deduped by id, ordered by DESCENDING id (recent
    first). Entries without a numeric id go last (order preserved)."""
    entries = voa.sitemap_entries(session)
    by_id: dict[str, dict] = {}
    no_id = []
    for e in entries:
        if e["id"]:
            # prefer the slugged url if duplicate ids appear
            if e["id"] not in by_id or "/a/" + e["id"] + ".html" not in e["url"]:
                by_id.setdefault(e["id"], e)
        else:
            no_id.append(e)
    ordered = sorted(by_id.values(), key=lambda e: int(e["id"]), reverse=True)
    return ordered + no_id


def crawl(limit: int | None = None, max_seconds: int | None = None,
          checkpoint_every: int = 50):
    os.makedirs(C2.VOA_DIR, exist_ok=True)
    os.makedirs(os.path.join(C2.VOA_DIR, "html"), exist_ok=True)
    session = voa.make_session()
    common.log("[voa_crawl] enumerating sitemap...")
    ordered = _ordered_urls(session)
    common.log(f"[voa_crawl] {len(ordered):,} unique article URLs (descending id)")

    state = _load_state()
    done = set(state["done_ids"])
    counts = state["counts"]
    pub_years = state["pub_years"]
    cursor = state["cursor"]
    t0 = time.time()
    processed = 0

    fout = open(C2.VOA_ARTICLES, "a", encoding="utf-8")
    try:
        while cursor < len(ordered):
            e = ordered[cursor]
            cursor += 1
            key = e["id"] or e["url"]
            if key in done:
                continue
            if not e["id"]:
                counts["no_id"] += 1
                done.add(key)
                continue

            r = voa.get(session, e["url"], allow_redirects=False)
            if voa.is_topic_redirect(r):
                counts["topic_redirect"] += 1
            elif r.status_code != 200:
                counts["http_error"] += 1
            else:
                d = voa.extract(r.text, e["url"])
                if not d["is_article"]:
                    counts["non_article"] += 1
                elif d["wire_byline"]:
                    counts["wire"] += 1
                else:
                    py = (d["date_published"] or "")[:4] or "unknown"
                    pub_years[py] = pub_years.get(py, 0) + 1
                    counts["article_pd_clean"] += 1
                    rec = {
                        "id": e["id"], "url": e["url"],
                        "date_published": d["date_published"],
                        "date_modified": d["date_modified"],
                        "author": d["author"], "sections": d["sections"],
                        "headline": d["headline"], "text": d["clean_text"],
                        "n_paragraphs": d["n_paragraphs"],
                        "wire_para_stripped": d["wire_para_count"],
                        "sitemap_lastmod": e["lastmod"],
                    }
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fout.flush()
                    # raw HTML provenance for kept articles
                    with gzip.open(os.path.join(C2.VOA_DIR, "html", f"{e['id']}.html.gz"),
                                   "wt", encoding="utf-8") as hf:
                        hf.write(r.text)

            done.add(key)
            processed += 1
            if processed % checkpoint_every == 0:
                state.update({"cursor": cursor, "done_ids": list(done),
                              "counts": counts, "pub_years": pub_years})
                _save_state(state)
                common.log(f"[voa_crawl] cursor={cursor}/{len(ordered)} "
                           f"kept={counts['article_pd_clean']} wire={counts['wire']} "
                           f"audio/vid={counts['non_article']} redirect={counts['topic_redirect']} "
                           f"({time.time()-t0:.0f}s)")
            if limit and processed >= limit:
                common.log(f"[voa_crawl] hit --limit {limit}"); break
            if max_seconds and (time.time() - t0) >= max_seconds:
                common.log(f"[voa_crawl] hit --max-seconds {max_seconds}"); break
    finally:
        fout.close()
        state.update({"cursor": cursor, "done_ids": list(done),
                      "counts": counts, "pub_years": pub_years})
        _save_state(state)

    common.log(f"[voa_crawl] STOP cursor={cursor}/{len(ordered)} "
               f"kept={counts['article_pd_clean']} "
               f"pub_years={dict(sorted(pub_years.items()))}")
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-seconds", type=int, default=None)
    args = ap.parse_args()
    crawl(limit=args.limit, max_seconds=args.max_seconds)


if __name__ == "__main__":
    main()
