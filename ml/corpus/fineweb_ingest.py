"""Workstream J4 — materialize fineweb-2 hat_Latn survivors (Pile B volume).

Re-streams fineweb-2 hat_Latn, re-applies the standing langid + junk filters and
MinHash dedup vs v0.1 (identical gates to J0's measurement), and WRITES the
survivors as register-tagged Document records to the v0.2 ingest dir. This is the
~108M-token web-crawl volume the user chose to include (register=web_crawl), held
at mix weight 1.0 so it does not inflate the authored share.

Heavy: ~224k docs streamed, one MinHash + LSH query each (~15 min). Run in the
background. Output (untracked): data/interim/v0_2_ingest/fineweb2_hat.jsonl

Run:  uv run python -m corpus.fineweb_ingest
"""

from __future__ import annotations

import json
import os

from . import common, corpus_index, junk, schema
from . import config_v0_2 as C2
from .j0_fineweb import FINEWEB_REPO, FINEWEB_CONFIG, LANGID_MIN, _load_token


def run():
    _load_token()
    from datasets import load_dataset
    common.log("[fineweb_ingest] loading v0.1 minhash index...")
    lsh, mh = corpus_index.load()
    common.log(f"[fineweb_ingest] streaming {FINEWEB_REPO}:{FINEWEB_CONFIG} ...")
    ds = load_dataset(FINEWEB_REPO, FINEWEB_CONFIG, split="train", streaming=True)

    os.makedirs(C2.V0_2_INGEST, exist_ok=True)
    out = os.path.join(C2.V0_2_INGEST, "fineweb2_hat.jsonl")
    rights = common.rights_for("fineweb2_hat")
    now = common.now_iso()

    n = kept = 0
    with open(out, "w", encoding="utf-8") as f:
        for r in ds:
            text = (r.get("text") or "").strip()
            if not text:
                continue
            n += 1
            try:
                score = float(r.get("language_score") or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            if r.get("language") != "hat" or score < LANGID_MIN:
                continue
            if junk.junk_reason(text, "crawl"):
                continue
            if corpus_index.query(lsh, text):
                continue
            fid = str(r.get("id") or n)
            doc_id = f"fineweb2_hat:{fid}"
            rec = schema.Document(
                text=text, origin="web_crawl", genre="web",
                acquisition={
                    "source": "fineweb2_hat", "source_name": "fineweb-2 hat_Latn",
                    "url": r.get("url"), "revision": None,
                    "download_timestamp": now, "doc_id": doc_id,
                    "raw_content_hash": common.content_hash(text),
                },
                rights=rights, split="train",
            ).model_dump(mode="json")
            rec["register"] = "web_crawl"
            rec["fineweb_lang_score"] = round(score, 4)
            rec["cc_date"] = r.get("date")
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1
            if n % 20000 == 0:
                common.log(f"    streamed {n:,} | kept {kept:,}")
    common.log(f"[fineweb_ingest] DONE streamed {n:,} docs -> kept {kept:,} survivors -> {out}")
    return {"streamed": n, "kept": kept, "out": out}


if __name__ == "__main__":
    run()
