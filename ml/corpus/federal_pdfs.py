"""Workstream J2 — U.S. federal-agency Haitian-Creole PDF harvest.

Federal works are public domain (17 U.S.C. §105). Consumes a verified URL list
(producer / register / url / title), downloads each PDF politely, extracts text
with multi-column handling, ingests it register-tagged, and mines the IRS Pub 850
EN<->HT glossary pairs into the committable corpus/glossary_pairs_federal.json.

MedlinePlus is a ROUTER: only federal-producer PDFs are included (the discovery
step excludes ACS / IAC / Mass DPH / Mass General). Per-doc provenance carries
the producing agency + url + register.

URL list: ml/data/interim/federal_urls.json  (written from J2 discovery; untracked)
Run:  uv run python -m corpus.federal_pdfs harvest
Nothing under ml/data/ is committed; the mined IRS pairs (federal PD) are.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import requests

from . import common, family, schema
from . import config_v0_2 as C2

UA = ("kreyol-chat/0.2 (+https://github.com/creolepro/kreyol-chat; "
      "patricedouge@gmail.com)")
URL_LIST = os.path.join(common.config.DATA, "interim", "federal_urls.json")
FEDERAL_STATS = os.path.join(common.config.DATA, "interim", "federal_stats.json")


def _session():
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Accept": "application/pdf,*/*"})
    return s


def download(session, url: str, dest: str, timeout: int = 60) -> bool:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return True
    try:
        r = session.get(url, timeout=timeout, stream=True)
        time.sleep(1.0)
        if r.status_code != 200:
            common.log(f"  [federal] {r.status_code} {url}")
            return False
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)
        return os.path.getsize(dest) > 0
    except requests.RequestException as e:
        common.log(f"  [federal] ERR {type(e).__name__} {url}")
        return False


def extract_pdf_text(path: str) -> str:
    """Column-aware extraction: per page, cluster text blocks into columns by x0
    and read column-by-column (top-to-bottom). Falls back to plain reading order
    for single-column pages."""
    import fitz
    doc = fitz.open(path)
    out = []
    for page in doc:
        blocks = [b for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()]
        if not blocks:
            continue
        # detect 2 columns: split at page mid if blocks populate both halves
        mid = page.rect.width / 2
        left = [b for b in blocks if (b[0] + b[2]) / 2 < mid]
        right = [b for b in blocks if (b[0] + b[2]) / 2 >= mid]
        if len(left) >= 3 and len(right) >= 3:
            ordered = sorted(left, key=lambda b: b[1]) + sorted(right, key=lambda b: b[1])
        else:
            ordered = sorted(blocks, key=lambda b: (round(b[1] / 3), b[0]))
        out.append("\n".join(b[4].strip() for b in ordered))
    doc.close()
    return "\n\n".join(out)


def harvest(url_list_path: str = URL_LIST) -> dict:
    if not os.path.exists(url_list_path):
        raise FileNotFoundError(f"URL list not found: {url_list_path} "
                                "(run J2 discovery first)")
    with open(url_list_path, encoding="utf-8") as f:
        urls = json.load(f)
    urls = [u for u in urls if u.get("url")]
    session = _session()
    os.makedirs(C2.V0_2_INGEST, exist_ok=True)

    records = []
    per_producer: dict[str, dict] = {}
    samples: dict[str, list] = {}
    irs_pub850_path = None

    for i, u in enumerate(urls):
        prod = u.get("producer", "unknown")
        reg = u.get("register", "government")
        url = u["url"]
        h = hashlib.sha256(url.encode()).hexdigest()[:12]
        dest = os.path.join(C2.FEDERAL_DIR, prod, f"{h}.pdf")
        if not download(session, url, dest):
            per_producer.setdefault(prod, _blank())["failed"] += 1
            continue
        try:
            text = extract_pdf_text(dest)
        except Exception as e:
            common.log(f"  [federal] extract fail {url}: {type(e).__name__}")
            per_producer.setdefault(prod, _blank())["failed"] += 1
            continue
        if len(text.strip()) < 200:
            per_producer.setdefault(prod, _blank())["empty"] += 1
            continue
        doc_id = f"us_federal_pdfs:{prod}:{h}"
        rec = schema.Document(
            text=text, origin="human_translation", genre="educational",
            acquisition={
                "source": "us_federal_pdfs",
                "source_name": f"US federal PDF ({prod})",
                "url": url, "revision": None,
                "download_timestamp": common.now_iso(),
                "doc_id": doc_id, "raw_content_hash": common.content_hash(text),
            },
            rights=common.rights_for("us_federal_pdfs"), split="train",
        ).model_dump(mode="json")
        rec["register"] = reg
        rec["agency"] = prod
        rec["title"] = u.get("title")
        records.append(rec)
        pp = per_producer.setdefault(prod, _blank())
        pp["docs"] += 1
        pp["chars"] += len(text)
        # keep up to 3 spot-check samples per producer
        samples.setdefault(prod, [])
        if len(samples[prod]) < 3:
            samples[prod].append({"url": url, "title": u.get("title"),
                                  "head": text[:300], "chars": len(text)})
        # note the IRS Pub 850 glossary for pair mining
        if prod == "irs" and ("850" in (u.get("title") or "") or "850" in url
                              or "glossary" in (u.get("notes") or "").lower()):
            irs_pub850_path = dest

    ingest = os.path.join(C2.V0_2_INGEST, "us_federal_pdfs.jsonl")
    common.write_jsonl(ingest, records)

    # IRS Pub 850 EN<->HT pairs -> committable federal pairs file
    irs_pairs = []
    if irs_pub850_path:
        try:
            irs_pairs = family.two_col_pairs(irs_pub850_path, skip_pages=1)
            family._write_glossary_pairs_federal({"irs_pub850": irs_pairs})
        except Exception as e:
            common.log(f"  [federal] IRS pair mine failed: {type(e).__name__}: {e}")

    stats = {
        "url_count": len(urls),
        "ingested_docs": len(records),
        "per_producer": per_producer,
        "samples": samples,
        "irs_pub850_pairs": len(irs_pairs),
        "total_chars": sum(len(r["text"]) for r in records),
    }
    with open(FEDERAL_STATS, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    common.log(f"[federal] ingested {len(records)} docs from "
               f"{len(per_producer)} producers; IRS pairs={len(irs_pairs)}")
    for prod, pp in sorted(per_producer.items()):
        common.log(f"    {prod:12s} docs={pp['docs']:3d} chars={pp['chars']:8,d} "
                   f"failed={pp['failed']} empty={pp['empty']}")
    return stats


def _blank():
    return {"docs": 0, "chars": 0, "failed": 0, "empty": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["harvest"])
    ap.add_argument("--urls", default=URL_LIST)
    args = ap.parse_args()
    harvest(args.urls)


if __name__ == "__main__":
    main()
