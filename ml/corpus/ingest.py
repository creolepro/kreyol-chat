"""Stage 1 — ingest. Rights-clear sources -> per-source JSONL with §5.2 metadata.

Each document gets the full Document schema (origin, genre, acquisition anchors,
rights copied from rights.yaml, split from splits.yaml) and is validated against
it. Raw downloads stay immutable in data/raw/downloads/; this stage only reads
them. Re-runnable and deterministically samplable (the 1% first-pass run).

Run:  python -m corpus.ingest [--sample]
"""

from __future__ import annotations

import argparse
import bz2
import gzip
import hashlib
import json
import os
import re
import xml.etree.ElementTree as ET

import mwparserfromhell

from . import common, config
from .schema import Document

# --- MADLAD-400 ht clean ------------------------------------------------------

# MADLAD ht clean encodes line breaks as LITERAL backslash-n (two chars), not
# real newlines — so paragraph structure is invisible until decoded. Source-
# specific decode at ingest (Wikipedia extraction already yields real newlines).
def _madlad_decode_newlines(text: str) -> str:
    return text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", " ")


def ingest_madlad(sample: bool):
    """MADLAD ht clean: one JSON doc per line, only a `text` field."""
    common.ensure_madlad()
    tag = common.run_tag(sample)
    frac = config.SAMPLE_FRAC if sample else 1.0
    out = common.stage_path(tag, "ingest", "madlad")
    rights = common.rights_for(config.MADLAD_SOURCE_KEY)
    dl_ts = common.file_mtime_iso(config.MADLAD_LOCAL)

    def records():
        with gzip.open(config.MADLAD_LOCAL, "rt", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                text = _madlad_decode_newlines(json.loads(line).get("text", ""))
                if not text.strip():
                    continue
                doc_id = f"{config.MADLAD_SOURCE_KEY}:{i:07d}"
                if not common.deterministic_keep(doc_id, frac, config.SAMPLE_SEED):
                    continue
                doc = Document(
                    text=text,
                    origin="web_crawl",           # crawl of unknown authorship
                    genre="web",
                    acquisition={
                        "source": config.MADLAD_SOURCE_KEY,
                        "source_name": "MADLAD-400 ht clean",
                        "url": None,               # no per-doc URL in the release
                        "revision": config.MADLAD_REVISION,
                        "download_timestamp": dl_ts,
                        "doc_id": doc_id,
                        "raw_content_hash": common.content_hash(text),
                    },
                    rights=rights,
                    split="train",
                )
                yield doc.model_dump(mode="json")

    n = common.write_jsonl(out, records())
    common.log(f"  ingest MADLAD -> {out}  ({n} docs, frac={frac})")
    return out, n


# --- htwiki (dump -> wikitext -> plaintext) -----------------------------------

_WIKI_NS = "{http://www.mediawiki.org/xml/export-0.11/}"
_REDIRECT_RE = re.compile(r"^\s*#\s*(REDIRECT|REDIRECTION|REDIREKSYON)", re.IGNORECASE)
# Lines to drop from stripped output: table markup, category/file/image lines.
_DROP_LINE_RE = re.compile(
    r"^\s*(\{\||\|\}|\|[-+}]?|!|(Kategori|Category|Fichye|File|Image|Modèl|Template)\s*:)",
    re.IGNORECASE,
)


def _wikitext_to_plaintext(wikitext: str) -> str:
    """Strip wikitext to readable prose while preserving paragraph boundaries."""
    try:
        code = mwparserfromhell.parse(wikitext)
        text = code.strip_code(normalize=True, collapse=False)
    except Exception:
        text = wikitext
    kept = []
    for ln in text.split("\n"):
        if _DROP_LINE_RE.match(ln):
            continue
        kept.append(ln)
    return "\n".join(kept)


def _wiki_stub_signals(wikitext: str, plaintext: str) -> dict:
    """Cheap signals for the bot-stub heuristic (computed at filter time)."""
    try:
        code = mwparserfromhell.parse(wikitext)
        n_templates = len(code.filter_templates())
        n_links = len(code.filter_wikilinks())
    except Exception:
        n_templates = wikitext.count("{{")
        n_links = wikitext.count("[[")
    return {
        "wiki_wikitext_chars": len(wikitext),
        "wiki_plaintext_chars": len(plaintext),
        "wiki_n_templates": n_templates,
        "wiki_n_links": n_links,
    }


def _iter_pages(path: str):
    """Yield (pageid, title, wikitext) for ns=0, non-redirect pages."""
    with bz2.open(path, "rb") as f:
        for _event, elem in ET.iterparse(f, events=("end",)):
            if elem.tag != _WIKI_NS + "page":
                continue
            ns = elem.findtext(_WIKI_NS + "ns")
            title = elem.findtext(_WIKI_NS + "title") or ""
            pageid = elem.findtext(_WIKI_NS + "id") or ""
            redirect = elem.find(_WIKI_NS + "redirect")
            rev = elem.find(_WIKI_NS + "revision")
            text = ""
            if rev is not None:
                text = rev.findtext(_WIKI_NS + "text") or ""
            elem.clear()
            if ns != "0" or redirect is not None:
                continue
            if _REDIRECT_RE.match(text):
                continue
            if not text.strip():
                continue
            yield pageid, title, text


def ingest_htwiki(sample: bool):
    common.ensure_htwiki()
    tag = common.run_tag(sample)
    frac = config.SAMPLE_FRAC if sample else 1.0
    out = common.stage_path(tag, "ingest", "htwiki")
    rights = common.rights_for(config.HTWIKI_SOURCE_KEY)
    dl_ts = common.file_mtime_iso(config.HTWIKI_LOCAL)

    def records():
        for pageid, title, wikitext in _iter_pages(config.HTWIKI_LOCAL):
            doc_id = f"{config.HTWIKI_SOURCE_KEY}:{pageid}"
            if not common.deterministic_keep(doc_id, frac, config.SAMPLE_SEED):
                continue
            plaintext = _wikitext_to_plaintext(wikitext)
            if not plaintext.strip():
                continue
            signals = _wiki_stub_signals(wikitext, plaintext)
            doc = Document(
                text=plaintext,
                origin="authored_kreyol",
                genre="encyclopedic",
                acquisition={
                    "source": config.HTWIKI_SOURCE_KEY,
                    "source_name": "Haitian Creole Wikipedia",
                    "url": f"https://ht.wikipedia.org/?curid={pageid}",
                    "revision": f"dump {config.HTWIKI_DUMP_DATE}",
                    "download_timestamp": dl_ts,
                    "doc_id": doc_id,
                    "raw_content_hash": common.content_hash(plaintext),
                },
                rights=rights,
                split="train",
                wiki_title=title,
                wiki_pageid=pageid,
                **signals,
            )
            yield doc.model_dump(mode="json")

    n = common.write_jsonl(out, records())
    common.log(f"  ingest htwiki -> {out}  ({n} docs, frac={frac})")
    return out, n


# --- owned proverbs (CreolePro) — teachable into train, probe held out --------

def ingest_proverbs(sample: bool):
    """Owned proverbs. Deterministic teachable/probe split (splits.yaml):
    the PROVERBS_PROBE_N with the smallest seeded hash are the held-out probe
    set (written to a separate file, NEVER ingested into the training corpus);
    the rest are teachable and flow through the pipeline. Not subsampled — the
    set is tiny and the 1% run should still exercise the owned path.
    """
    from .proverbs_fetch import fetch_proverbs

    if not os.path.exists(config.PROVERBS_LOCAL):
        fetch_proverbs()
    rows = list(common.read_jsonl(config.PROVERBS_LOCAL))
    tag = common.run_tag(sample)
    out = common.stage_path(tag, "ingest", "proverbs")
    rights = common.rights_for(config.PROVERBS_SOURCE_KEY)
    dl_ts = common.file_mtime_iso(config.PROVERBS_LOCAL)

    def _hkey(r):
        return hashlib.sha256(
            f"{config.PROVERBS_SPLIT_SEED}:{r['kreyol']}".encode("utf-8")).hexdigest()

    ranked = sorted(rows, key=_hkey)
    probe_nums = {r["num"] for r in ranked[:config.PROVERBS_PROBE_N]}

    def _doc(r, split_name, split_enum):
        doc_id = f"{config.PROVERBS_SOURCE_KEY}:{r['num']:02d}"
        return Document(
            text=r["kreyol"],                 # Kreyòl only; English is metadata
            origin="authored_kreyol",
            genre="proverb",
            acquisition={
                "source": config.PROVERBS_SOURCE_KEY,
                "source_name": "CreolePro proverbs",
                "url": config.PROVERBS_URL,
                "revision": f"blog {config.PROVERBS_FETCH_DATE}",
                "download_timestamp": dl_ts,
                "doc_id": doc_id,
                "raw_content_hash": common.content_hash(r["kreyol"]),
            },
            rights=rights,
            split=split_enum,
            proverb_num=r["num"],
            proverb_english=r.get("english"),
            proverb_category=r.get("category"),
            proverb_split=split_name,
        ).model_dump(mode="json")

    teachable = [_doc(r, "teachable", "train") for r in rows if r["num"] not in probe_nums]
    probe = [_doc(r, "probe", "exhibit_examples") for r in rows if r["num"] in probe_nums]
    n = common.write_jsonl(out, teachable)
    common.write_jsonl(config.PROVERBS_PROBE_LOCAL, probe)
    common.log(f"  ingest proverbs -> {out}  ({n} teachable; "
               f"{len(probe)} probe held out -> {config.PROVERBS_PROBE_LOCAL})")
    return out, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true", help="ingest a deterministic 1%% sample")
    args = ap.parse_args()
    common.log(f"[ingest] sample={args.sample}")
    ingest_madlad(args.sample)
    ingest_htwiki(args.sample)
    ingest_proverbs(args.sample)


if __name__ == "__main__":
    main()
