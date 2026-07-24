"""Workstream J4 — small clean wins (Bib La, Konstitisyon, Storybooks Haiti).

Each is rights-registered in rights.yaml BEFORE ingestion. Downloads (cached
under ml/data/smallwins/, untracked), extracts clean Kreyòl text, and writes
register-tagged Document records to the v0.2 ingest dir. Religious register (Bib
La) is capped to <=2% of the corpus at build time (config_v0_2.RELIGIOUS_CAP_FRAC).

fineweb-2 survivors are materialized separately (corpus/fineweb_ingest.py, heavy).
Bloom is pending HF gate acceptance.

Run:  uv run python -m corpus.small_wins [bib|konst|storybooks|all]
"""

from __future__ import annotations

import argparse
import io
import os
import re
import zipfile

import requests
from bs4 import BeautifulSoup

from . import common, schema
from . import config_v0_2 as C2

UA = ("kreyol-chat/0.2 (+https://github.com/creolepro/kreyol-chat; "
      "patricedouge@gmail.com)")
SW = C2.SMALLWINS_DIR
STORYBOOKS_ZIP = "https://codeload.github.com/global-asp/storybooks-haiti/zip/refs/heads/master"


def _get(url: str, timeout: int = 120) -> bytes:
    r = requests.get(url, headers={"User-Agent": UA}, timeout=timeout)
    r.raise_for_status()
    return r.content


def _doc(source: str, text: str, doc_id: str, register: str, genre: str,
         origin: str, url: str | None = None) -> dict:
    rec = schema.Document(
        text=text, origin=origin, genre=genre,
        acquisition={
            "source": source, "source_name": source, "url": url, "revision": None,
            "download_timestamp": common.now_iso(), "doc_id": doc_id,
            "raw_content_hash": common.content_hash(text),
        },
        rights=common.rights_for(source), split="train",
    ).model_dump(mode="json")
    rec["register"] = register
    return rec


# --- Bib La 1985 (eBible USFM) ---------------------------------------------
def _usfm_to_text(usfm: str) -> str:
    s = re.sub(r"\\f\b.*?\\f\*", "", usfm, flags=re.S)     # footnotes
    s = re.sub(r"\\x\b.*?\\x\*", "", s, flags=re.S)         # cross-refs
    verses = [m.group(1) for m in re.finditer(r"\\v\s+\d+[a-z]?\s+(.*)", s)]
    text = " ".join(verses)
    text = re.sub(r"\\[a-z0-9]+\*?", " ", text)             # strip remaining markers
    text = re.sub(r"\s+", " ", text).strip()
    return text


def bib_la() -> int:
    os.makedirs(SW, exist_ok=True)
    zpath = os.path.join(SW, "hat_usfm.zip")
    if not (os.path.exists(zpath) and os.path.getsize(zpath) > 0):
        with open(zpath, "wb") as f:
            f.write(_get(C2.BIB_LA_USFM_URL))
    recs = []
    with zipfile.ZipFile(zpath) as z:
        for name in z.namelist():
            if not name.lower().endswith((".usfm", ".sfm")):
                continue
            book = re.sub(r"\.(usfm|sfm)$", "", os.path.basename(name))
            text = _usfm_to_text(z.read(name).decode("utf-8", errors="replace"))
            if len(text) < 200:
                continue
            recs.append(_doc("bib_la_1985", text, f"bib_la_1985:{book}",
                             "religious", "religious", "human_translation",
                             url="https://ebible.org/hat/"))
    out = os.path.join(C2.V0_2_INGEST, "bib_la_1985.jsonl")
    common.write_jsonl(out, recs)
    common.log(f"[small_wins] bib_la: {len(recs)} books")
    return len(recs)


# --- Konstitisyon 1987 (Wikisource) ----------------------------------------
def konstitisyon() -> int:
    html = _get(C2.KONSTITISYON_WIKISOURCE).decode("utf-8", errors="replace")
    soup = BeautifulSoup(html, "lxml")
    c = soup.select_one("div.mw-parser-output")
    for tag in c.select(".mw-editsection, style, .reflist, table, .navbox, sup.reference"):
        tag.decompose()
    text = c.get_text("\n", strip=True)
    text = re.sub(r"\n{2,}", "\n", text).strip()
    rec = _doc("konstitisyon_1987", text, "konstitisyon_1987:0000", "legal",
               "historical", "human_translation", url=C2.KONSTITISYON_WIKISOURCE)
    out = os.path.join(C2.V0_2_INGEST, "konstitisyon_1987.jsonl")
    common.write_jsonl(out, [rec])
    common.log(f"[small_wins] konstitisyon: {len(text):,} chars")
    return 1


# --- Storybooks Haiti (global-asp) -----------------------------------------
_SB_NAV_HEAD = re.compile(r".*?Back to stories list\s*", re.S)


def storybooks() -> int:
    os.makedirs(SW, exist_ok=True)
    zpath = os.path.join(SW, "storybooks.zip")
    if not (os.path.exists(zpath) and os.path.getsize(zpath) > 0):
        with open(zpath, "wb") as f:
            f.write(_get(STORYBOOKS_ZIP))
    recs = []
    with zipfile.ZipFile(zpath) as z:
        ht_pages = [n for n in z.namelist()
                    if "/stories/ht/" in n and n.endswith("index.html")]
        for n in ht_pages:
            soup = BeautifulSoup(z.read(n).decode("utf-8", errors="replace"), "lxml")
            # story pages are bilingual; .l1 = primary language (HT), .l2 = French.
            # Take HT only, drop the French parallel text.
            parts = [e.get_text(" ", strip=True) for e in soup.select(".l1")]
            txt = " ".join(p for p in parts if p)
            txt = re.sub(r"\s+", " ", txt).strip()
            if len(txt) < 80:
                continue
            sid = n.split("/stories/ht/")[1].split("/")[0]
            recs.append(_doc("storybooks_haiti", txt, f"storybooks_haiti:{sid}",
                             "children", "educational", "human_translation",
                             url=f"https://global-asp.github.io/storybooks-haiti/stories/ht/{sid}/"))
    out = os.path.join(C2.V0_2_INGEST, "storybooks_haiti.jsonl")
    common.write_jsonl(out, recs)
    common.log(f"[small_wins] storybooks: {len(recs)} HT stories")
    return len(recs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="?", default="all",
                    choices=["bib", "konst", "storybooks", "all"])
    args = ap.parse_args()
    os.makedirs(C2.V0_2_INGEST, exist_ok=True)
    if args.which in ("bib", "all"):
        bib_la()
    if args.which in ("konst", "all"):
        konstitisyon()
    if args.which in ("storybooks", "all"):
        storybooks()


if __name__ == "__main__":
    main()
