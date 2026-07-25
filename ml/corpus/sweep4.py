"""Sweep-4 addendum — ingest the four verified TRAIN-OK sweep-4 sources.

Runs AFTER the v0.2 build; feeds the incremental v0.2.1 build (corpus.build_v0_2_1).
Each source is rights-registered in rights.yaml BEFORE this runs. Nothing here is
committed — raw downloads live under data/smallwins/sweep4/ (untracked), ingest
JSONL under data/interim/v0_2_1_ingest/ (untracked).

Sources (docs/data.md §1 sweep-4 rows; docs/data-sources-sweep-4.md):
  * cmu_haitian     — CMU permissive license. newswire+medical -> train
                      (news_translated / health_translated); glossary -> lexicon
                      resource (NOT LM text).
  * cric_crac_1901  — Georges Sylvain, PD 1901. dLOC page-level OCR; keep the
                      HT-dominant pages (facing French dropped); pre-reform
                      orthography preserved EXACTLY; page provenance; probe-check
                      load-bearing; one fable held out for the heritage exhibit.
  * anthologie_1925 — Manioc CC0; clean PDF text layer; clearly-Kreyòl passages
                      only; historical orthography preserved.
  * lessons_1921    — PD, but ingestion ABANDONED on OCR-quality grounds (English
                      manual; Kreyòl only as garbled drill cells). Measured + logged.

Run:  uv run python -m corpus.sweep4 [cmu|cric|anthologie|lessons|all]
"""

from __future__ import annotations

import argparse
import json
import os
import re

from . import common, schema
from . import config_v0_2 as C2
from .build_v0_2 import _has_probe, _norm, _pnorm  # standing probe-proverb guard

SW = C2.SWEEP4_DIR
OUT = C2.V0_2_1_INGEST

# One complete Cric? Crac! fable is held out of training entirely for the heritage
# exhibit (reports/heritage_cric_crac_fable.md). Set to the (start, end) INCLUSIVE
# dLOC pageorder span of the chosen fable after inspecting the extracted segments;
# those pages are excluded from the corpus shard. See sweep4_stats fable listing.
# "Le Loup, la Chèvre et le Chevreau" (Wolf, Goat & Kid) — the wolf-at-the-door /
# password fable, complete and self-contained, dLOC pageorders 38–41. Held out of
# training entirely as the heritage exhibit ("oldest text connected to the model").
HERITAGE_FABLE_SPAN: tuple[int, int] | None = (38, 41)


# --------------------------------------------------------------------------- #
# Historical-orthography Kreyòl vs French line/page classifier.
# langid FAILS on pre-reform spelling (docs/data.md §4), so this is a marker-based
# pass (strong distinctive markers + Kreyòl orthographic signals), reviewed by
# eyeballing the output — the "model-assisted then eyeball" the spec calls for.
# --------------------------------------------------------------------------- #
_K_STRONG = set("""gnou moin moé zott zot 9a ça té ac con kon cé cè pité pitit pititt
apé ape oué oue rhélé mwen nanpwen tande kite rele pouki koté kisa toutt bett tett
jodi kijan konsa cabritt cochon chien loup rat zaffé zafè bagay pié piti-mi""".split())
_K_MED = set("""moin li yo nan pou pa pas sa se ak te ap ki nou ou tout moun fè wè di
dit gen jou fait ale min con nou-menm""".split())
_F_STRONG = set("""les des une dans avec qui que est vous être leur cette était toutes
bêtes parlaient mais dont pour nous mes appelait fut avait c'est j'ai elle ils elles
chose bien tous maîtres quand aussi je son ses ce au aux la le du en""".split())
_E_STRONG = set("""the of and to in is was by which for that with have this as are be
at chief service authority government health personnel examination going coming
would tense present past""".split())
_TOK = re.compile(r"[A-Za-zÀ-ÿ'9]+")


def _toks(line: str):
    return _TOK.findall(line.lower())


def _orth_bonus(ts) -> int:
    return sum(1 for t in ts
              if "gn" in t or t.endswith(("tt", "nn", "ll", "dd"))
              or re.search(r"9[aeiou]", t) or t.startswith(("l'", "m'", "t'", "n'")))


def _score(line: str) -> tuple[int, int, int]:
    ts = _toks(line)
    if len(ts) < 2:
        return 0, 0, 0
    k = (sum(2 for t in ts if t in _K_STRONG) + sum(1 for t in ts if t in _K_MED)
         + _orth_bonus(ts))
    f = sum(2 for t in ts if t in _F_STRONG)
    e = sum(2 for t in ts if t in _E_STRONG)
    return k, f, e


def _is_kreyol_line(line: str) -> bool:
    k, f, e = _score(line)
    return k >= 3 and k > f and k > e


def _clean_line(line: str) -> str:
    """Strip OCR page furniture but PRESERVE orthography of kept text."""
    s = line.strip()
    s = re.sub(r"^\W*\d+\W*$", "", s)          # bare page numbers / footnote refs
    s = re.sub(r"\(\d+\)", "", s)              # inline footnote markers like (10)
    return s.strip()


def _kb():
    from tokenizer.core import KreyolBPE
    return KreyolBPE.load_pkl(os.path.join(common.config.REPO_ROOT, "tokenizer",
                                            "kreyol-bpe", "tokenizer.pkl"))


def _doc(source, text, doc_id, register, genre, origin, url=None, **extra):
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
    rec.update(extra)
    return rec


# --------------------------------------------------------------------------- #
# 1. CMU Haitian Corpus
# --------------------------------------------------------------------------- #
CMU_CHUNK = 40          # sentences per document (translated newswire has no doc structure)


def _read_sentences(path: str) -> list[str]:
    seen, out = set(), []
    for ln in open(path, encoding="utf-8", errors="replace"):
        s = _norm(ln)
        if len(s) < 15:
            continue
        h = _pnorm(s)
        if h in seen:               # internal exact-dedup
            continue
        seen.add(h)
        out.append(s)
    return out


def _chunk_docs(sents, source, tag, register, genre) -> list[dict]:
    recs = []
    for i in range(0, len(sents), CMU_CHUNK):
        block = "\n".join(sents[i:i + CMU_CHUNK])
        recs.append(_doc(source, block, f"{source}:{tag}:{i//CMU_CHUNK:04d}",
                         register, genre, "human_translation",
                         url="https://www.speech.cs.cmu.edu/haitian/text/"))
    return recs


def cmu(stats: dict) -> int:
    news = _read_sentences(os.path.join(SW, "cmu_newswire.ht"))
    med = _read_sentences(os.path.join(SW, "cmu_medical.ht"))
    recs = (_chunk_docs(news, "cmu_haitian", "newswire", "news_translated", "news")
            + _chunk_docs(med, "cmu_haitian", "medical", "health_translated", "educational"))
    common.write_jsonl(os.path.join(OUT, "cmu_haitian.jsonl"), recs)

    # glossary-all-fix.{ht,en} -> lexicon resource (NOT LM training text)
    ht = [l.rstrip("\n") for l in open(os.path.join(SW, "cmu_glossary.ht"), encoding="utf-8", errors="replace")]
    en = [l.rstrip("\n") for l in open(os.path.join(SW, "cmu_glossary.en"), encoding="utf-8", errors="replace")]
    pairs, seen = [], set()
    for h, e in zip(ht, en):
        h, e = h.strip(), e.strip()
        if h and e and h.lower() != e.lower() and (h, e) not in seen:
            seen.add((h, e))
            pairs.append({"ht": h, "en": e})
    os.makedirs(OUT, exist_ok=True)
    with open(C2.CMU_LEXICON, "w", encoding="utf-8") as f:
        json.dump({"source": "cmu_haitian", "kind": "en_ht_glossary",
                   "note": "CMU glossary term pairs — lexicon resource, NOT LM training text",
                   "n_pairs": len(pairs), "pairs": pairs}, f, ensure_ascii=False, indent=1)

    kb = _kb()
    stats["cmu_haitian"] = {
        "newswire_sentences": len(news), "medical_sentences": len(med),
        "docs": len(recs), "tokens_kb": sum(kb.count(r["text"]) for r in recs),
        "lexicon_pairs": len(pairs), "verdict": "usable",
    }
    common.log(f"[sweep4] cmu: {len(news)} news + {len(med)} med sentences "
               f"-> {len(recs)} docs; {len(pairs)} lexicon pairs")
    return len(recs)


# --------------------------------------------------------------------------- #
# 2. Cric? Crac! 1901 (dLOC page OCR) — HT-dominant pages, fable-run docs
# --------------------------------------------------------------------------- #
def _cric_pages() -> list[dict]:
    return [json.loads(l) for l in open(os.path.join(SW, "cric_pages.jsonl"),
                                        encoding="utf-8")]


def _page_kreyol_text(page_text: str) -> tuple[str, int, int]:
    """Return (kept Kreyòl text, kreyol_score_total, french_score_total) for a page."""
    keep, K, F = [], 0, 0
    for ln in page_text.split("\n"):
        k, f, e = _score(ln)
        K += k
        F += f
        if _is_kreyol_line(ln):
            c = _clean_line(ln)
            if c:
                keep.append(c)
    return "\n".join(keep), K, F


def _cric_segments():
    """Group contiguous HT-dominant pages into fable-runs. Each segment keeps its
    per-page (order, kreyol_text) so the heritage fable can be held out at PAGE
    granularity (fable boundaries do not always align with page-run boundaries).
    Returns list of {span:(lo,hi), page_texts:[(order,text)...], text:str}."""
    pages = _cric_pages()
    segs, cur = [], None
    for p in pages:
        txt, K, F = _page_kreyol_text(p["pagetext"] or "")
        ht = K > F * 1.15 and K >= 6 and len(txt) > 60
        if ht:
            if cur is None:
                cur = []
            cur.append((p["pageorder"], txt))
        elif cur:
            segs.append(cur)
            cur = None
    if cur:
        segs.append(cur)
    out = []
    for pt in segs:
        out.append({"span": (pt[0][0], pt[-1][0]), "page_texts": pt,
                    "text": "\n".join(t for _, t in pt)})
    return out


def _in_heritage(order: int) -> bool:
    return bool(HERITAGE_FABLE_SPAN and
               HERITAGE_FABLE_SPAN[0] <= order <= HERITAGE_FABLE_SPAN[1])


def cric(stats: dict, list_only: bool = False) -> int:
    segs = _cric_segments()
    kb = _kb()
    if list_only:
        common.log(f"[sweep4] cric: {len(segs)} HT fable-runs")
        for i, s in enumerate(segs):
            common.log(f"   seg {i:02d} pages {s['span'][0]}-{s['span'][1]} "
                       f"chars {len(s['text']):>5} tok {kb.count(s['text']):>5} "
                       f"| {s['text'][:60].replace(chr(10),' ')}")
        return len(segs)

    recs, probe_hits, held_pages = [], 0, []
    for s in segs:
        # hold out heritage-fable pages at PAGE granularity; the rest of the run
        # (a different fable sharing the run) still enters training
        kept = [(o, t) for (o, t) in s["page_texts"] if not _in_heritage(o)]
        held_pages += [(o, t) for (o, t) in s["page_texts"] if _in_heritage(o)]
        if not kept:
            continue
        text = "\n".join(t for _, t in kept)
        lo, hi = kept[0][0], kept[-1][0]
        # probe-proverb guard is LOAD-BEARING here (these are fables)
        if _has_probe(_pnorm(text)):
            probe_hits += 1
            continue
        if len(text) < 80:
            continue
        recs.append(_doc(
            "cric_crac_1901", text,
            f"cric_crac_1901:p{lo:03d}-{hi:03d}", "historical_literary",
            "historical", "authored_kreyol",
            url="https://dloc.com/UF00076576/00001",
            orthography="pre_reform", page_span=[lo, hi]))
    common.write_jsonl(os.path.join(OUT, "cric_crac_1901.jsonl"), recs)

    if held_pages:
        held_pages.sort()
        htext = "\n".join(t for _, t in held_pages)
        with open(os.path.join(OUT, "_cric_heritage_fable.json"), "w", encoding="utf-8") as f:
            json.dump({"span": list(HERITAGE_FABLE_SPAN),
                       "orders": [o for o, _ in held_pages],
                       "text": htext, "tokens_kb": kb.count(htext)},
                      f, ensure_ascii=False, indent=1)
    stats["cric_crac_1901"] = {
        "fable_runs": len(segs), "docs": len(recs),
        "tokens_kb": sum(kb.count(r["text"]) for r in recs),
        "probe_pages_excluded": probe_hits,
        "heritage_fable_span": list(HERITAGE_FABLE_SPAN) if HERITAGE_FABLE_SPAN else None,
        "verdict": "usable_rough_ocr",
    }
    common.log(f"[sweep4] cric: {len(recs)} fable-run docs, "
               f"{probe_hits} probe pages excluded, heritage span {HERITAGE_FABLE_SPAN}")
    return len(recs)


# --------------------------------------------------------------------------- #
# 3. Anthologie 1925 (Manioc CC0) — clean PDF text layer, Kreyòl passages only
# --------------------------------------------------------------------------- #
def anthologie(stats: dict) -> int:
    import fitz
    doc = fitz.open(os.path.join(SW, "anthologie_1925.pdf"))
    kb = _kb()
    recs, probe_hits, ht_pages = [], 0, 0
    for i in range(doc.page_count):
        txt, K, F = _page_kreyol_text(doc[i].get_text())
        if not (K >= 8 and K > F * 0.8 and len(txt) > 120):
            continue
        ht_pages += 1
        if _has_probe(_pnorm(txt)):
            probe_hits += 1
            continue
        recs.append(_doc(
            "anthologie_1925", txt, f"anthologie_1925:p{i:03d}",
            "historical_literary", "historical", "authored_kreyol",
            url="https://www.manioc.org/patrimon/PAP11095",
            orthography="pre_reform", pdf_page=i))
    common.write_jsonl(os.path.join(OUT, "anthologie_1925.jsonl"), recs)
    stats["anthologie_1925"] = {
        "pdf_pages": doc.page_count, "kreyol_dense_pages": ht_pages,
        "docs": len(recs), "tokens_kb": sum(kb.count(r["text"]) for r in recs),
        "probe_pages_excluded": probe_hits, "verdict": "usable_clean_textlayer",
    }
    common.log(f"[sweep4] anthologie: {ht_pages} Kreyòl-dense pages -> {len(recs)} docs")
    return len(recs)


# --------------------------------------------------------------------------- #
# 4. Lessons in Haitian Creole 1921 — measured, then ABANDONED (OCR quality)
# --------------------------------------------------------------------------- #
def lessons(stats: dict) -> int:
    path = os.path.join(SW, "lessons_1921_djvu.txt")
    lines = open(path, encoding="utf-8", errors="replace").read().split("\n")
    kre = [_clean_line(l) for l in lines if _is_kreyol_line(l)]
    kre = [k for k in kre if k]
    clean_chars = sum(len(k) for k in kre)
    stats["lessons_haitian_1921"] = {
        "total_lines": len(lines), "kreyol_lines": len(kre),
        "clean_kreyol_chars": clean_chars, "docs": 0,
        "verdict": "abandoned",
        "reason": ("English pedagogical manual; Kreyòl appears only as isolated "
                   "vocabulary entries and verb-conjugation paradigm cells (1:1 "
                   "with English prompts), heavily OCR-corrupted. No extractable "
                   "Kreyòl prose (densest contiguous block ~a 'vini' conjugation "
                   "drill). Not ingested to avoid injecting English + OCR noise."),
    }
    common.log(f"[sweep4] lessons: ABANDONED "
               f"({len(kre)} kreyol-ish lines / {clean_chars} clean chars)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="?", default="all",
                    choices=["cmu", "cric", "cric-list", "anthologie", "lessons", "all"])
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if args.which == "cric-list":
        cric({}, list_only=True)
        return
    stats_path = os.path.join(common.config.DATA, "interim", "sweep4_stats.json")
    stats = {}
    if os.path.exists(stats_path):
        stats = json.load(open(stats_path, encoding="utf-8"))
    if args.which in ("cmu", "all"):
        cmu(stats)
    if args.which in ("cric", "all"):
        cric(stats)
    if args.which in ("anthologie", "all"):
        anthologie(stats)
    if args.which in ("lessons", "all"):
        lessons(stats)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    common.log(f"[sweep4] stats -> {stats_path}")


if __name__ == "__main__":
    main()
