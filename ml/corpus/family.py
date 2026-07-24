"""Workstream J3 — family-contributed set (local only; NEVER committed).

Processes the six documents in ml/data/local/family-v1/ per the docs/data.md §2 J3
table and the rights.yaml verdicts, then writes a contributor record (chain-of-
custody). Copyrighted / eval-only items stay OUT of every training artifact
regardless of possession.

Per-file (verdicts from rights.yaml):
  cfpb_glossary_family  TRAIN-OK (federal PD) -> ingest text (train) + mine EN<->HT
                        pairs into the COMMITTABLE corpus/glossary_pairs_federal.json
  nj_court_glossary     EVAL-ONLY (state) -> legal-terminology probe pool (git-ignored)
  ma_dese_iep           EVAL-ONLY (state) -> education-terminology probe pool
  bmc_clinical_glossary EVAL-ONLY (courtesy) -> clinical probe pool + ORTHOGRAPHY AUDIT
                        (nonstandard French-style accents: santé/rivé/réchèch — tag,
                        do NOT normalize)
  family_file_7167      EVAL-ONLY (Beverly Hospital, private) -> recorded, not extracted
  freeman_medical_dict  QUARANTINE (copyrighted) -> recorded, NOT extracted

Run:  uv run python -m corpus.family
Nothing under ml/data/ is committed; the pairs JSON + contributor record are the
only committable outputs (both are federal-PD or metadata).
"""

from __future__ import annotations

import json
import os
import re

from . import common, schema
from . import config_v0_2 as C2

FILES = {
    "cfpb_glossary_family": "cfpb_adult-fin-ed_hatiancreole-style-guide-glossary.pdf",
    "nj_court_glossary": "11783_glossary_haitian_COURT TERMS.pdf",
    "ma_dese_iep": "iep-form-haitiancreole (1).pdf",
    "bmc_clinical_glossary": "Clinical_Trial_Glossary-English-Haitian Creole.pdf",
    "family_file_7167": "FILE_7167.pdf",
    "freeman_medical_dictionary": "Haitian-English Dictionnary Bryant C. Freeman.pdf",
}

EVAL_DIR = os.path.join(common.config.DATA, "eval")
PROBE_MANIFESTS = {
    "nj_court_glossary": os.path.join(EVAL_DIR, "probe_nj_court_terms.jsonl"),
    "ma_dese_iep": os.path.join(EVAL_DIR, "probe_ma_iep_terms.jsonl"),
    "bmc_clinical_glossary": os.path.join(EVAL_DIR, "probe_bmc_clinical_terms.jsonl"),
}

# --- helpers ---------------------------------------------------------------

def _fpath(key: str) -> str:
    return os.path.join(C2.FAMILY_DIR, FILES[key])


def extract_text(path: str) -> str:
    import fitz
    doc = fitz.open(path)
    parts = [doc[i].get_text() for i in range(doc.page_count)]
    doc.close()
    return "\n".join(parts)


# heuristics to keep pair extraction high-precision (curated artifact)
_HT_HINT = re.compile(r"\b(yon|nan|pou|oswa|ak|ki|nou|gen|sou|se|pa|lan|an|"
                      r"tankou|elatriye|dwe|kapab)\b", re.I)
_EN_HINT = re.compile(r"\b(the|of|for|and|or|to|as|in|your|with|by|a|an)\b", re.I)


def two_col_pairs(path: str, skip_pages: int = 0, band_pt: int = 3,
                  header_re: str | None = None) -> list[dict]:
    """Extract EN<->HT pairs from a 2-column glossary PDF. Words are split by the
    page midline (EN left / HT right) and clustered into ~band_pt horizontal rows;
    only clean single-row entries with both halves non-empty are kept (precision
    over recall for a committable curated artifact)."""
    import pdfplumber
    hdr = re.compile(header_re) if header_re else None
    pairs: list[dict] = []
    seen = set()
    with pdfplumber.open(path) as pdf:
        for pg in pdf.pages[skip_pages:]:
            split = pg.width / 2
            rows: dict[int, dict] = {}
            for w in pg.extract_words(keep_blank_chars=False):
                xm = (w["x0"] + w["x1"]) / 2
                key = round(w["top"] / band_pt)
                rows.setdefault(key, {"L": [], "R": []})
                side = "L" if xm < split else "R"
                rows[key][side].append((w["x0"], w["text"]))
            for k in sorted(rows):
                en = " ".join(t for _, t in sorted(rows[k]["L"])).strip()
                ht = " ".join(t for _, t in sorted(rows[k]["R"])).strip()
                if not en or not ht:
                    continue
                if hdr and (hdr.search(en) or hdr.search(ht)):
                    continue
                # column-header row that repeats atop glossary pages
                if en.lower() == "english" and ht.lower() in (
                        "haitian creole", "kreyòl ayisyen", "kreyol ayisyen"):
                    continue
                if len(en) < 2 or len(ht) < 2 or len(en) > 90 or len(ht) > 120:
                    continue
                # drop obvious wrap-continuations (start lowercase / bracket close)
                if en[0].islower() or en[0] in ")]}" or ht[0] in ")]}":
                    continue
                # single-letter section headers ("A", "B", "C")
                if len(en) == 1:
                    continue
                keyp = (en.lower(), ht.lower())
                if keyp in seen:
                    continue
                seen.add(keyp)
                pairs.append({"en": en, "ht": ht})
    return pairs


# French-style (nonstandard) orthography markers for the BMC audit. Standard
# Haitian Creole uses è/e/ò; these are French-accent spellings.
_FR_ACCENT = re.compile(r"[éàâêîôûëïüÉÀ]")
_FR_WORDS = re.compile(r"\b\w*(?:é|â|ê|î|ô|û)\w*\b")


def orthography_audit(text: str) -> dict:
    words = re.findall(r"\b[^\W\d_]+\b", text, re.UNICODE)
    fr_words = [w for w in words if _FR_WORDS.fullmatch(w)]
    n = len(words) or 1
    examples = sorted(set(w for w in fr_words if len(w) > 2))[:15]
    return {
        "words": len(words),
        "fr_accent_chars": len(_FR_ACCENT.findall(text)),
        "fr_accent_words": len(fr_words),
        "fr_accent_word_frac": len(fr_words) / n,
        "examples": examples,
        "nonstandard": len(fr_words) / n > 0.02,
    }


def _ingest_doc(key: str, text: str, doc_id: str, register: str, genre: str,
                origin: str, split: str = "train") -> dict:
    return schema.Document(
        text=text, origin=origin, genre=genre,
        acquisition={
            "source": key, "source_name": key,
            "url": None, "revision": None,
            "download_timestamp": common.now_iso(),
            "doc_id": doc_id, "raw_content_hash": common.content_hash(text),
        },
        rights=common.rights_for(key), split=split,
    ).model_dump(mode="json") | {"register": register}


def _write_probe_pool(key: str, pairs: list[dict], extra_tag: dict | None = None):
    os.makedirs(EVAL_DIR, exist_ok=True)
    path = PROBE_MANIFESTS[key]
    with open(path, "w", encoding="utf-8") as f:
        for i, p in enumerate(pairs):
            rec = {"source": key, "id": f"{key}:{i:04d}", **p, "split": "exhibit_examples"}
            if extra_tag:
                rec.update(extra_tag)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


# --- main ------------------------------------------------------------------

def process() -> dict:
    os.makedirs(C2.V0_2_INGEST, exist_ok=True)
    stats: dict = {"files": {}}

    # 1) CFPB — federal PD: mine pairs (committable) + ingest text (train)
    cfpb_pairs = two_col_pairs(_fpath("cfpb_glossary_family"), skip_pages=2,
                               header_re=r"GLOSSARY OF FINANCIAL TERMS|CONSUMER FINANCIAL")
    cfpb_text = extract_text(_fpath("cfpb_glossary_family"))
    ing = os.path.join(C2.V0_2_INGEST, "cfpb_glossary_family.jsonl")
    common.write_jsonl(ing, [_ingest_doc("cfpb_glossary_family", cfpb_text,
                             "cfpb_glossary_family:0000", "financial", "dictionary",
                             "human_translation")])
    stats["files"]["cfpb_glossary_family"] = {
        "verdict": "TRAIN-OK (federal PD)", "pairs": len(cfpb_pairs),
        "ingest_chars": len(cfpb_text)}

    # 2) NJ court — EVAL-ONLY: legal-terminology probe pool
    nj_pairs = two_col_pairs(_fpath("nj_court_glossary"), skip_pages=4,
                             header_re=r"Glossary of Legal|Revised|Revise|CN 11783|page \d|paj \d")
    _write_probe_pool("nj_court_glossary", nj_pairs, {"register": "legal"})
    stats["files"]["nj_court_glossary"] = {
        "verdict": "EVAL-ONLY (state)", "probe_terms": len(nj_pairs)}

    # 3) MA DESE IEP — EVAL-ONLY: mostly form labels; keep as tiny probe pool
    ma_text = extract_text(_fpath("ma_dese_iep"))
    # IEP is mostly form labels/blanks; keep non-blank Kreyòl label lines
    ma_labels = [{"ht": ln.strip()} for ln in ma_text.splitlines()
                 if 8 < len(ln.strip()) < 80 and "_" not in ln]
    _write_probe_pool("ma_dese_iep", ma_labels[:200], {"register": "education"})
    stats["files"]["ma_dese_iep"] = {
        "verdict": "EVAL-ONLY (state)", "probe_labels": len(ma_labels[:200])}

    # 4) BMC clinical — EVAL-ONLY + ORTHOGRAPHY AUDIT
    bmc_text = extract_text(_fpath("bmc_clinical_glossary"))
    audit = orthography_audit(bmc_text)
    # BMC layout: EN term / definition / Kreyòl term / Senaryo — not a clean 2-col;
    # keep the whole text as a tagged eval probe (nonstandard orthography).
    _write_probe_pool("bmc_clinical_glossary",
                      [{"ht": p.strip()} for p in bmc_text.split("\n\n") if len(p.strip()) > 20][:200],
                      {"register": "clinical", "tag": "nonstandard_orthography"})
    stats["files"]["bmc_clinical_glossary"] = {
        "verdict": "EVAL-ONLY (courtesy)", "orthography_audit": audit}

    # 5) FILE_7167 — EVAL-ONLY (Beverly Hospital, private): recorded, NOT extracted
    stats["files"]["family_file_7167"] = {
        "verdict": "EVAL-ONLY (private hospital — Beverly Hospital / Beth Israel Lahey Health)",
        "producer_identified": "Beverly Hospital (Beth Israel Lahey Health)",
        "action": "recorded only; not extracted into any artifact"}

    # 6) Freeman — QUARANTINE: recorded, NOT extracted
    stats["files"]["freeman_medical_dictionary"] = {
        "verdict": "QUARANTINE (copyrighted, Bryant C. Freeman)",
        "action": "recorded only; not opened/extracted"}

    # committable federal pairs file (CFPB now; IRS Pub 850 appended in J2)
    _write_glossary_pairs_federal({"cfpb_glossary_family": cfpb_pairs})
    _write_contributor_record(stats)

    with open(os.path.join(common.config.DATA, "interim", "family_stats.json"),
              "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    common.log(f"[family] CFPB pairs={len(cfpb_pairs)} NJ probe={len(nj_pairs)} "
               f"BMC nonstandard_orthography={audit['nonstandard']} "
               f"(fr_accent_word_frac={audit['fr_accent_word_frac']:.3f})")
    return stats


def _write_glossary_pairs_federal(by_source: dict[str, list[dict]]):
    """Committable curated EN<->HT pairs from FEDERAL PD sources only."""
    existing = {"sources": {}, "pairs": []}
    if os.path.exists(C2.GLOSSARY_PAIRS_FEDERAL):
        with open(C2.GLOSSARY_PAIRS_FEDERAL, encoding="utf-8") as f:
            existing = json.load(f)
    for src, pairs in by_source.items():
        existing["sources"][src] = {
            "license": "US-GOV-PD (17 U.S.C. §105)", "count": len(pairs)}
        # replace this source's pairs
        existing["pairs"] = [p for p in existing["pairs"] if p.get("source") != src]
        existing["pairs"] += [{"source": src, **p} for p in pairs]
    existing["total_pairs"] = len(existing["pairs"])
    existing["note"] = ("EN<->HT term pairs mined from FEDERAL public-domain "
                        "glossaries (17 U.S.C. §105). Committable. Feeds Workstream I "
                        "SFT translation turns. Rebuilt by corpus/family.py + J2 IRS.")
    os.makedirs(os.path.dirname(C2.GLOSSARY_PAIRS_FEDERAL), exist_ok=True)
    with open(C2.GLOSSARY_PAIRS_FEDERAL, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=1)
    common.log(f"[family] wrote {C2.GLOSSARY_PAIRS_FEDERAL} "
               f"({existing['total_pairs']} pairs)")


def _write_contributor_record(stats: dict):
    os.makedirs(C2.CONTRIBUTORS_DIR, exist_ok=True)
    path = os.path.join(C2.CONTRIBUTORS_DIR, "family-contributor-1.md")
    L = []
    A = L.append
    A("# Contributor record — family contributor #1")
    A("")
    A("*Chain-of-custody for third-party works contributed to the project. Per "
      "docs/plan.md §10 governance. This is a record of CUSTODY, **not** a rights "
      "grant: copyrighted / eval-only items stay quarantined from training "
      "regardless of possession. No personal name is recorded beyond the "
      "placeholder the contributor has approved.*")
    A("")
    A("| field | value |")
    A("|---|---|")
    A("| Contributor | family contributor #1 (placeholder; real name withheld pending approval) |")
    A("| Role | professional interpreter / translator (reference collection) |")
    A("| Date received | 2026-07-23 (files copied to ml/data/local/family-v1/, untracked) |")
    A("| Nature | chain-of-custody of third-party works — NOT a rights grant |")
    A("")
    A("## Per-file provenance + rights outcome")
    A("")
    A("| file | producer | rights verdict | outcome |")
    A("|---|---|---|---|")
    A("| cfpb_adult-fin-ed…glossary.pdf | Consumer Financial Protection Bureau (federal) | "
      "TRAIN-OK (17 U.S.C. §105 PD) | ingested (train) + EN↔HT pairs → committable federal pairs file |")
    A("| 11783_glossary…COURT TERMS.pdf | New Jersey Judiciary (state) | "
      "PERMISSION-ROUTE (state work ≠ PD) | EVAL-ONLY legal-terminology probe pool; terms-check email pending |")
    A("| iep-form-haitiancreole.pdf | MA DESE (state) | PERMISSION-ROUTE | "
      "EVAL-ONLY education probe pool (form labels); tiny |")
    A("| Clinical_Trial_Glossary…pdf | Boston Medical Center (\"Courtesy\") | "
      "PERMISSION-ROUTE (courtesy ≠ license) | EVAL-ONLY clinical probe pool; **nonstandard orthography tagged** |")
    A("| FILE_7167.pdf | **Beverly Hospital (Beth Israel Lahey Health)** — identified from page-1 logo | "
      "PRIVATE, not federal PD | EVAL-ONLY pending permission; **not extracted** into any artifact |")
    A("| Haitian-English Dictionary (Bryant C. Freeman) | published, copyrighted | "
      "QUARANTINE | recorded only; **not opened/extracted** |")
    A("")
    A("## Notes")
    A("")
    A("- Only the **CFPB** glossary (federal PD) enters training; its EN↔HT pairs are "
      "the sole family-set text that becomes a committed artifact "
      "(`corpus/glossary_pairs_federal.json`).")
    A("- The three state/private/courtesy items are held **EVAL-ONLY** as terminology "
      "probe pools (registered in `splits.yaml` `eval_slices_v0_2.terminology_probes`); "
      "the permission emails are a J5 human action.")
    bmc = stats["files"]["bmc_clinical_glossary"]["orthography_audit"]
    A(f"- **BMC orthography audit:** {bmc['fr_accent_word_frac']:.1%} of words carry "
      f"French-style accents (e.g. {', '.join(bmc['examples'][:6])}) → "
      f"tagged `nonstandard_orthography`, **not** silently normalized (measure-don't-filter).")
    A("- The Freeman dictionary was **not opened**; its quarantine is independent of possession.")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    common.log(f"[family] wrote {path}")


if __name__ == "__main__":
    process()
