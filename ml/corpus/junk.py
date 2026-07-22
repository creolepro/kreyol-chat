"""Phase-1 Workstream E, stage 1 — junk filter → corpus v0.1.

A NEW pipeline stage layered on the frozen corpus-v0 shard: reads
`data/clean/corpus_v0-{tag}.jsonl`, drops documents matching the audit's
deterministic MT/CMS junk fingerprints, and writes `corpus_v0_1-{tag}.jsonl`.

v0.1 is a strict document-SUBSET of v0 (drop-only) — every surviving document is
byte-identical to its v0 form, so provenance/hashes carry over and the size
deltas are exact. The raw v0 build is untouched and stays reproducible.

Filters (each calibrated to fire on ZERO human-labeled *natural* crawl docs; see
config_v0_1.JUNK): MT placeholder (XNUMX/XNMX), commercial/gambling/pharma spam,
price-listing density, residual HTML entities, foreign-script dominance. Bot-stubs
stay IN (flagged) — their fate is fleet Q4's, not this stage's.

Run:  python -m corpus.junk [--sample]
"""

from __future__ import annotations

import argparse
import json
import os
import re

from . import common, config
from . import config_v0_1 as CV

# --- fingerprint regexes ------------------------------------------------------
_ENTITY_RE = re.compile(
    r"&#\d+;|&#x[0-9a-fA-F]+;|"
    r"&(?:amp|quot|lt|gt|nbsp|apos|hellip|mdash|ndash|rsquo|lsquo|ldquo|rdquo|"
    r"laquo|raquo|copy|reg|trade|deg|middot|bull|eacute|egrave|agrave|ccedil);"
)
_XNUM_RE = re.compile(r"XNUMX|XNMX|\bXN[UM]{2,3}X?\b")
_PRICE_RE = re.compile(r"(?:US\$|\$|€|£|Rs\.?|USD|EUR|HTG|Gourdes?)\s?\d")
_SPAM_RES = {w: re.compile(r"\b" + re.escape(w) + r"\b", re.I) for w in CV.SPAM_MARKERS}


def _is_latin(cp: int) -> bool:
    return (0x41 <= cp <= 0x5A or 0x61 <= cp <= 0x7A or 0xC0 <= cp <= 0x24F
            or 0x1E00 <= cp <= 0x1EFF or 0x2C60 <= cp <= 0x2C7F)


def _foreign_script_ratio(text: str) -> float:
    """Fraction of alphabetic chars that are NOT Latin-script (Kreyòl is Latin)."""
    nalpha = nonlatin = 0
    for c in text:
        if c.isalpha():
            nalpha += 1
            if not _is_latin(ord(c)):
                nonlatin += 1
    return (nonlatin / nalpha) if nalpha else 0.0


def junk_reason(text: str, prio_class: str) -> str | None:
    """First tripped junk filter (deterministic order), or None. `prio_class` is
    the source's priority class (crawl/wikipedia/owned) for source-scoped rules."""
    def applies(rule):
        srcs = CV.JUNK[rule]["sources"]
        return srcs is None or prio_class in srcs

    chars = len(text)

    if applies("mt_placeholder"):
        if len(_XNUM_RE.findall(text)) >= CV.JUNK["mt_placeholder"]["min_count"]:
            return "mt_placeholder"

    if applies("commercial_spam"):
        distinct = sum(1 for rgx in _SPAM_RES.values() if rgx.search(text))
        if distinct >= CV.JUNK["commercial_spam"]["min_distinct"]:
            return "commercial_spam"

    if applies("price_listing"):
        if len(_PRICE_RE.findall(text)) >= CV.JUNK["price_listing"]["min_count"]:
            return "price_listing"

    if applies("html_entity"):
        ne = len(_ENTITY_RE.findall(text))
        dens = (1000 * ne / chars) if chars else 0.0
        if ne >= CV.JUNK["html_entity"]["min_count"] and dens >= CV.JUNK["html_entity"]["min_density_per_1k"]:
            return "html_entity"

    if applies("foreign_script"):
        cfg = CV.JUNK["foreign_script"]
        if chars >= cfg["min_chars"] and _foreign_script_ratio(text[:cfg["scan_chars"]]) >= cfg["min_ratio"]:
            return "foreign_script"

    return None


def _units(text: str, tok) -> tuple[int, int, int, int]:
    return (common.n_bytes(text), common.n_chars(text),
            common.n_ws_words(text), tok.count(text))


def _blank_units() -> dict:
    return {"docs": 0, "bytes": 0, "chars": 0, "words": 0, "tokens": 0}


def _add(acc: dict, u: tuple):
    acc["docs"] += 1
    acc["bytes"] += u[0]; acc["chars"] += u[1]; acc["words"] += u[2]; acc["tokens"] += u[3]


def junk_stage(sample: bool) -> dict:
    tag = common.run_tag(sample)
    src = CV.CORPUS_V0.format(tag=tag)
    out = CV.CORPUS_V0_1.format(tag=tag)
    tok = common.RefTokenizer()

    reasons = list(CV.JUNK.keys())
    # per-source-class rollups
    classes = ["crawl", "wikipedia", "owned"]
    kept = {c: _blank_units() for c in classes}
    dropped = {c: _blank_units() for c in classes}
    dropped_by_reason = {c: {r: _blank_units() for r in reasons} for c in classes}

    os.makedirs(os.path.dirname(out), exist_ok=True)
    n_out = 0
    with open(out, "w", encoding="utf-8") as fout:
        for d in common.read_jsonl(src):
            text = d["text"]
            cls = common.priority_class(d["acquisition"]["source"])
            u = _units(text, tok)
            reason = junk_reason(text, cls)
            if reason is None:
                _add(kept[cls], u)
                fout.write(json.dumps(d, ensure_ascii=False) + "\n")
                n_out += 1
            else:
                _add(dropped[cls], u)
                _add(dropped_by_reason[cls][reason], u)

    # totals + assembled stats
    def total(dd):
        t = _blank_units()
        for c in classes:
            for k in t:
                t[k] += dd[c][k]
        return t

    stats = {
        "snapshot_date": CV.SNAPSHOT_DATE,
        "input_shard": os.path.basename(src),
        "output_shard": os.path.basename(out),
        "reference_tokenizer": tok.name,
        "filters": CV.JUNK,
        "spam_markers": CV.SPAM_MARKERS,
        "v0_total": _sum2(total(kept), total(dropped)),
        "v0_1_total": total(kept),
        "removed_total": total(dropped),
        "by_class": {c: {"kept": kept[c], "removed": dropped[c],
                          "removed_by_reason": {r: dropped_by_reason[c][r] for r in reasons
                                                if dropped_by_reason[c][r]["docs"] > 0}}
                     for c in classes},
        "removed_by_reason_total": {r: _sum_over_classes(dropped_by_reason, r, classes)
                                    for r in reasons},
    }
    stats_path = CV.JUNK_STATS.format(tag=tag)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)

    rt = stats["removed_total"]
    v0 = stats["v0_total"]
    common.log(f"  junk -> {out}  (kept={n_out:,}/{v0['docs']:,}  "
               f"removed={rt['docs']:,} = {100 * rt['docs'] / max(1, v0['docs']):.2f}% docs, "
               f"{100 * rt['tokens'] / max(1, v0['tokens']):.2f}% o200k tokens)")
    for r in reasons:
        rr = stats["removed_by_reason_total"][r]
        common.log(f"      {r:16s} removed {rr['docs']:,} docs")
    return stats


def _sum2(a: dict, b: dict) -> dict:
    return {k: a[k] + b[k] for k in a}


def _sum_over_classes(dbr: dict, reason: str, classes) -> dict:
    t = _blank_units()
    for c in classes:
        for k in t:
            t[k] += dbr[c][reason][k]
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    args = ap.parse_args()
    common.log(f"[junk] sample={args.sample}")
    junk_stage(args.sample)


if __name__ == "__main__":
    main()
