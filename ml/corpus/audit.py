"""Stage 5 — quality audit. Deterministic stratified sample + machine first-pass.

docs/phase-0.md A2.5: a deterministic, stratified sample (~200 docs across
sources/length-bands), reviewed; report estimated rates of wrong-language,
boilerplate, unreadable, and translation-shaped text.

Here we do the MACHINE first pass (fasttext lid.176 language id + boilerplate /
unreadable heuristics) and clearly label the rates as machine-estimated, pending
human review. Two artifacts:
  * ml/reports/audit_sample.jsonl        — full records incl. text (git-ignored:
                                           *.jsonl is ignored repo-wide)
  * ml/reports/audit_sample_review.md    — committable: doc ids + metadata +
                                           machine flags + snippet. MADLAD
                                           snippets are REDACTED (rights.yaml:
                                           MADLAD redistribution is unresolved);
                                           full text stays in the git-ignored jsonl.

Run:  python -m corpus.audit [--sample]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re

from . import common, config

_LID = None


def _lid_predict(text: str):
    """(lang, confidence) top-1 via fasttext, numpy-2.x-safe (bypass np.array)."""
    global _LID
    if _LID is None:
        import fasttext
        _LID = fasttext.load_model(common.ensure_lid_model())
    t = text.replace("\n", " ").strip()
    if not t:
        return "und", 0.0
    preds = _LID.f.predict(t, 1, 0.0, "strict")  # [(prob, "__label__xx")]
    if not preds:
        return "und", 0.0
    prob, label = preds[0]
    return label.replace("__label__", ""), float(prob)


def _band(nchars: int) -> str:
    for name, lo, hi in config.AUDIT_LENGTH_BANDS:
        if lo <= nchars < hi:
            return name
    return config.AUDIT_LENGTH_BANDS[-1][0]


def _boilerplate_ratio(text: str) -> float:
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if len(lines) < 2:
        return 0.0
    return 1.0 - len(set(lines)) / len(lines)


def _symbol_ratio(text: str) -> float:
    if not text:
        return 1.0
    letters = sum(1 for c in text if c.isalpha())
    spaces = sum(1 for c in text if c.isspace())
    digits = sum(1 for c in text if c.isdigit())
    return (len(text) - letters - spaces - digits) / len(text)


def _stratified_sample(docs, size, seed):
    """Deterministic stratified sample across (source, length-band)."""
    strata = {}
    for d in docs:
        key = (d["acquisition"]["source"], _band(len(d["text"])))
        strata.setdefault(key, []).append(d)
    total = len(docs)
    picked = []
    for key in sorted(strata):
        bucket = sorted(strata[key], key=lambda d: d["acquisition"]["doc_id"])
        random.Random(f"{seed}:{key}").shuffle(bucket)
        take = max(1, round(size * len(bucket) / total))
        picked.append((key, bucket[:take]))
    # trim/pad deterministically to ~size
    flat = [d for _key, ds in picked for d in ds]
    flat.sort(key=lambda d: d["acquisition"]["doc_id"])
    random.Random(seed).shuffle(flat)
    return flat[:size]


def audit_stage(sample: bool):
    tag = common.run_tag(sample)
    corpus = os.path.join(config.CLEAN, f"corpus_v0-{tag}.jsonl")
    docs = list(common.read_jsonl(corpus))
    picked = _stratified_sample(docs, config.AUDIT_SAMPLE_SIZE, config.AUDIT_SEED)
    common.log(f"  audit: {len(picked)} docs sampled from {len(docs)}")

    records = []
    for d in picked:
        text = d["text"]
        lang, conf = _lid_predict(text)
        sym = _symbol_ratio(text)
        repl = text.count("�") / max(1, len(text))
        boiler = _boilerplate_ratio(text)
        flags = {
            "wrong_language": lang != config.TARGET_LANG,
            "low_confidence_ht": lang == config.TARGET_LANG and conf < config.AUDIT["min_langid_conf_for_ht"],
            "boilerplate": boiler > config.AUDIT["boilerplate_repeat_line_ratio"],
            "unreadable": sym > config.AUDIT["unreadable_symbol_ratio"] or repl > config.AUDIT["unreadable_replacement_ratio"],
            # translation-shaped can't be reliably machine-estimated; left for humans.
            "translation_shaped": None,
        }
        records.append({
            "doc_id": d["acquisition"]["doc_id"],
            "source": d["acquisition"]["source"],
            "origin": d["origin"],
            "genre": d["genre"],
            "length_band": _band(len(text)),
            "n_chars": len(text),
            "wiki_bot_stub": d.get("wiki_bot_stub"),
            "langid": lang,
            "langid_conf": round(conf, 3),
            "symbol_ratio": round(sym, 3),
            "boilerplate_ratio": round(boiler, 3),
            "flags": flags,
            "text": text,
            "snippet": text.replace("\n", " ")[:200],
        })

    # --- aggregate machine-estimated rates (overall + per source) -------------
    # langid is unreliable on short docs, so we also report wrong-language among
    # docs >= 300 chars — the more trustworthy estimate (see report).
    LONG = 300

    def rates(rows):
        n = len(rows) or 1
        keys = ["wrong_language", "low_confidence_ht", "boilerplate", "unreadable"]
        out = {k: round(sum(1 for r in rows if r["flags"][k]) / n, 4) for k in keys}
        longrows = [r for r in rows if r["n_chars"] >= LONG]
        out["n_ge300"] = len(longrows)
        out["wrong_language_ge300"] = (
            round(sum(1 for r in longrows if r["flags"]["wrong_language"]) / len(longrows), 4)
            if longrows else None)
        return out

    by_source = {}
    for r in records:
        by_source.setdefault(r["source"], []).append(r)
    summary = {
        "sampled": len(records),
        "target_size": config.AUDIT_SAMPLE_SIZE,
        "seed": config.AUDIT_SEED,
        "machine_estimated": True,
        "pending_human_review": True,
        "overall_rates": rates(records),
        "per_source_rates": {s: rates(rs) for s, rs in by_source.items()},
        "per_source_n": {s: len(rs) for s, rs in by_source.items()},
        "langid_note": "fasttext lid.176; short docs are unreliable, and its Kreyòl confidence runs modest — treat wrong-language as an UPPER bound pending human review.",
    }

    # --- full records (git-ignored: *.jsonl) ----------------------------------
    os.makedirs(config.REPORTS, exist_ok=True)
    full_name = "audit_sample-sample.jsonl" if sample else "audit_sample.jsonl"
    full_path = os.path.join(config.REPORTS, full_name)
    common.write_jsonl(full_path, records)
    summary_path = os.path.join(config.REPORTS, f"audit_summary-{tag}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)

    _write_review_md(records, summary, tag)
    common.log(f"  audit -> {full_path} (full, git-ignored) + review md; "
               f"overall {summary['overall_rates']}")
    return summary


# Sources whose license permits committing text excerpts. MADLAD is excluded
# (rights.yaml: redistribution unresolved) -> its snippets are redacted.
_SNIPPET_OK = {config.HTWIKI_SOURCE_KEY, config.PROVERBS_SOURCE_KEY}


def _write_review_md(records, summary, tag):
    path = os.path.join(config.REPORTS, "audit_sample_review.md")
    lines = [
        "# Corpus v0 — quality audit sample (machine first pass)",
        "",
        f"*Deterministic stratified sample of {summary['sampled']} docs "
        f"(seed {summary['seed']}), across sources × length bands. Build: `{tag}`. "
        f"Snapshot {config.SNAPSHOT_DATE}.*",
        "",
        "**These flags are MACHINE-ESTIMATED (fasttext lid.176 + heuristics) and "
        "PENDING HUMAN REVIEW.** langid is unreliable on short docs and runs modest "
        "on Kreyòl, so `wrong_language` is an upper bound. `translation_shaped` is "
        "left blank — it needs a human ear.",
        "",
        "> **MADLAD snippets are redacted.** rights.yaml marks MADLAD redistribution "
        "*unresolved* (CC-BY vs ODC-BY), so its text is not excerpted in a committed "
        "file. Full text for every row (MADLAD included) is in the git-ignored "
        "`ml/reports/audit_sample.jsonl` for local human review.",
        "",
        "## Machine-estimated rates",
        "",
        "`wrong_lang` is the raw langid flag; `wrong_lang≥300` restricts to docs "
        "≥300 chars (rules out the short-doc confound). Both are UPPER BOUNDS: "
        "eyeballing shows most flagged docs are genuinely Kreyòl encyclopedic stubs "
        "dense with French/Spanish proper nouns (foreign names, film titles) that "
        "fool lid.176 — i.e. this mostly measures langid's weakness on Kreyòl, not "
        "contamination. Human review needed.",
        "",
        "| scope | n | wrong_lang | wrong_lang≥300 (n) | low_conf_ht | boilerplate | unreadable |",
        "|---|--:|--:|--:|--:|--:|--:|",
    ]

    def _ge(r):
        v = r.get("wrong_language_ge300")
        return "—" if v is None else f"{v:.1%} ({r['n_ge300']})"

    o = summary["overall_rates"]
    lines.append(f"| **overall** | {summary['sampled']} | {o['wrong_language']:.1%} | {_ge(o)} | "
                 f"{o['low_confidence_ht']:.1%} | {o['boilerplate']:.1%} | {o['unreadable']:.1%} |")
    for s, r in summary["per_source_rates"].items():
        n = summary["per_source_n"][s]
        lines.append(f"| {s} | {n} | {r['wrong_language']:.1%} | {_ge(r)} | {r['low_confidence_ht']:.1%} "
                     f"| {r['boilerplate']:.1%} | {r['unreadable']:.1%} |")
    lines += [
        "",
        "## Sampled documents",
        "",
        "| doc_id | source | genre | band | langid (conf) | flags | snippet |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in sorted(records, key=lambda x: (x["source"], x["doc_id"])):
        flagset = [k for k, v in r["flags"].items() if v is True]
        flagstr = ", ".join(flagset) if flagset else "—"
        if r["source"] in _SNIPPET_OK:
            snip = r["snippet"].replace("|", "\\|")
        else:
            snip = "«redacted — MADLAD redistribution unresolved (see git-ignored jsonl)»"
        lines.append(f"| `{r['doc_id']}` | {r['source']} | {r['genre']} | {r['length_band']} "
                     f"| {r['langid']} ({r['langid_conf']:.2f}) | {flagstr} | {snip} |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    args = ap.parse_args()
    common.log(f"[audit] sample={args.sample}")
    audit_stage(args.sample)


if __name__ == "__main__":
    main()
