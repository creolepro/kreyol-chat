"""Stage 3 — filter. Source-specific quality thresholds + Wikipedia bot-stub flag.

docs/phase-0.md A2.3: source-specific thresholds (a 20-char floor would delete
proverbs; crawl can afford stricter floors), max symbol/digit ratio, boilerplate
lines. Flag Wikipedia bot-stubs via simple heuristics — FLAG, do not drop (the
report reports their share).

Drop reasons + bot-stub counts are written to a per-source stats sidecar for the
report. Dropped documents are not carried forward (kept only as counts).

Run:  python -m corpus.filter [--sample]
"""

from __future__ import annotations

import argparse
import json
import os
import re

from . import common, config

_URL_LINE_RE = re.compile(r"^\s*https?://\S+\s*$", re.IGNORECASE)
_BREADCRUMB_RE = re.compile(r"(?:\s[>›»]\s).*(?:\s[>›»]\s)")  # >= 2 crumb separators
_NONWORD_LINE_RE = re.compile(r"^[\W\d_]+$")  # line with no letters at all
_GEO_STUB_RES = [re.compile(p, re.IGNORECASE) for p in config.BOTSTUB["geo_stub_patterns"]]


def _strip_boilerplate_lines(text: str) -> str:
    kept = []
    for ln in text.split("\n"):
        s = ln.strip()
        if not s:
            kept.append("")
            continue
        if _URL_LINE_RE.match(s) or _BREADCRUMB_RE.search(s) or _NONWORD_LINE_RE.match(s):
            continue
        kept.append(ln)
    # collapse blank runs the removal may have created
    out = re.sub(r"\n{3,}", config.PARAGRAPH_SEP, "\n".join(kept))
    return out.strip("\n").strip()


def _metrics(text: str) -> dict:
    chars = len(text)
    words = text.split()
    n_words = len(words)
    letters = sum(1 for c in text if c.isalpha())
    digits = sum(1 for c in text if c.isdigit())
    spaces = sum(1 for c in text if c.isspace())
    repl = text.count("�")
    symbol = chars - letters - digits - spaces
    return {
        "chars": chars,
        "n_words": n_words,
        "symbol_ratio": (symbol / chars) if chars else 1.0,
        "digit_ratio": (digits / chars) if chars else 0.0,
        "replacement_char_ratio": (repl / chars) if chars else 0.0,
        "mean_word_len": (sum(len(w) for w in words) / n_words) if n_words else 0.0,
    }


def _drop_reason(m: dict, th: dict):
    if m["chars"] < th["min_chars"]:
        return "too_short_chars"
    if m["n_words"] < th["min_words"]:
        return "too_few_words"
    if m["symbol_ratio"] > th["max_symbol_ratio"]:
        return "high_symbol_ratio"
    if m["digit_ratio"] > th["max_digit_ratio"]:
        return "high_digit_ratio"
    if m["replacement_char_ratio"] > th["max_replacement_char_ratio"]:
        return "mojibake"
    if m["mean_word_len"] < th["min_mean_word_len"]:
        return "degenerate_words"
    return None


def _is_bot_stub(rec: dict) -> bool:
    """Short + template-driven, or a classic geo/species stub opening."""
    plain = rec.get("wiki_plaintext_chars", len(rec["text"]))
    if plain > config.BOTSTUB["max_plaintext_chars"]:
        return False
    ntmpl = rec.get("wiki_n_templates", 0)
    tmpl_ratio = (ntmpl / plain) if plain else 0.0
    if tmpl_ratio >= config.BOTSTUB["min_template_ratio"]:
        return True
    head = rec["text"][:300]
    return any(r.search(head) for r in _GEO_STUB_RES)


def filter_stage(sample: bool, source: str):
    tag = common.run_tag(sample)
    src = common.stage_path(tag, "normalize", source)
    out = common.stage_path(tag, "filter", source)
    prio = common.priority_class(config.SOURCE_KEYS[source])
    th = config.FILTER[prio]
    stats = {"source": source, "priority_class": prio, "in": 0, "kept": 0,
             "dropped": {}, "bot_stubs": 0}

    def records():
        for r in common.read_jsonl(src):
            stats["in"] += 1
            text = _strip_boilerplate_lines(r["text"])
            if not text.strip():
                stats["dropped"]["empty_after_boilerplate"] = stats["dropped"].get("empty_after_boilerplate", 0) + 1
                continue
            r["text"] = text
            m = _metrics(text)
            reason = _drop_reason(m, th)
            if reason:
                stats["dropped"][reason] = stats["dropped"].get(reason, 0) + 1
                continue
            r["acquisition"]["cleaned_content_hash"] = common.content_hash(text)
            r["n_paragraphs"] = sum(1 for p in text.split(config.PARAGRAPH_SEP) if p.strip())
            if source == "htwiki":
                r["wiki_bot_stub"] = _is_bot_stub(r)
                if r["wiki_bot_stub"]:
                    stats["bot_stubs"] += 1
            stats["kept"] += 1
            yield r

    n = common.write_jsonl(out, records())
    stats_path = os.path.join(os.path.dirname(out), f"{source}.stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    common.log(f"  filter {source} -> {out}  (in={stats['in']} kept={n} "
               f"botstubs={stats['bot_stubs']} dropped={sum(stats['dropped'].values())})")
    return out, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    args = ap.parse_args()
    common.log(f"[filter] sample={args.sample}")
    for source in config.PIPELINE_SOURCES:
        filter_stage(args.sample, source)


if __name__ == "__main__":
    main()
