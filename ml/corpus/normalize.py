"""Stage 2 — normalize. NFC + whitespace, paragraph boundaries preserved.

Per docs/phase-0.md A2.2: NFC Unicode (è/ò consistency), strip control chars,
normalize spaces *within* lines but PRESERVE paragraph boundaries (structure is
signal). NO lowercasing, NO accent stripping. Records the cleaned-content hash.

Run:  python -m corpus.normalize [--sample]
"""

from __future__ import annotations

import argparse
import re
import unicodedata

from . import common, config

# Control chars to drop entirely (keep \n and \t; \t is turned into a space).
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# Runs of horizontal whitespace within a line -> single space. `[^\S\n]` is
# "any whitespace except newline" (covers tab, NBSP U+00A0, U+2000-200A, U+3000).
_INLINE_WS_RE = re.compile(r"[^\S\n]+")
# 3+ newlines -> exactly two (one blank line == a paragraph boundary).
_MULTI_NL_RE = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = _CTRL_RE.sub("", text)
    lines = []
    for line in text.split("\n"):
        line = _INLINE_WS_RE.sub(" ", line).strip()
        lines.append(line)
    text = "\n".join(lines)
    text = _MULTI_NL_RE.sub(config.PARAGRAPH_SEP, text)
    return text.strip("\n").strip()


def n_paragraphs(text: str) -> int:
    return sum(1 for p in text.split(config.PARAGRAPH_SEP) if p.strip())


def normalize_stage(sample: bool, source: str):
    tag = common.run_tag(sample)
    src = common.stage_path(tag, "ingest", source)
    out = common.stage_path(tag, "normalize", source)

    def records():
        for r in common.read_jsonl(src):
            norm = normalize_text(r["text"])
            if not norm:
                continue
            r["text"] = norm
            r["acquisition"]["cleaned_content_hash"] = common.content_hash(norm)
            r["n_paragraphs"] = n_paragraphs(norm)
            yield r

    n = common.write_jsonl(out, records())
    common.log(f"  normalize {source} -> {out}  ({n} docs)")
    return out, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    args = ap.parse_args()
    common.log(f"[normalize] sample={args.sample}")
    for source in config.PIPELINE_SOURCES:
        normalize_stage(args.sample, source)


if __name__ == "__main__":
    main()
