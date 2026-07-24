"""Authored vs translated Kreyòl fertility — closes the standing Workstream C TODO.

FLORES's Haitian side is itself translated, so Workstream C had no authored-Kreyòl
set of real size to compare against (only the 15 probe proverbs). Workstream J adds
one: `authored_eval_v2` (held-out VOA journalism). This measures whether our
kreyol-bpe tokenizer fits AUTHORED Kreyòl differently from TRANSLATED/translationese
Kreyòl (`translation_shaped_eval`), with `authored_eval` (Wikipedia) as a second
authored point.

Metric: tokens-per-byte (byte-normalized -> cross-tokenizer comparable, BPB-style)
and tokens-per-word, for kreyol-bpe / cl100k / o200k on each held-out slice. A lower
authored tokens-per-byte means the tokenizer (trained on our corpus) is not overfit
to translationese — it fits native journalism at least as well.

Run:  uv run python -m fertility.authored   (after corpus.build_v0_2 writes authored_eval_v2)
Writes: ml/data/interim/fertility_authored.json (git-ignored sidecar).
"""

from __future__ import annotations

import json
import os
import re

ML_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(ML_DIR, "data", "eval")
SIDECAR = os.path.join(ML_DIR, "data", "interim", "fertility_authored.json")

SLICES = {
    "authored_eval_v2": ("authored", "VOA journalism (native-authored, held out)"),
    "authored_eval": ("authored", "Wikipedia authored (v0.1 slice)"),
    "translation_shaped_eval": ("translated", "crawl translationese (v0.1 slice)"),
}

_WORD = re.compile(r"\S+")


def _load(name):
    p = os.path.join(EVAL_DIR, f"{name}.jsonl")
    if not os.path.exists(p):
        return None
    return [json.loads(l)["text"] for l in open(p, encoding="utf-8") if l.strip()]


def _counters():
    import tiktoken
    from tokenizer.core import KreyolBPE
    kb = KreyolBPE.load_pkl(os.path.join(ML_DIR, "tokenizer", "kreyol-bpe", "tokenizer.pkl"))
    cl = tiktoken.get_encoding("cl100k_base")
    o2 = tiktoken.get_encoding("o200k_base")
    return {
        "kreyol-bpe": lambda t: len(kb.enc.encode_ordinary(t)),
        "cl100k": lambda t: len(cl.encode(t, disallowed_special=())),
        "o200k": lambda t: len(o2.encode(t, disallowed_special=())),
    }


def measure() -> dict:
    counters = _counters()
    out = {"slices": {}, "counters": list(counters)}
    for name, (kind, desc) in SLICES.items():
        texts = _load(name)
        if not texts:
            out["slices"][name] = {"kind": kind, "desc": desc, "present": False}
            continue
        nbytes = sum(len(t.encode("utf-8")) for t in texts)
        nwords = sum(len(_WORD.findall(t)) for t in texts)
        row = {"kind": kind, "desc": desc, "present": True, "docs": len(texts),
               "bytes": nbytes, "words": nwords, "tok_per_byte": {}, "tok_per_word": {}}
        for cname, cnt in counters.items():
            toks = sum(cnt(t) for t in texts)
            row["tok_per_byte"][cname] = round(toks / nbytes, 5) if nbytes else 0.0
            row["tok_per_word"][cname] = round(toks / nwords, 4) if nwords else 0.0
        out["slices"][name] = row

    # authored-vs-translated gap for our tokenizer (primary: authored_eval_v2)
    tr = out["slices"].get("translation_shaped_eval", {})
    gaps = {}
    if tr.get("present"):
        for a in ("authored_eval_v2", "authored_eval"):
            av = out["slices"].get(a, {})
            if av.get("present"):
                gaps[a] = {
                    c: round(av["tok_per_byte"][c] - tr["tok_per_byte"][c], 5)
                    for c in counters
                }
    out["authored_minus_translated_tok_per_byte"] = gaps
    os.makedirs(os.path.dirname(SIDECAR), exist_ok=True)
    with open(SIDECAR, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"  wrote {SIDECAR}")
    for name, row in out["slices"].items():
        if row.get("present"):
            print(f"  {name:26s} kreyol-bpe tok/byte={row['tok_per_byte']['kreyol-bpe']:.4f} "
                  f"tok/word={row['tok_per_word']['kreyol-bpe']:.3f} ({row['docs']} docs)")
    return out


if __name__ == "__main__":
    measure()
