"""Workstream H, Part 1 — the English-24k ABLATION tokenizer for Q1.

Q1 asks whether the *Kreyòl vocabulary* (not just its byte-efficiency) improves
LEARNING. The control is a tokenizer identical to kreyol-bpe in every way EXCEPT the
learned vocabulary: same algorithm (rustbpe), same vocab size (24,576 incl. the 9
special tokens), same `kreyol_aware` pre-tokenizer pattern — but trained on ENGLISH
text instead of the Kreyòl corpus. The only degree of freedom is which merges the BPE
trainer learns.

Workstream B trained only a 16k English tokenizer in-memory (the whole-word "shatter"
ablation, `tokenizer/run.run_ablation`) and never exported it. This trains + exports a
full 24,576-vocab bundle (tokenizer.pkl / tokenizer.json / meta.json) under
`tokenizer/english-24k/`, matching the kreyol-bpe artifact layout so the fleet data
pipeline can load it exactly like kreyol-bpe.

English source: fineweb English (`HuggingFaceFW/fineweb`, sample-10BT) — web-crawl
text, the same genre that dominates the Kreyòl corpus kreyol-bpe was trained on — with
an automatic fallback to wikitext-103-raw (the already-wired Workstream-B English
sample) if the stream is unavailable. Sample size matches kreyol-bpe's
`TRAIN_SAMPLE_CHARS` (120M chars).

Run:  cd ml && uv run python -m train.fleet_tokenizer
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # ml/

from tokenizer import config as TB     # Workstream-B tokenizer config (pattern, vocab, sample size)
from tokenizer import convert as TCONV
from tokenizer.core import KreyolBPE

OUT_DIR = os.path.join(TB.TOK_DIR, "english-24k")     # committed ablation artifact
SAMPLE_CACHE = os.path.join(TB.WORK, "english_fineweb_sample.txt")
FINEWEB_REPO = "HuggingFaceFW/fineweb"
FINEWEB_NAME = "sample-10BT"
TARGET_CHARS = TB.TRAIN_SAMPLE_CHARS                  # 120M chars — size-matched to kreyol-bpe


def _load_env():
    root = os.path.dirname(TB.REPO_ROOT)
    path = os.path.join(root, ".env")
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _nfc(s: str) -> str:
    import unicodedata
    return unicodedata.normalize("NFC", s)


def load_english_sample(target_chars: int = TARGET_CHARS) -> dict:
    """~target_chars of English web text. fineweb (web-crawl, genre-matched) → fallback
    wikitext-103 (encyclopedic). Cached to WORK; records which source actually served."""
    if os.path.exists(SAMPLE_CACHE):
        with open(SAMPLE_CACHE, encoding="utf-8") as f:
            txt = f.read()
        src = "fineweb" if os.path.exists(SAMPLE_CACHE + ".fineweb") else "wikitext103"
        print(f"[en-tok] using cached English sample ({len(txt)/1e6:.1f}M chars, {src})")
        return {"text": txt, "source": src}

    _load_env()
    text, source = None, None
    try:
        from datasets import load_dataset
        print(f"[en-tok] streaming {FINEWEB_REPO}/{FINEWEB_NAME} for {target_chars/1e6:.0f}M chars …")
        ds = load_dataset(FINEWEB_REPO, name=FINEWEB_NAME, split="train",
                          streaming=True, token=os.environ.get("HF_TOKEN"))
        buf, n = [], 0
        for row in ds:
            t = row.get("text", "")
            if not t or not t.strip():
                continue
            buf.append(t)
            n += len(t)
            if n >= target_chars:
                break
        if n >= target_chars * 0.9:
            text = _nfc("\n".join(buf))[:target_chars]
            source = "fineweb"
    except Exception as e:
        print(f"[en-tok] fineweb stream failed ({type(e).__name__}: {str(e)[:140]}); falling back")

    if text is None:
        from tokenizer.external import load_wikitext_sample
        text = load_wikitext_sample(target_chars)
        source = "wikitext103"

    os.makedirs(TB.WORK, exist_ok=True)
    with open(SAMPLE_CACHE, "w", encoding="utf-8") as f:
        f.write(text)
    if source == "fineweb":
        open(SAMPLE_CACHE + ".fineweb", "w").close()
    print(f"[en-tok] English sample: {len(text)/1e6:.1f}M chars from {source}")
    return {"text": text, "source": source}


def train_and_export() -> dict:
    sample = load_english_sample()
    print(f"[en-tok] training rustbpe: vocab={TB.CHOSEN_VOCAB or 24576}, "
          f"pattern={TB.CHOSEN_PATTERN_NAME} (SAME as kreyol-bpe), on ENGLISH")
    vocab = 24576   # == kreyol-bpe (F.VOCAB_SIZE): 24,567 content + 9 special
    t0 = time.time()
    kb = KreyolBPE.train(iter([sample["text"]]), vocab, TB.CHOSEN_SPLIT_PATTERN)
    train_s = round(time.time() - t0, 1)
    assert kb.vocab_size == vocab, f"vocab mismatch: {kb.vocab_size} != {vocab}"

    os.makedirs(OUT_DIR, exist_ok=True)
    kb.save_pkl(os.path.join(OUT_DIR, "tokenizer.pkl"))
    TCONV.export_hf(kb, os.path.join(OUT_DIR, "tokenizer.json"))
    kb.save_meta(os.path.join(OUT_DIR, "meta.json"),
                 extra={"role": "Q1 English-control ablation tokenizer",
                        "trained_on": "english", "english_source": sample["source"],
                        "sample_chars": len(sample["text"]),
                        "same_as_kreyol_bpe": ["vocab_size", "pattern", "algorithm(rustbpe)",
                                               "special_tokens"],
                        "only_difference": "learned vocabulary (English merges, not Kreyòl)",
                        "train_seconds": train_s})
    # sanity: the two tokenizers must differ ONLY in merges — same size + pattern
    from tokenizer.core import KreyolBPE as KB2
    kbpe = KB2.load_pkl(TB.KREYOL_BPE_PKL) if hasattr(TB, "KREYOL_BPE_PKL") else None
    print(f"[en-tok] exported english-24k -> {OUT_DIR} "
          f"(vocab {kb.vocab_size}, {kb.n_content_tokens} content merges, {train_s}s)")
    # quick fertility contrast on a Kreyòl string (shows the ablation actually shatters Kreyòl)
    probe = "Dèyè mòn gen mòn. Lè chat pa la, rat pran kay. Mwen renmen peyi mwen anpil."
    print(f"[en-tok] Kreyòl-fertility check «{probe[:40]}…»: english-24k tokens={kb.count(probe)}")
    return {"out_dir": OUT_DIR, "vocab": kb.vocab_size, "english_source": sample["source"],
            "train_seconds": train_s, "sample_chars": len(sample["text"])}


if __name__ == "__main__":
    train_and_export()
