"""Corpus streaming, the tokenizer_eval holdout, weighted training samples, word freqs.

The corpus (data/clean/corpus_v0-full.jsonl) already excludes the 15 probe
proverbs by construction (they were held out in Workstream A), so no training or
eval text here can contain them. `assert_no_probe` re-checks that invariant.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import unicodedata

from . import config

SOURCE_CLASS = {
    "madlad_400_ht_clean": "crawl",
    "ht_wikipedia": "wikipedia",
    "owned_proverbs": "owned",
}


def log(msg: str):
    print(msg, file=sys.stderr, flush=True)


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def iter_docs(path: str = None):
    path = path or config.CORPUS
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _u01(key: str) -> float:
    """Deterministic uniform(0,1) from a string key."""
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF


def in_holdout(doc_id: str) -> bool:
    """Deterministic ~HOLDOUT_FRAC of docs -> tokenizer_eval holdout (NEVER trained).

    Uniform per-doc hash => each source contributes ~HOLDOUT_FRAC of its docs
    (stratified-by-source in expectation, exact at corpus scale).
    """
    return _u01(f"holdout:{config.SPLIT_SEED}:{doc_id}") < config.HOLDOUT_FRAC


# --- probe-proverb guard ------------------------------------------------------

_PROBE_TEXTS = None


def probe_texts():
    global _PROBE_TEXTS
    if _PROBE_TEXTS is None:
        _PROBE_TEXTS = set()
        if os.path.exists(config.PROVERBS_PROBE):
            with open(config.PROVERBS_PROBE, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        _PROBE_TEXTS.add(nfc(json.loads(line)["text"]).strip())
    return _PROBE_TEXTS


def assert_no_probe(texts):
    probe = probe_texts()
    for t in texts:
        if nfc(t).strip() in probe:
            raise AssertionError("probe proverb leaked into a tokenizer set!")


# --- per-source char totals (non-holdout), cached -----------------------------

def source_char_totals():
    cache = os.path.join(config.WORK, "source_char_totals.json")
    if os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)
    totals = {}
    for d in iter_docs():
        if in_holdout(d["acquisition"]["doc_id"]):
            continue
        cls = SOURCE_CLASS[d["acquisition"]["source"]]
        totals[cls] = totals.get(cls, 0) + len(d["text"])
    os.makedirs(config.WORK, exist_ok=True)
    with open(cache, "w") as f:
        json.dump(totals, f)
    return totals


# --- weighted training samples ------------------------------------------------

def _keep_prob(weighting: str, target_chars: int, totals: dict) -> dict:
    """Per-source-class keep probability to realize the requested weighting."""
    if weighting == "natural":
        total = sum(totals.values())
        p = min(1.0, target_chars / total)
        return {c: p for c in totals}
    if weighting == "sensitivity":
        # include ALL wikipedia + owned; downsample crawl so crawl ~= 60% of the
        # resulting sample (wiki+owned ~= 40%). proverbs are negligible by volume.
        non_crawl = totals.get("wikipedia", 0) + totals.get("owned", 0)
        crawl_target = 1.5 * non_crawl  # 60/40
        p_crawl = min(1.0, crawl_target / max(1, totals.get("crawl", 1)))
        return {"crawl": p_crawl, "wikipedia": 1.0, "owned": 1.0}
    raise ValueError(weighting)


def iter_training_texts(weighting: str, target_chars: int, seed: int):
    """Yield NFC texts for BPE training: non-holdout docs, per-source keep prob."""
    totals = source_char_totals()
    keep = _keep_prob(weighting, target_chars, totals)
    for d in iter_docs():
        did = d["acquisition"]["doc_id"]
        if in_holdout(did):
            continue
        cls = SOURCE_CLASS[d["acquisition"]["source"]]
        if _u01(f"sample:{weighting}:{seed}:{did}") < keep[cls]:
            yield nfc(d["text"])


def sample_composition(weighting: str, target_chars: int, seed: int):
    """Report the realized per-source char composition of a training sample."""
    totals = source_char_totals()
    keep = _keep_prob(weighting, target_chars, totals)
    comp = {}
    for d in iter_docs():
        did = d["acquisition"]["doc_id"]
        if in_holdout(did):
            continue
        cls = SOURCE_CLASS[d["acquisition"]["source"]]
        if _u01(f"sample:{weighting}:{seed}:{did}") < keep[cls]:
            comp[cls] = comp.get(cls, 0) + len(d["text"])
    return comp


# --- holdout docs (tokenizer_eval) --------------------------------------------

def holdout_docs():
    out = []
    for d in iter_docs():
        if in_holdout(d["acquisition"]["doc_id"]):
            out.append({"source": SOURCE_CLASS[d["acquisition"]["source"]],
                        "text": nfc(d["text"])})
    return out


# --- top-N corpus-frequency words (train split, probe excluded) ---------------

def top_words(n: int, seed: int = None):
    """Top-N whitespace words by frequency over the TRAIN split (holdout + probe
    excluded). Case-sensitive on NFC text (no lowercasing — matches corpus)."""
    from collections import Counter
    c = Counter()
    for d in iter_docs():
        if in_holdout(d["acquisition"]["doc_id"]):
            continue
        for w in nfc(d["text"]).split():
            c[w] += 1
    return [w for w, _ in c.most_common(n)], c


# --- probe set for B0 parity (mix of sources) ---------------------------------

def probe_lines(per_source: int = 350):
    """A source-mixed set of non-empty lines for the format-parity probe."""
    buckets = {"crawl": [], "wikipedia": [], "owned": []}
    for d in iter_docs():
        cls = SOURCE_CLASS[d["acquisition"]["source"]]
        if len(buckets[cls]) >= per_source:
            if all(len(v) >= per_source for v in buckets.values()):
                break
            continue
        for ln in nfc(d["text"]).split("\n"):
            ln = ln.strip()
            if len(ln) >= 12:
                buckets[cls].append(ln)
                if len(buckets[cls]) >= per_source:
                    break
    return buckets["crawl"] + buckets["wikipedia"] + buckets["owned"]
