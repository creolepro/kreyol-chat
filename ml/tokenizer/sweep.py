"""B2 — vocab sweep + decision/exhibit metrics.

Materializes seeded, source-weighted training samples (train split only; holdout
+ probe proverbs excluded), trains the {8,16,24,32}k sweep plus a 16k
crawl-downweighted sensitivity variant, and evaluates each on the held-out
tokenizer_eval slice + FLORES+ (measurement-only).

Metrics live here so the report is a thin renderer.
"""

from __future__ import annotations

import json
import os
import time

from . import config, data
from .core import KreyolBPE

# ---------------------------------------------------------------------------
# training-sample materialization
# ---------------------------------------------------------------------------

def materialize_sample(weighting: str, target_chars: int, seed: int) -> dict:
    path = os.path.join(config.WORK, f"sample_{weighting}_{target_chars//1_000_000}M.jsonl")
    meta_path = path + ".meta.json"
    if os.path.exists(path) and os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    os.makedirs(config.WORK, exist_ok=True)
    probe = data.probe_texts()
    totals = data.source_char_totals()
    keep = data._keep_prob(weighting, target_chars, totals)
    n = chars = leaked = 0
    comp = {}
    with open(path, "w", encoding="utf-8") as f:
        for d in data.iter_docs():
            did = d["acquisition"]["doc_id"]
            if data.in_holdout(did):
                continue
            cls = data.SOURCE_CLASS[d["acquisition"]["source"]]
            if data._u01(f"sample:{weighting}:{seed}:{did}") >= keep[cls]:
                continue
            t = data.nfc(d["text"])
            if t.strip() in probe:
                leaked += 1
                continue
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
            n += 1
            chars += len(t)
            comp[cls] = comp.get(cls, 0) + len(t)
    assert leaked == 0, f"{leaked} probe proverbs leaked into {weighting} sample!"
    meta = {"weighting": weighting, "docs": n, "chars": chars,
            "composition_chars": comp, "path": path}
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=1)
    data.log(f"  materialized {weighting}: {n:,} docs / {chars/1e6:.1f}M chars "
             f"comp={ {k: round(v/1e6,1) for k,v in comp.items()} }")
    return meta


def _iter_sample(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def train_one(sample_path: str, vocab: int, tag: str) -> dict:
    pkl = os.path.join(config.WORK, f"tok_{tag}_{vocab}.pkl")
    t0 = time.time()
    kb = KreyolBPE.train(_iter_sample(sample_path), vocab, config.CHOSEN_SPLIT_PATTERN)
    dt = time.time() - t0
    kb.save_pkl(pkl)
    data.log(f"  trained {tag} {vocab:,} in {dt:.1f}s -> {pkl}")
    return {"tag": tag, "vocab": vocab, "pkl": pkl, "train_seconds": round(dt, 1)}


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def compression(kbpe: KreyolBPE, texts) -> dict:
    """bytes-per-token (higher = better) + tokens & bytes totals."""
    nbytes = ntok = 0
    for t in texts:
        nbytes += len(t.encode("utf-8"))
        ntok += kbpe.count(t)
    return {"bytes": nbytes, "tokens": ntok,
            "bytes_per_token": (nbytes / ntok) if ntok else 0.0}


def roundtrip_ok(kbpe: KreyolBPE, texts, k=2000) -> dict:
    fails = 0
    n = 0
    for t in texts[:k]:
        n += 1
        if kbpe.decode(kbpe.encode_ordinary(t)) != t:
            fails += 1
    return {"checked": n, "failures": fails}


def embedding_cost(vocab: int) -> dict:
    """Token-embedding parameter cost at nanochat d12 width (wte + lm_head, untied)."""
    params = vocab * config.D12_MODEL_DIM * config.EMBED_MATRICES
    return {"vocab": vocab, "dim": config.D12_MODEL_DIM,
            "embed_params": params, "embed_params_m": round(params / 1e6, 1)}


# whole-word survival (reuse Workstream C's exact definition)
def survival(kbpe: KreyolBPE, words) -> dict:
    from fertility.counting import survival as _surv
    n_single, n_total, rows = _surv(kbpe.count, words)
    return {"n_single": n_single, "n_total": n_total,
            "frac": n_single / n_total if n_total else 0.0, "rows": rows}


ROBUSTNESS_PROBES = {
    "caps": ["Ayiti", "AYITI", "ayiti", "PÒTOPRENS", "Pòtoprens"],
    "numbers": ["1804", "2026", "3.14", "100000", "12h30"],
    "spelling_variants": ["ou", "w", "mwen", "m", "li", "l", "kreyòl", "kreyol"],
    "accents_apostrophe": ["fè", "fe", "m'ap", "n'ta", "lòt", "sè"],
}


def robustness(kbpe: KreyolBPE) -> dict:
    out = {}
    for group, items in ROBUSTNESS_PROBES.items():
        out[group] = {w: [kbpe.decode([i]) for i in kbpe.encode_ordinary(w)] for w in items}
    return out
