"""Phase-1 Workstream E, stage 2 — standing eval slices.

Builds two held-out quality axes from corpus v0.1, registered in splits.yaml and
NEVER trained on (carved at consumption by doc_id, like the tokenizer_eval holdout):

  * authored_eval          — native-voice Kreyòl: human-verified audit-natural docs
                             (tier A) + a capped, seeded sample of non-bot-stub
                             Wikipedia (tier B).
  * translation_shaped_eval — machine/translationese Kreyòl: human-verified audit
                             "translated" docs (tier A) + a capped, seeded sample of
                             crawl docs carrying a residual MT/CMS fingerprint too weak
                             to be dropped as junk (tier B).

Every member is a surviving v0.1 document (junk already removed). Tier A is the
gold, human-concurred core (ml/reports/audit_model_summary.md §Human verification);
tier B is a precise, documented heuristic expansion, labelled separately in the
manifest so a reader always knows which docs are verified.

Manifests (git-ignored, under data/eval/) carry {doc_id, source, slice, tier, text}
so BPB evaluation can read them and any training run can exclude their doc_ids.

Run:  python -m corpus.evalslices [--sample]   (requires corpus.junk to have run)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os

from . import common
from . import config_v0_1 as CV
from .junk import _ENTITY_RE, _XNUM_RE


def _load_audit_labels() -> dict:
    """doc_id -> (language, quality, translation_shaped) from the human-concurred audit."""
    out = {}
    with open(CV.AUDIT_MODEL_LABELS, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r["doc_id"]] = (r["model_language"], r["model_quality"],
                                r["model_translation_shaped"])
    return out


def _seeded_rank(doc_id: str) -> int:
    return int(hashlib.sha256(f"{CV.EVAL_SEED}:{doc_id}".encode()).hexdigest()[:12], 16)


def _residual_mt_fingerprint(text: str) -> bool:
    """Light translationese signal (below the junk-drop threshold): any HTML entity,
    or exactly one XNUMX/XNMX placeholder."""
    return len(_ENTITY_RE.findall(text)) >= 1 or len(_XNUM_RE.findall(text)) == 1


def build_slices(sample: bool) -> dict:
    tag = common.run_tag(sample)
    corpus = CV.CORPUS_V0_1.format(tag=tag)
    labels = _load_audit_labels()

    audit_natural = {d for d, (lg, q, ts) in labels.items()
                     if lg == "ht" and q == "ok" and ts == "natural"}
    audit_translated = {d for d, (lg, q, ts) in labels.items() if ts == "translated"}

    # --- pass 1: classify surviving v0.1 docs into slice pools -----------------
    authored_A, trans_A = set(), set()
    authored_B_pool, trans_B_pool = [], []      # (rank, doc_id)
    for d in common.read_jsonl(corpus):
        did = d["acquisition"]["doc_id"]
        cls = common.priority_class(d["acquisition"]["source"])
        if did in audit_natural:
            authored_A.add(did)
        elif did in audit_translated:
            trans_A.add(did)
        # tier-B pools (exclude anything already human-labelled to keep tiers disjoint)
        if did in labels:
            continue
        if cls == "wikipedia" and not d.get("wiki_bot_stub", False):
            authored_B_pool.append((_seeded_rank(did), did))
        elif cls == "crawl" and _residual_mt_fingerprint(d["text"]):
            trans_B_pool.append((_seeded_rank(did), did))

    authored_B = {did for _, did in sorted(authored_B_pool)[:CV.AUTHORED_WIKI_CAP]}
    trans_B = {did for _, did in sorted(trans_B_pool)[:CV.TRANS_HEURISTIC_CAP]}

    members = {
        "authored_eval": {"A": authored_A, "B": authored_B},
        "translation_shaped_eval": {"A": trans_A, "B": trans_B},
    }
    want = {did: (slice_name, tier)
            for slice_name, tiers in members.items()
            for tier, ids in tiers.items() for did in ids}

    # --- pass 2: pull text for selected docs, write manifests ------------------
    tok = common.RefTokenizer()
    sfx = CV.eval_suffix(sample)
    paths = {"authored_eval": CV.AUTHORED_EVAL.format(suffix=sfx),
             "translation_shaped_eval": CV.TRANSLATION_SHAPED_EVAL.format(suffix=sfx)}
    os.makedirs(CV.EVAL_DIR, exist_ok=True)
    fhs = {name: open(paths[name], "w", encoding="utf-8") for name in paths}
    agg = {name: {"A": _u0(), "B": _u0()} for name in paths}
    for d in common.read_jsonl(corpus):
        did = d["acquisition"]["doc_id"]
        if did not in want:
            continue
        slice_name, tier = want[did]
        text = d["text"]
        rec = {"doc_id": did, "source": d["acquisition"]["source"],
               "slice": slice_name, "tier": tier, "split": slice_name,
               "wiki_bot_stub": d.get("wiki_bot_stub", False), "text": text}
        fhs[slice_name].write(json.dumps(rec, ensure_ascii=False) + "\n")
        u = agg[slice_name][tier]
        u["docs"] += 1
        u["bytes"] += common.n_bytes(text); u["chars"] += common.n_chars(text)
        u["words"] += common.n_ws_words(text); u["tokens"] += tok.count(text)
    for fh in fhs.values():
        fh.close()

    stats = {
        "snapshot_date": CV.SNAPSHOT_DATE, "seed": CV.EVAL_SEED,
        "reference_tokenizer": tok.name,
        "source_corpus": os.path.basename(corpus),
        "caps": {"authored_wiki_tier_b": CV.AUTHORED_WIKI_CAP,
                 "translation_heuristic_tier_b": CV.TRANS_HEURISTIC_CAP},
        "slices": {name: {"tier_a_verified": agg[name]["A"],
                          "tier_b_heuristic": agg[name]["B"],
                          "total": _sum(agg[name]["A"], agg[name]["B"])}
                   for name in paths},
        "tier_b_pool_sizes": {"authored_wiki_nonstub": len(authored_B_pool),
                              "crawl_residual_fingerprint": len(trans_B_pool)},
    }
    stats_path = CV.EVAL_SLICE_STATS.format(tag=tag)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)

    for name in paths:
        t = stats["slices"][name]["total"]
        a = stats["slices"][name]["tier_a_verified"]["docs"]
        common.log(f"  {name}: {t['docs']} docs ({a} verified + {t['docs'] - a} heuristic), "
                   f"{t['bytes']:,} bytes, {t['tokens']:,} o200k tokens -> {os.path.basename(paths[name])}")
    return stats


def _u0() -> dict:
    return {"docs": 0, "bytes": 0, "chars": 0, "words": 0, "tokens": 0}


def _sum(a: dict, b: dict) -> dict:
    return {k: a[k] + b[k] for k in a}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    args = ap.parse_args()
    common.log(f"[evalslices] sample={args.sample}")
    build_slices(args.sample)


if __name__ == "__main__":
    main()
