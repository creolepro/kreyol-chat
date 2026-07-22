"""Data prep for the base-model probe — all LOCAL (no GPU), all pinned/seeded.

Selection corpus is FLORES+ **dev** (devtest reserved). BPB text comes from the
corpus tokenizer_eval holdout (never-trained docs), reusing the Workstream B
holdout definition so the "never trained" guarantee is identical. The 15 probe
proverbs (data/eval/proverbs_probe.jsonl) are the proverb-completion eval and
appear in NO training set by construction; we re-assert they are absent from the
BPB slices.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import unicodedata

from . import config

# Reuse the Workstream B holdout/never-trained definitions verbatim.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizer import data as TD  # noqa: E402
from tokenizer import config as TCONF  # noqa: E402


def log(msg: str):
    print(msg, file=sys.stderr, flush=True)


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def load_env():
    """Read HF_TOKEN from the nearest .env (repo root), without extra deps."""
    here = os.path.dirname(os.path.abspath(__file__))
    d = here
    for _ in range(5):
        cand = os.path.join(d, ".env")
        if os.path.exists(cand):
            for line in open(cand, encoding="utf-8"):
                line = line.strip()
                if line.startswith("HF_TOKEN="):
                    v = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if v:
                        os.environ.setdefault("HF_TOKEN", v)
        d = os.path.dirname(d)
    return os.environ.get("HF_TOKEN")


# --- FLORES+ dev (ht/en/fr joined by id) --------------------------------------

def load_flores_dev(hf_token: str) -> dict:
    """Pinned dev JSONL for ht/en/fr, NFC-normalized, joined by id (== Workstream
    C join, but the dev split). Returns {'ids', 'text': {code: {id: str}}}."""
    from huggingface_hub import hf_hub_download

    by_lang = {}
    for code, stem in config.FLORES_LANGS.items():
        path = hf_hub_download(
            config.FLORES_REPO,
            f"{config.FLORES_SPLIT}/{stem}.jsonl",
            repo_type="dataset",
            revision=config.FLORES_REVISION,
            token=hf_token,
        )
        rows = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                rows[str(r["id"])] = nfc(r["text"])
        by_lang[code] = rows

    common = set.intersection(*(set(v) for v in by_lang.values()))
    ids = sorted(common, key=int)
    log(f"  FLORES+ {config.FLORES_REPO}@{config.FLORES_REVISION[:12]} "
        f"{config.FLORES_SPLIT}: joined {len(ids)} sentences by (split, id)")
    return {"ids": ids, "text": by_lang}


def select_ids(all_ids):
    """One seeded shuffle -> disjoint exemplar ids and eval-subset ids.

    exemplars = first N_SHOT, eval subset = next EVAL_SUBSET_N. Deterministic and
    provably disjoint. Also returns the SMOKE_N prefix of the eval subset.
    """
    ids = list(all_ids)
    random.Random(config.SELECT_SEED).shuffle(ids)
    exemplars = ids[: config.N_SHOT]
    subset = ids[config.N_SHOT: config.N_SHOT + config.EVAL_SUBSET_N]
    smoke = subset[: config.SMOKE_N]
    assert not (set(exemplars) & set(subset)), "exemplar/eval overlap"
    return {"exemplars": exemplars, "subset": subset, "smoke": smoke,
            "full_dev": [i for i in ids if i not in set(exemplars)]}


def mt_prompts(flores, ex_ids, eval_ids):
    """Both-direction 5-shot MT prompts + references for a set of eval ids."""
    t = flores["text"]
    shots_e2h = [(t["en"][i], t["ht"][i]) for i in ex_ids]
    shots_h2e = [(t["ht"][i], t["en"][i]) for i in ex_ids]
    e2h = [config.build_mt_prompt(shots_e2h, "en", "ht", t["en"][i]) for i in eval_ids]
    h2e = [config.build_mt_prompt(shots_h2e, "ht", "en", t["ht"][i]) for i in eval_ids]
    refs_e2h = [t["ht"][i] for i in eval_ids]
    refs_h2e = [t["en"][i] for i in eval_ids]
    return {"eng2hat": {"prompts": e2h, "refs": refs_e2h},
            "hat2eng": {"prompts": h2e, "refs": refs_h2e}}


# --- BPB holdout slices (full + authored-only) --------------------------------

def _subsample_bytes(texts, budget, seed):
    order = list(range(len(texts)))
    random.Random(seed).shuffle(order)
    picked, total = [], 0
    for i in order:
        picked.append(texts[i])
        total += len(texts[i].encode("utf-8"))
        if total >= budget:
            break
    return picked, total


def holdout_slices():
    """(a) full tokenizer_eval holdout (seed-subsampled to a byte budget) and
    (b) authored-only subset = Wikipedia non-stub + owned docs (scored in full).

    Both are NFC text of never-trained docs. Probe proverbs are asserted absent.
    """
    full, authored = [], []
    for d in TD.iter_docs():
        did = d["acquisition"]["doc_id"]
        if not TD.in_holdout(did):
            continue
        src = d["acquisition"]["source"]
        text = TD.nfc(d["text"])
        full.append(text)
        if src == "ht_wikipedia" and not d.get("wiki_bot_stub", False):
            authored.append(text)
        elif src == "owned_proverbs":
            authored.append(text)

    TD.assert_no_probe(authored)  # never-trained AND never a probe proverb
    n_full_all = len(full)
    full_sub, full_bytes = _subsample_bytes(
        full, config.BPB_FULL_BYTE_BUDGET, config.BPB_SUBSAMPLE_SEED)
    auth_bytes = sum(len(t.encode("utf-8")) for t in authored)
    log(f"  BPB full slice: {len(full_sub)}/{n_full_all} docs "
        f"({full_bytes:,} B, budget {config.BPB_FULL_BYTE_BUDGET:,}); "
        f"authored-only: {len(authored)} docs ({auth_bytes:,} B, full)")
    return {
        "full": {"texts": full_sub, "bytes": full_bytes,
                 "n_docs": len(full_sub), "n_docs_available": n_full_all},
        "authored": {"texts": authored, "bytes": auth_bytes,
                     "n_docs": len(authored)},
    }


# --- proverb completion (15-item probe split) ---------------------------------

# Two extremely well-known proverbs used ONLY as few-shot format exemplars. Both
# are verified NOT among the 15 probe proverbs (asserted below at build time).
PROVERB_SHOTS = [
    "Piti piti zwazo fè nich li.",
    "Men anpil, chay pa lou.",
]


def _split_proverb(text):
    words = text.split()
    k = max(1, math.ceil(len(words) / 2))
    return " ".join(words[:k]), " ".join(words[k:])


def load_probe_proverbs():
    path = TCONF.PROVERBS_PROBE
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    probe_texts = {nfc(r["text"]).strip() for r in rows}
    for s in PROVERB_SHOTS:
        assert nfc(s).strip() not in probe_texts, \
            f"proverb-shot exemplar {s!r} collides with a probe proverb"
    items = []
    for r in rows:
        full = nfc(r["text"]).strip()
        prompt_half, gold = _split_proverb(full)
        items.append({"num": r.get("proverb_num"), "full": full,
                      "prompt_half": prompt_half, "gold": gold,
                      "english": r.get("proverb_english", "")})
    return items


def proverb_prompts(items):
    header = "Men kèk pwovèb kreyòl:\n"
    shots = "".join(f"- {s}\n" for s in PROVERB_SHOTS)
    return [f"{header}{shots}- {it['prompt_half']}" for it in items]


# --- naturalness completion prompts -------------------------------------------

def naturalness_prompts():
    return [p["prompt"] for p in config.NATURALNESS_PROMPTS]
