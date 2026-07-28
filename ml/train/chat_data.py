"""Workstream I — SFT/midtraining data engineering (Layers 1 & 3).

Download → clean → assemble the three-layer stack's real-data layers into a unified
conversation schema (Layer 2 is generated separately in chat_layer2). Everything lands
under ml/data/interim/chat/ (git-ignored); nothing here is committed.

Unified conversation record (one JSON object per line):
  {"messages": [{"role": "user"|"assistant", "content": str}, ...],
   "source": str, "layer": 1|3, "meta": {...}}

Standing rules enforced here:
  • the 15 held-out probe proverbs appear in NO example (corpus.build_v0_2._has_probe);
  • FLORES (our standing MT eval) never leaks — every FLORES-derived template row is
    dropped even though the licenses permit training;
  • English <think> reasoning + system fields are stripped from kakugo (English
    reasoning traces must not enter training);
  • per-turn langid drops non-Kreyòl conversations.

Run:  uv run python -m train.chat_data all          # download + assemble everything
      uv run python -m train.chat_data download      # just the per-source cleaned jsonl
      uv run python -m train.chat_data assemble       # just re-assemble the layers
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata

from . import chat_config as C
from . import config as F

sys.path.insert(0, F.REPO_ROOT)   # so `corpus` (sibling package) imports


# ============================ shared helpers ==================================

_LID = None
_PROBES = None


def _lid(text: str):
    """(lang, conf) top-1 via the corpus fastText lid.176 (same model the audit uses)."""
    global _LID
    if _LID is None:
        import fasttext
        from corpus import common
        _LID = fasttext.load_model(common.ensure_lid_model())
    t = text.replace("\n", " ").strip()
    if not t:
        return "und", 0.0
    prob, label = _LID.f.predict(t, 1, 0.0, "strict")[0]
    return label.replace("__label__", ""), float(prob)


def _pnorm(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s).lower()).strip()


def _probes():
    global _PROBES
    if _PROBES is None:
        p = os.path.join(F.DATA, "eval", "proverbs_probe.jsonl")
        _PROBES = []
        if os.path.exists(p):
            for line in open(p, encoding="utf-8"):
                d = json.loads(line)
                t = (d.get("kreyol") or d.get("text") or d.get("proverb") or "").strip()
                if len(t) > 15:
                    _PROBES.append(_pnorm(t))
    return _PROBES


def _has_probe(text: str) -> bool:
    pn = _pnorm(text)
    return any(pr in pn for pr in _probes())


def _norm(text: str) -> str:
    return re.sub(r"[ \t]+", " ", unicodedata.normalize("NFC", text)).strip()


def _foreign_turn(text: str) -> bool:
    """Is this turn CLEARLY not Kreyòl? langid is unreliable on short text and reads Kreyòl
    as French at modest confidence, so: only judge turns ≥ 40 chars, hold French to a high
    bar (0.85) and English to 0.55, other foreign langs to LANGID_DROP_CONF."""
    if len(text) < 40:
        return False
    lang, conf = _lid(text)
    if lang == "ht":
        return False
    if lang == "fr":
        return conf >= 0.85
    if lang == "en":
        return conf >= 0.55
    return lang in C.FOREIGN_LANGS and conf >= C.LANGID_DROP_CONF


def _degenerate(content: str) -> bool:
    """A turn that is empty, too short, or a trivial single-char/word repeat."""
    c = content.strip()
    if len(c) < C.MIN_CONTENT_CHARS:
        return True
    toks = c.split()
    if len(toks) >= 6 and len(set(toks)) <= 2:      # "yon yon yon yon …"
        return True
    return False


def _write_jsonl(path: str, records: list) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(records)


def _dedup_key(messages: list) -> str:
    joined = " ".join(m["content"] for m in messages)
    return _pnorm(joined)[:C.NEAR_DUP_KEY_CHARS]


def _hf_token():
    d = F.REPO_ROOT
    for _ in range(5):
        cand = os.path.join(d, ".env")
        if os.path.exists(cand):
            for line in open(cand, encoding="utf-8"):
                if line.strip().startswith("HF_TOKEN="):
                    v = line.strip().split("=", 1)[1].strip().strip('"').strip("'")
                    if v:
                        return v
        d = os.path.dirname(d)
    return None


def _log(msg):
    print(f"[chat_data] {msg}", flush=True)


# ============================ kakugo-hat (Layer 1) ============================

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_think(content: str) -> str:
    content = _THINK_RE.sub("", content)
    # drop any stray unmatched think tags + collapse whitespace
    content = re.sub(r"</?think>", "", content, flags=re.IGNORECASE)
    return content.strip()


def clean_kakugo() -> dict:
    """kakugo-hat: strip English <think>/system, drop non-Kreyòl turns, degenerate/near-dup,
    probe leaks. Multi-turn conversations → unified schema."""
    from datasets import load_dataset
    os.environ.setdefault("HF_TOKEN", _hf_token() or "")
    ds = load_dataset(C.DS_KAKUGO["repo"], split="train", token=_hf_token())
    stats = {"in": 0, "empty_after_think": 0, "foreign_turn": 0, "degenerate": 0,
             "probe_leak": 0, "near_dup": 0, "kept": 0}
    seen, out = set(), []
    for row in ds:
        stats["in"] += 1
        msgs_in = row.get("messages") or []
        msgs, bad = [], False
        for m in msgs_in:
            role = m.get("role") or ("assistant" if m.get("from") == "gpt" else "user")
            content = m.get("content") or m.get("value") or ""
            if role == "assistant":
                content = _strip_think(content)
            content = _norm(content)
            if not content:
                bad = True
                break
            msgs.append({"role": "user" if role not in ("assistant",) else "assistant",
                         "content": content})
        if bad or not msgs:
            stats["empty_after_think"] += 1
            continue
        if any(_degenerate(m["content"]) for m in msgs):
            stats["degenerate"] += 1
            continue
        if any(_foreign_turn(m["content"]) for m in msgs):
            stats["foreign_turn"] += 1
            continue
        if any(_has_probe(m["content"]) for m in msgs):
            stats["probe_leak"] += 1
            continue
        k = _dedup_key(msgs)
        if k in seen:
            stats["near_dup"] += 1
            continue
        seen.add(k)
        out.append({"messages": msgs, "source": "kakugo-hat", "layer": 1,
                    "meta": {"topic": row.get("topic"),
                             "generation_method": row.get("generation_method"),
                             "n_turns": len(msgs)}})
    stats["kept"] = _write_jsonl(C.KAKUGO_CLEAN, out)
    _log(f"kakugo: {json.dumps(stats)}")
    return stats


# ================= aya_collection + xP3x (Layer 1, streamed + capped) =========

def _clean_single_turn_stream(spec, out_path, source, cap, drop_flores):
    """Stream a single-turn inputs→targets dataset (aya_collection / xP3x), applying the
    FLORES carve-out, langid on the Kreyòl target, probe screen, dedup, and a hard cap."""
    from datasets import load_dataset
    os.environ.setdefault("HF_TOKEN", _hf_token() or "")
    kw = {"split": "train", "streaming": True, "token": _hf_token()}
    if spec.get("config"):
        kw["name"] = spec["config"]
    ds = load_dataset(spec["repo"], **kw)
    stats = {"in": 0, "flores": 0, "eval_split": 0, "unk_corrupt": 0, "empty": 0,
             "foreign_target": 0, "degenerate": 0, "probe_leak": 0, "near_dup": 0, "kept": 0}
    seen, out = set(), []
    for row in ds:
        stats["in"] += 1
        # eval carve-out: FLORES-derived or eval-split rows never enter training
        prov = " ".join(str(row.get(k, "")) for k in
                        ("dataset", "dataset_name", "sub_dataset_name", "template", "config")).lower()
        if drop_flores and any(m in prov for m in C.FLORES_MARKERS):
            stats["flores"] += 1
            continue
        if str(row.get("split", "train")).lower() in C.EVAL_SPLITS:
            stats["eval_split"] += 1
            continue
        u = _norm(str(row.get("inputs", "")))
        a = _norm(str(row.get("targets", "")))
        if not u or not a:
            stats["empty"] += 1
            continue
        if a.count("<unk>") >= 1:                     # corrupted target
            stats["unk_corrupt"] += 1
            continue
        if _degenerate(a):
            stats["degenerate"] += 1
            continue
        if _foreign_turn(a):                          # the Kreyòl OUTPUT must be Kreyòl
            stats["foreign_target"] += 1
            continue
        if _has_probe(u) or _has_probe(a):
            stats["probe_leak"] += 1
            continue
        msgs = [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
        k = _dedup_key(msgs)
        if k in seen:
            stats["near_dup"] += 1
            continue
        seen.add(k)
        out.append({"messages": msgs, "source": source, "layer": 1,
                    "meta": {"task_type": row.get("task_type"),
                             "dataset_name": row.get("dataset_name") or row.get("dataset")}})
        if len(out) >= cap:
            break
    stats["kept"] = _write_jsonl(out_path, out)
    _log(f"{source}: {json.dumps(stats)}")
    return stats


def clean_aya_collection() -> dict:
    return _clean_single_turn_stream(C.DS_AYA_COLLECTION, C.AYA_COLLECTION_CLEAN,
                                     "aya_collection", C.AYA_COLLECTION_CAP, drop_flores=True)


def clean_xp3x() -> dict:
    return _clean_single_turn_stream(C.DS_XP3X, C.XP3X_CLEAN, "xp3x", C.XP3X_CAP, drop_flores=True)


# ==================== muri-it hat + aya gold (Layer 3, pyarrow) ===============

def _hf_parquet_filter(repo, split_name, field, value, columns):
    """Read only the rows where field==value from a HF parquet dataset via HfFileSystem +
    pyarrow predicate pushdown (avoids downloading the whole 3.8GB muri-it to keep 9,876)."""
    import pyarrow.dataset as pads
    import pyarrow.compute as pc
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem(token=_hf_token())
    files = [p for p in fs.glob(f"datasets/{repo}/**/*.parquet")
             if split_name in os.path.basename(p)]
    if not files:
        files = [p for p in fs.glob(f"datasets/{repo}/**/*.parquet") if "train" in os.path.basename(p)]
    dset = pads.dataset(files, filesystem=fs, format="parquet")
    tbl = dset.to_table(filter=pc.field(field) == value, columns=columns)
    return tbl.to_pylist()


def clean_muri_it() -> dict:
    rows = _hf_parquet_filter(C.DS_MURI_IT["repo"], "train",
                              C.DS_MURI_IT["lang_field"], C.DS_MURI_IT["lang"],
                              ["input", "output", "dataset_name", "subdataset_name", "language"])
    stats = {"in": len(rows), "empty": 0, "foreign_target": 0, "degenerate": 0,
             "probe_leak": 0, "near_dup": 0, "kept": 0}
    seen, out = set(), []
    for r in rows:
        u = _norm(str(r.get("input", "")))
        a = _norm(str(r.get("output", "")))
        if not u or not a:
            stats["empty"] += 1
            continue
        if _degenerate(a):
            stats["degenerate"] += 1
            continue
        if _foreign_turn(a):
            stats["foreign_target"] += 1
            continue
        if _has_probe(u) or _has_probe(a):
            stats["probe_leak"] += 1
            continue
        msgs = [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
        k = _dedup_key(msgs)
        if k in seen:
            stats["near_dup"] += 1
            continue
        seen.add(k)
        out.append({"messages": msgs, "source": "muri-it", "layer": 3,
                    "meta": {"subdataset": r.get("subdataset_name")}})
    stats["kept"] = _write_jsonl(C.MURI_IT_HAT, out)
    _log(f"muri-it: {json.dumps(stats)}")
    return stats


def clean_aya_gold() -> dict:
    rows = _hf_parquet_filter(C.DS_AYA_DATASET["repo"], "train",
                              C.DS_AYA_DATASET["lang_field"], C.DS_AYA_DATASET["lang"],
                              ["inputs", "targets", "language_code"])
    stats = {"in": len(rows), "empty": 0, "probe_leak": 0, "near_dup": 0, "kept": 0}
    seen, out = set(), []
    for r in rows:
        u = _norm(str(r.get("inputs", "")))
        a = _norm(str(r.get("targets", "")))
        if not u or not a:
            stats["empty"] += 1
            continue
        if _has_probe(u) or _has_probe(a):
            stats["probe_leak"] += 1
            continue
        msgs = [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
        k = _dedup_key(msgs)
        if k in seen:
            stats["near_dup"] += 1
            continue
        seen.add(k)
        out.append({"messages": msgs, "source": "aya-gold", "layer": 3, "meta": {"gold": True}})
    stats["kept"] = _write_jsonl(C.AYA_GOLD, out)
    _log(f"aya-gold: {json.dumps(stats)}")
    return stats


# ==================== translation turns (glossary + CMU lexicon) ==============

# Kreyòl instruction phrasings (rights-clear PD pairs → translation-QA turns). Varied so
# the model learns the TASK, not one template. {en}/{ht} filled per pair.
_EN2HT = [
    "Tradui fraz sa a an kreyòl: «{en}»",
    "Ki jan ou di «{en}» an kreyòl?",
    "An kreyòl, «{en}» vle di kisa?",
]
_HT2EN = [
    "Tradui fraz sa a an anglè: «{ht}»",
    "Ki sa «{ht}» vle di an anglè?",
]


def build_translation_turns() -> dict:
    """Translation-QA turns from the committable federal PD glossary (1,955 EN↔HT pairs) +
    a capped sample of the CMU lexicon (32,231 pairs). Rights-clear; format teaching."""
    import random
    rng = random.Random(C.DL_SEED)
    pairs = []
    gp = os.path.join(F.REPO_ROOT, "corpus", "glossary_pairs_federal.json")
    if os.path.exists(gp):
        for p in json.load(open(gp, encoding="utf-8")).get("pairs", []):
            en, ht = _norm(p.get("en", "")), _norm(p.get("ht", ""))
            if en and ht:
                pairs.append(("glossary_federal", en, ht))
    lex = os.path.join(F.DATA, "interim", "v0_2_1_ingest", "cmu_lexicon.json")
    if os.path.exists(lex):
        lp = json.load(open(lex, encoding="utf-8")).get("pairs", [])
        # prefer longer (more sentence-like) lexicon entries; cap the sample
        lp = sorted(lp, key=lambda d: -len(str(d.get("ht", ""))))[:C.CMU_LEXICON_SAMPLE]
        for p in lp:
            en, ht = _norm(str(p.get("en", ""))), _norm(str(p.get("ht", "")))
            if en and ht and len(ht) >= C.MIN_CONTENT_CHARS:
                pairs.append(("cmu_lexicon", en, ht))
    stats = {"pairs": len(pairs), "probe_leak": 0, "kept": 0}
    out = []
    for src, en, ht in pairs:
        if _has_probe(ht):
            stats["probe_leak"] += 1
            continue
        if rng.random() < 0.5:                # ~half each direction
            u = rng.choice(_EN2HT).format(en=en); a = ht
        else:
            u = rng.choice(_HT2EN).format(ht=ht); a = en
        out.append({"messages": [{"role": "user", "content": u},
                                 {"role": "assistant", "content": a}],
                    "source": f"translation:{src}", "layer": 1,
                    "meta": {"en": en, "ht": ht, "glossary": src}})
    rng.shuffle(out)
    out = out[:C.TRANSLATION_TURNS_CAP]
    stats["kept"] = _write_jsonl(C.TRANSLATION_TURNS, out)
    _log(f"translation-turns: {json.dumps(stats)}")
    return stats


# ============================ assemble the layers =============================

def _read_jsonl(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def assemble_layer1() -> dict:
    """Layer 1 (midtraining): kakugo (majority) + capped aya_collection + xP3x + translation
    turns. kakugo must dominate (native multi-turn); templated bulk stays minority."""
    import random
    parts = {"kakugo-hat": _read_jsonl(C.KAKUGO_CLEAN),
             "aya_collection": _read_jsonl(C.AYA_COLLECTION_CLEAN),
             "xp3x": _read_jsonl(C.XP3X_CLEAN),
             "translation": _read_jsonl(C.TRANSLATION_TURNS)}
    allrows = [r for v in parts.values() for r in v]
    random.Random(C.DL_SEED).shuffle(allrows)
    n = _write_jsonl(C.LAYER1_JSONL, allrows)
    comp = {k: len(v) for k, v in parts.items()}
    _log(f"layer1: total={n} composition={comp}")
    return {"total": n, "composition": comp}


def assemble_layer3(include_layer2=True) -> dict:
    """Layer 3 (SFT cap): muri-it hat + aya gold + a curated glossary-QA subset + (if present)
    the best Layer-2 items. Small + excellent; loss masked to responses at train time."""
    import random
    rng = random.Random(C.DL_SEED + 1)
    muri = _read_jsonl(C.MURI_IT_HAT)
    gold = _read_jsonl(C.AYA_GOLD)
    # a small clean glossary-only QA subset (federal PD, high precision) for Layer 3
    gloss = [r for r in _read_jsonl(C.TRANSLATION_TURNS)
             if r.get("meta", {}).get("glossary") == "glossary_federal"]
    rng.shuffle(gloss); gloss = gloss[:500]
    for r in gloss:
        r["layer"] = 3
    layer2 = (_read_jsonl(C.LAYER2_PILOT) + _read_jsonl(C.LAYER2_GEN)) if include_layer2 else []
    # keep only the higher-quality Layer-2 items (teacher self-score ≥ 4) for the SFT cap
    layer2 = [r for r in layer2 if (r.get("meta", {}).get("self_score") or 5) >= 4]
    for r in layer2:
        r["layer"] = 3
    allrows = muri + gold + gloss + layer2
    rng.shuffle(allrows)
    n = _write_jsonl(C.LAYER3_JSONL, allrows)
    comp = {"muri-it": len(muri), "aya-gold": len(gold), "glossary-qa": len(gloss),
            "layer2": len(layer2)}
    _log(f"layer3: total={n} composition={comp}")
    return {"total": n, "composition": comp}


# ================================ driver ======================================

def download_all() -> dict:
    """Run every source cleaner. Each is independent (writes its own jsonl); a failure in one
    is recorded and does not abort the rest (this runs as a long background job)."""
    os.makedirs(C.CHAT_RAW, exist_ok=True)
    os.makedirs(C.CHAT_WORK, exist_ok=True)
    steps = [("kakugo", clean_kakugo), ("aya_collection", clean_aya_collection),
             ("xp3x", clean_xp3x), ("muri_it", clean_muri_it),
             ("aya_gold", clean_aya_gold), ("translation", build_translation_turns)]
    s = {}
    for name, fn in steps:
        try:
            s[name] = fn()
        except Exception as e:
            import traceback
            s[name] = {"error": f"{type(e).__name__}: {e}"}
            _log(f"!! {name} FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
        with open(os.path.join(C.CHAT_WORK, "download_stats.json"), "w") as fh:
            json.dump(s, fh, indent=2, ensure_ascii=False)   # incremental save
    return s


def assemble_all(include_layer2=True) -> dict:
    l1 = assemble_layer1()
    l3 = assemble_layer3(include_layer2=include_layer2)
    return {"layer1": l1, "layer3": l3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["download", "assemble", "all"], default="all", nargs="?")
    ap.add_argument("--no-layer2", action="store_true", help="assemble Layer 3 without Layer-2 items")
    args = ap.parse_args()
    os.makedirs(C.CHAT_RAW, exist_ok=True)
    os.makedirs(C.CHAT_WORK, exist_ok=True)
    if args.step in ("download", "all"):
        download_all()
    if args.step in ("assemble", "all"):
        assemble_all(include_layer2=not args.no_layer2)


if __name__ == "__main__":
    main()
