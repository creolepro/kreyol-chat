"""Workstream H, Part 1b — build the fleet's training bins + fixed eval set, upload.

Produces one flat uint16 `.bin` per data variant (fleet_config.VARIANTS) and a single
`eval_texts.json` with the 5 STANDING BPB slices (fixed for every run, so BPB is
comparable across corpora AND across tokenizers). Runs locally (tiktoken only).

Discipline:
  * Every training bin excludes the frozen eval-slice doc_ids (authored_eval,
    translation_shaped_eval, authored_eval_v2) AND the tokenizer_eval holdout (by the
    same seeded hash Workstream E/G use) — so general_holdout stays held out of ALL
    corpora. Probe proverbs are absent from the corpus entirely (verified, Part 0b).
  * Doc ORDER within a bin is fixed per variant (DOC_SHUFFLE_SEED); the run seed only
    varies init + offset sampling. Q1's two tokenizers therefore share doc order — the
    only difference is the learned vocabulary.
  * Nothing here is committed (everything lands under data/, git-ignored).

Tokenization is shared across variants with the same (corpus, tokenizer): v0.2.1/kbpe
serves natural + weighted + the Q5 subsample from ONE pass.

Run:  cd ml && uv run python -m train.fleet_data            # build all bins + eval + manifest
      cd ml && uv run python -m train.fleet_data --upload   # push to the Modal Volume
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle

import numpy as np

from . import config as F
from . import fleet_config as H
from . import llama_config as G


# --- tokenizer bundles --------------------------------------------------------

def encoding(tok_key: str):
    import tiktoken
    d = pickle.load(open(H.TOKENIZERS[tok_key], "rb"))
    ranks, pattern = d["mergeable_ranks"], d["pattern"]
    offset = len(ranks)
    special = {name: offset + i for i, name in enumerate(F.SPECIAL_TOKENS)}
    enc = tiktoken.Encoding(name=f"fleet-{tok_key}", pat_str=pattern,
                            mergeable_ranks=ranks, special_tokens=special)
    assert enc.n_vocab == F.VOCAB_SIZE, f"{tok_key}: {enc.n_vocab} != {F.VOCAB_SIZE}"
    return enc


# --- exclusions (eval slices + tokenizer holdout) -----------------------------

def _u01(key: str) -> float:
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def _in_tokenizer_holdout(doc_id: str) -> bool:
    return _u01(f"holdout:{F.HOLDOUT_SPLIT_SEED}:{doc_id}") < F.HOLDOUT_FRAC


def _slice_ids(path: str) -> set:
    ids = set()
    if not os.path.exists(path):
        return ids
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        did = d.get("doc_id") or d.get("acquisition", {}).get("doc_id")
        if did:
            ids.add(did)
    return ids


def excluded_ids() -> set:
    ids = set()
    for p in (F.AUTHORED_EVAL, F.TRANSLATION_SHAPED_EVAL, H.AUTHORED_EVAL_V2):
        ids |= _slice_ids(p)
    return ids


# --- per-(corpus, tokenizer) tokenization pass --------------------------------

def _tokenize_group(corpus_key: str, tok_key: str, exclude: set):
    """Stream a shard once; return per-doc records (register, is_stub, token array).
    Applies the global exclusions (eval slices + tokenizer holdout)."""
    enc = encoding(tok_key)
    encode = enc.encode_ordinary
    bos = enc.encode_single_token("<|bos|>")
    shard = H.CORPUS_SHARDS[corpus_key]
    docs, n_excl, n_hold = [], 0, 0
    for line in open(shard, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        did = d["acquisition"]["doc_id"]
        if did in exclude:
            n_excl += 1
            continue
        if _in_tokenizer_holdout(did):
            n_hold += 1
            continue
        toks = encode(d["text"])
        arr = np.fromiter([bos, *toks], dtype=np.uint16, count=len(toks) + 1)
        docs.append((d.get("register") or "web_crawl", bool(d.get("wiki_bot_stub")), arr))
    return docs, bos, {"excluded_eval": n_excl, "tokenizer_holdout": n_hold}


def _weighted_expand(docs, seed):
    """Repeat each doc by its register mix weight (integer part + seeded fractional
    Bernoulli). web_crawl stays 1x; authored/register registers are oversampled."""
    rng = np.random.default_rng(seed)
    out = []
    for reg, is_stub, arr in docs:
        w = H.MIX_WEIGHTS.get(reg, 1.0)
        k = int(w)
        if rng.random() < (w - k):
            k += 1
        out.extend([(reg, is_stub, arr)] * k)
    return out


def _concat_shuffled(records, seed):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(records))
    if not len(records):
        return np.zeros(0, np.uint16)
    return np.concatenate([records[i][2] for i in order])


def _register_tokens(records):
    from collections import Counter
    c = Counter()
    for reg, _, arr in records:
        c[reg] += len(arr)
    return dict(c.most_common())


def build_variant_bins() -> dict:
    """Build every variant bin, grouping by (corpus, tokenizer) so each shard is
    tokenized once. Writes {variant}.bin + returns a manifest."""
    os.makedirs(H.FLEET_BUNDLE, exist_ok=True)
    exclude = excluded_ids()
    print(f"[fleet_data] excluding {len(exclude)} eval-slice doc_ids + tokenizer holdout")

    # group variants by (corpus, tok)
    groups: dict[tuple, list] = {}
    for vname, v in H.VARIANTS.items():
        groups.setdefault((v["corpus"], v["tok"]), []).append(vname)

    manifest = {"snapshot": H.SNAPSHOT_DATE, "doc_shuffle_seed": H.DOC_SHUFFLE_SEED,
                "variants": {}}
    q5_unique_tokens = None

    for (corpus_key, tok_key), vnames in groups.items():
        print(f"[fleet_data] tokenizing {corpus_key} with {tok_key} "
              f"(serves {vnames}) …")
        docs, bos, exinfo = _tokenize_group(corpus_key, tok_key, exclude)
        total_tok = sum(len(a) for _, _, a in docs)
        print(f"  {len(docs):,} docs, {total_tok:,} tokens "
              f"(excluded {exinfo['excluded_eval']} eval + {exinfo['tokenizer_holdout']} holdout)")

        for vname in vnames:
            v = H.VARIANTS[vname]
            if v.get("drop_bot_stubs"):
                recs = [r for r in docs if not r[1]]
            elif v["sampling"] == "weighted":
                recs = _weighted_expand(docs, H.DOC_SHUFFLE_SEED + 99)
            elif v.get("subsample_tokens"):
                # fixed unique subsample: seeded-shuffle, take docs until the budget
                rng = np.random.default_rng(H.DOC_SHUFFLE_SEED + 7)
                order = rng.permutation(len(docs))
                recs, acc = [], 0
                for i in order:
                    recs.append(docs[i])
                    acc += len(docs[i][2])
                    if acc >= v["subsample_tokens"]:
                        break
            else:
                recs = docs

            stream = _concat_shuffled(recs, H.DOC_SHUFFLE_SEED)
            out = os.path.join(H.FLEET_BUNDLE, f"{vname}.bin")
            stream.tofile(out)
            reg_tok = _register_tokens(recs)
            entry = {"corpus": corpus_key, "tokenizer": tok_key, "sampling": v["sampling"],
                     "docs": len(recs), "unique_docs": len(docs) if v["sampling"] != "weighted" else None,
                     "tokens": int(stream.size), "bos_id": int(bos),
                     "mb": round(stream.size * 2 / 1e6, 1),
                     "register_tokens_top": dict(list(reg_tok.items())[:8])}
            if v.get("drop_bot_stubs"):
                entry["dropped_bot_stubs"] = len(docs) - len(recs)
            manifest["variants"][vname] = entry
            print(f"    -> {vname}.bin: {stream.size:,} tokens ({entry['mb']} MB), {len(recs):,} docs")
            if vname == "q5sub_kbpe":
                q5_unique_tokens = int(stream.size)

    # Q5 epoch step counts, derived from the measured subsample size
    if q5_unique_tokens:
        manifest["q5_subsample_tokens"] = q5_unique_tokens
        manifest["q5_epoch_steps"] = {
            tag: max(1, round(mult * q5_unique_tokens / H.TOKENS_PER_STEP))
            for tag, mult in H.Q5_EPOCH_RUNS.items()}
        print(f"[fleet_data] q5 subsample {q5_unique_tokens:,} tokens -> "
              f"epoch steps {manifest['q5_epoch_steps']}")

    with open(os.path.join(H.FLEET_WORK, "fleet_data_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


# --- fixed eval slices --------------------------------------------------------

def _texts(path: str, field="text") -> list:
    if not os.path.exists(path):
        return []
    return [json.loads(l)[field] for l in open(path, encoding="utf-8") if l.strip()]


def _flores_texts() -> list:
    if not os.path.exists(G.FLORES_HAT_DEVTEST):
        return []
    return [l.rstrip("\n") for l in open(G.FLORES_HAT_DEVTEST, encoding="utf-8") if l.strip()]


def _general_holdout() -> list:
    """~700kB seeded sample of the v0.1 tokenizer-holdout pool — constructed IDENTICALLY
    to Workstream G's general_holdout (train/tokenize_g.build) so fleet BPB lines up with
    modelc_v0.md's learning curves. Held out of every fleet training bin."""
    exclude = excluded_ids()
    shard = H.CORPUS_SHARDS["v0.1"]
    holdout = []
    for line in open(shard, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        did = d["acquisition"]["doc_id"]
        if did in exclude:
            continue
        if _in_tokenizer_holdout(did):
            holdout.append(d["text"])
    rng = np.random.default_rng(G.TRAIN["seed"] + 1)     # G used seed+1 for hold_order
    order = rng.permutation(len(holdout))
    out, nbytes = [], 0
    for i in order:
        out.append(holdout[i])
        nbytes += len(holdout[i].encode("utf-8"))
        if nbytes >= H.GENERAL_HOLDOUT_BYTES:
            break
    return out


def build_eval_texts() -> dict:
    eval_texts = {
        "general_holdout": _general_holdout(),
        "authored_eval": _texts(F.AUTHORED_EVAL),
        "translation_shaped_eval": _texts(F.TRANSLATION_SHAPED_EVAL),
        "authored_eval_v2": _texts(H.AUTHORED_EVAL_V2),
        "flores_hat": _flores_texts(),
    }
    with open(os.path.join(H.FLEET_BUNDLE, "eval_texts.json"), "w", encoding="utf-8") as fh:
        json.dump(eval_texts, fh, ensure_ascii=False)
    stats = {k: {"docs": len(v), "bytes": sum(len(t.encode("utf-8")) for t in v)}
             for k, v in eval_texts.items()}
    print(f"[fleet_data] eval slices: { {k: s['docs'] for k, s in stats.items()} }")
    return stats


# --- upload -------------------------------------------------------------------

def do_upload():
    import modal
    vol = modal.Volume.from_name(F.MODAL_VOLUME, create_if_missing=True)
    assert os.path.isdir(H.FLEET_BUNDLE), "run `python -m train.fleet_data` first"
    print("[fleet_data] uploading bins + eval + tokenizers to the Volume …")
    with vol.batch_upload(force=True) as b:
        b.put_directory(H.FLEET_BUNDLE, "/fleet/data")
        for tok_key, pkl in H.TOKENIZERS.items():
            b.put_file(pkl, f"/fleet/tokenizers/{tok_key}.pkl")
    print("[fleet_data] upload done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()
    os.makedirs(H.FLEET_WORK, exist_ok=True)
    os.makedirs(H.FLEET_BUNDLE, exist_ok=True)
    if args.upload:
        do_upload()
        return
    if not args.eval_only:
        build_variant_bins()
    build_eval_texts()
    print("[fleet_data] done — run with --upload to push to the Volume")


if __name__ == "__main__":
    main()
