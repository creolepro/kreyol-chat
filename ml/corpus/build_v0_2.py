"""Workstream J6b — assemble corpus v0.2.

Design (chosen by the user: "Full v0.2 + mix control"): keep v0.1 FROZEN as the
base, add the NET-NEW survivors of the J sources (register-tagged), and drive the
authored/register emphasis at TRAIN time via config_v0_2.MIX_WEIGHTS — not by
inflating the corpus's authored share. The report states raw composition honestly.

Pipeline for each new source:
  normalize (NFC + whitespace)  →  per-source junk + langid pass (removals
  reported)  →  dedup vs v0.1 (corpus_index)  →  cross-new dedup (priority
  survivor: owned>authored>wikipedia>crawl)  →  add to shard.

Special handling:
  * VOA: articles.jsonl -> Document records; the NEWEST slice by date_published is
    carved into authored_eval_v2 (held out of the shard, manifest only).
  * Bib La: religious register capped to <= RELIGIOUS_CAP_FRAC of corpus tokens.

Output (untracked): data/clean/corpus_v0_2-full.jsonl + build_stats.json
Run:  uv run python -m corpus.build_v0_2
"""

from __future__ import annotations

import json
import os
import re

from datasketch import MinHashLSH

from . import audit, common, corpus_index, dedup, junk, schema
from . import config_v0_1 as CV
from . import config_v0_2 as C2

NEW_SOURCES = ["voa_nouvel", "fineweb2_hat", "us_federal_pdfs",
               "cfpb_glossary_family", "bib_la_1985", "konstitisyon_1987",
               "storybooks_haiti"]

# fasttext langid: drop only CLEARLY-foreign docs (a non-ht, non-en language at
# high confidence) — federal/glossary docs are legitimately EN-mixed; fineweb is
# pre-filtered at ingest. Kept lenient because lid.176 runs modest on Kreyòl.
_FOREIGN = {"es", "pt", "fr", "it", "de", "nl", "ca", "gl", "ro", "id",
            "tl", "sw", "zh", "ar", "ru", "ja", "ko", "hi", "vi", "tr"}
LANGID_DROP_CONF = 0.60
VOA_EVAL_NEWEST = 400          # newest VOA articles by date -> authored_eval_v2
AUTHORED_EVAL_V2 = os.path.join(common.config.DATA, "eval", "authored_eval_v2.jsonl")


def _norm(text: str) -> str:
    return re.sub(r"[ \t]+", " ", common.nfc(text)).strip()


def _kb():
    from tokenizer.core import KreyolBPE
    return KreyolBPE.load_pkl(os.path.join(common.config.REPO_ROOT, "tokenizer",
                                            "kreyol-bpe", "tokenizer.pkl"))


# --- VOA articles.jsonl -> Document records + authored_eval_v2 carve-out -----
def prep_voa(kb) -> dict:
    art_path = C2.VOA_ARTICLES
    if not os.path.exists(art_path):
        common.log("[build] no VOA articles.jsonl (crawl not run); skipping VOA")
        return {"train": 0, "eval": 0}
    arts = [json.loads(l) for l in open(art_path, encoding="utf-8") if l.strip()]
    # dedup by id (crawl may append across resumes)
    by_id = {a["id"]: a for a in arts}
    arts = list(by_id.values())
    # newest by date_published -> authored_eval_v2 (temporal holdout, no leakage).
    # Adaptive to the actual VOA size so a partial crawl doesn't starve train.
    arts.sort(key=lambda a: (a.get("date_published") or ""), reverse=True)
    eval_n = min(VOA_EVAL_NEWEST, max(50, len(arts) // 4))
    eval_arts = arts[:eval_n]
    train_arts = arts[eval_n:]
    rights = common.rights_for("voa_nouvel")

    def _mk(a, split):
        rec = schema.Document(
            text=_norm(a["text"]), origin="authored_kreyol", genre="news",
            acquisition={
                "source": "voa_nouvel", "source_name": "VOA Nouvèl",
                "url": a["url"], "revision": None,
                "download_timestamp": common.now_iso(),
                "doc_id": f"voa_nouvel:{a['id']}",
                "raw_content_hash": common.content_hash(a["text"]),
            },
            rights=rights, split=split,
        ).model_dump(mode="json")
        rec["register"] = "journalism"
        rec["date_published"] = a.get("date_published")
        rec["author"] = a.get("author")
        return rec

    os.makedirs(os.path.dirname(AUTHORED_EVAL_V2), exist_ok=True)
    with open(AUTHORED_EVAL_V2, "w", encoding="utf-8") as f:
        for a in eval_arts:
            f.write(json.dumps(_mk(a, "exhibit_examples"), ensure_ascii=False) + "\n")
    common.write_jsonl(os.path.join(C2.V0_2_INGEST, "voa_nouvel.jsonl"),
                       [_mk(a, "train") for a in train_arts])
    common.log(f"[build] VOA: {len(train_arts)} train + {len(eval_arts)} authored_eval_v2 "
               f"(newest by date); eval date range "
               f"{(eval_arts[-1].get('date_published') or '?')[:10]}..{(eval_arts[0].get('date_published') or '?')[:10]}"
               if eval_arts else "[build] VOA: no eval")
    return {"train": len(train_arts), "eval": len(eval_arts)}


# The crawl junk filter (price_listing/spam/mt_placeholder/…) was CALIBRATED on
# web crawl. It false-fires on curated PD material (a financial glossary is all
# "$" terms; a legal doc is structured), so it is applied ONLY to web-crawl
# sources. Curated sources are trusted (rights-registered, human/PD).
JUNK_APPLIES = {"fineweb2_hat"}


def _v01_prio(doc_id: str) -> int:
    """Survivor priority of a v0.1 doc from its doc_id source prefix."""
    return C2.survivor_priority(common.priority_class(doc_id.split(":")[0]))


# --- per-source filter (normalize + junk + langid + dedup vs v0.1) -----------
def _filter_source(src: str, lsh_v01, kb, stats: dict, replace_v01: set):
    path = os.path.join(C2.V0_2_INGEST, f"{src}.jsonl")
    if not os.path.exists(path):
        common.log(f"[build] {src}: ingest file missing, skip")
        return []
    prio = common.priority_class(src)
    new_prio = C2.survivor_priority(prio)
    s = {"in": 0, "junk": 0, "langid": 0, "dup_v01": 0, "replaced_v01": 0,
         "kept": 0, "junk_reasons": {}}
    survivors = []
    for d in common.read_jsonl(path):
        s["in"] += 1
        text = _norm(d["text"])
        if len(text) < 40:
            s["junk"] += 1
            continue
        if src in JUNK_APPLIES:
            jr = junk.junk_reason(text, "crawl")
            if jr:
                s["junk"] += 1
                s["junk_reasons"][jr] = s["junk_reasons"].get(jr, 0) + 1
                continue
        lang, conf = audit._lid_predict(text)
        if lang in _FOREIGN and conf >= LANGID_DROP_CONF:
            s["langid"] += 1
            continue
        hits = corpus_index.query(lsh_v01, text)
        if hits:
            # authored-beats-crawl: an authored new doc REPLACES the lower-priority
            # v0.1 dup (so the well-provenanced version + register tag survives);
            # otherwise the v0.1 copy stands and the new doc is dropped.
            if new_prio < min(_v01_prio(h) for h in hits):
                replace_v01.update(hits)
                s["replaced_v01"] += 1
            else:
                s["dup_v01"] += 1
                continue
        d["text"] = text
        d["_mh"] = dedup._minhash(text)
        d["_prio"] = new_prio
        survivors.append(d)
    s["kept"] = len(survivors)
    stats[src] = s
    common.log(f"[build] {src}: in={s['in']} junk={s['junk']} langid={s['langid']} "
               f"dup_v01={s['dup_v01']} replaced_v01={s['replaced_v01']} -> kept {s['kept']}")
    return survivors


# --- cross-new dedup (priority survivor) ------------------------------------
def _cross_dedup(survivors: list) -> list:
    lsh = MinHashLSH(threshold=common.config.MINHASH_THRESHOLD,
                     num_perm=common.config.MINHASH_NUM_PERM)
    by_key = {}
    for i, d in enumerate(survivors):
        d["_k"] = i
        by_key[i] = d
        lsh.insert(i, d["_mh"])
    uf = dedup._UF(list(by_key))
    for i, d in by_key.items():
        for j in lsh.query(d["_mh"]):
            if j != i:
                uf.union(i, j)
    clusters = {}
    for i in by_key:
        clusters.setdefault(uf.find(i), []).append(i)
    kept = []
    for members in clusters.values():
        best = min(members, key=lambda k: (by_key[k]["_prio"],
                                           -len(by_key[k]["text"]),
                                           by_key[k]["acquisition"]["doc_id"]))
        kept.append(by_key[best])
    return kept


def _religious_cap(survivors, v01_tokens, kb, stats):
    """Cap religious-register tokens to <= RELIGIOUS_CAP_FRAC of the whole corpus."""
    rel = [d for d in survivors if d.get("register") == "religious"]
    if not rel:
        return survivors
    new_tokens = sum(kb.count(d["text"]) for d in survivors)
    total = v01_tokens + new_tokens
    rel_tokens = sum(kb.count(d["text"]) for d in rel)
    cap = C2.RELIGIOUS_CAP_FRAC * total
    if rel_tokens <= cap:
        stats["religious_cap"] = {"capped": False, "rel_tokens": rel_tokens,
                                  "cap": int(cap)}
        return survivors
    # keep whole books (by doc order) until the cap is reached
    rel.sort(key=lambda d: kb.count(d["text"]))
    keep_rel, acc = [], 0
    for d in rel:
        t = kb.count(d["text"])
        if acc + t > cap:
            break
        keep_rel.append(d)
        acc += t
    keep_ids = {id(d) for d in keep_rel}
    out = [d for d in survivors if d.get("register") != "religious" or id(d) in keep_ids]
    stats["religious_cap"] = {"capped": True, "rel_tokens_before": rel_tokens,
                              "rel_tokens_after": acc, "cap": int(cap),
                              "books_kept": len(keep_rel), "books_total": len(rel)}
    common.log(f"[build] religious cap: {rel_tokens:,} -> {acc:,} tok "
               f"({len(keep_rel)}/{len(rel)} books)")
    return out


def build() -> dict:
    kb = _kb()
    common.log("[build] loading v0.1 minhash index...")
    lsh_v01, _ = corpus_index.load()
    stats = {"sources": {}}

    # VOA prep (articles -> Documents + eval carve-out)
    stats["voa_prep"] = prep_voa(kb)

    # per-source filter + dedup vs v0.1 (authored may REPLACE v0.1 crawl dups)
    survivors = []
    replace_v01: set = set()
    for src in NEW_SOURCES:
        survivors += _filter_source(src, lsh_v01, kb, stats["sources"], replace_v01)
    stats["v01_replaced_by_authored"] = len(replace_v01)
    common.log(f"[build] {len(replace_v01)} v0.1 crawl docs replaced by authored sources")

    # cross-new dedup
    n_before = len(survivors)
    survivors = _cross_dedup(survivors)
    stats["cross_new_dedup_removed"] = n_before - len(survivors)
    common.log(f"[build] cross-new dedup: {n_before} -> {len(survivors)}")

    # v0.1 base token count (streamed; skip docs replaced by authored sources)
    v01_shard = CV.CORPUS_V0_1.format(tag="full")
    v01_docs = v01_tokens = 0
    for d in common.read_jsonl(v01_shard):
        if d["acquisition"]["doc_id"] in replace_v01:
            continue
        v01_docs += 1
        v01_tokens += kb.count(d["text"])
    survivors = _religious_cap(survivors, v01_tokens, kb, stats)

    # assemble shard: v0.1 (minus replaced) + new survivors (register-tagged)
    out = C2.CORPUS_V0_2.format(tag="full")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    comp = {}   # register -> {docs, tokens}
    with open(out, "w", encoding="utf-8") as f:
        for d in common.read_jsonl(v01_shard):
            if d["acquisition"]["doc_id"] in replace_v01:
                continue
            reg = _v01_register(d)
            d.setdefault("register", reg)
            _tally(comp, reg, kb.count(d["text"]))
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
        for d in survivors:
            for k in ("_mh", "_prio", "_k"):
                d.pop(k, None)
            _tally(comp, d.get("register", "web_crawl"), kb.count(d["text"]))
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    stats["composition_by_register"] = comp
    stats["v01_docs"] = v01_docs
    stats["v01_tokens_kb"] = v01_tokens
    stats["new_survivor_docs"] = len(survivors)
    stats["new_survivor_tokens_kb"] = sum(kb.count(d["text"]) for d in survivors)
    stats["total_docs"] = len(v01_docs) + len(survivors)
    stats["total_tokens_kb"] = v01_tokens + stats["new_survivor_tokens_kb"]
    stats["mix_weights"] = C2.MIX_WEIGHTS
    stats["output_shard"] = os.path.basename(out)

    with open(C2.V0_2_STATS.format(tag="full"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    common.log(f"[build] corpus v0.2: {stats['total_docs']:,} docs / "
               f"{stats['total_tokens_kb']:,} kb-tok "
               f"(+{stats['new_survivor_tokens_kb']:,} net-new) -> {out}")
    return stats


def _v01_register(d) -> str:
    src = d["acquisition"]["source"]
    if src == "ht_wikipedia":
        return "encyclopedic"
    if src == "owned_proverbs":
        return "proverb"
    return "web_crawl"      # madlad


def _tally(comp, reg, toks):
    g = comp.setdefault(reg, {"docs": 0, "tokens": 0})
    g["docs"] += 1
    g["tokens"] += toks


if __name__ == "__main__":
    build()
