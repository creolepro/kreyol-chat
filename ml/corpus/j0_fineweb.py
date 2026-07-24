"""J0 — fineweb-2 hat_Latn: raw size + TRUE net-new after junk + dedup vs corpus.

docs/data.md flags this as "a dedup-merge experiment, not new volume" (expect
heavy MADLAD overlap, since both derive from Common Crawl). We measure exactly:
raw tokens, tokens surviving the standing junk filter, and tokens surviving
MinHash dedup against the existing v0.1 corpus (the real net-new).

fineweb-2 hat_Latn ships its own fastText langid (language='hat',
language_score~1.0), so the langid pass here is a light score gate; the uniform
fastText pass is re-run for every source at J6 build time.

Run:  uv run python -m corpus.j0_scoping fineweb   (after corpus_index build)
"""

from __future__ import annotations

import os

from . import common, corpus_index, j0_scoping, junk

FINEWEB_REPO = "HuggingFaceFW/fineweb-2"
FINEWEB_CONFIG = "hat_Latn"
LANGID_MIN = 0.65   # fastText hat score floor (fineweb's own score)


def _load_token():
    env = os.path.join(os.path.dirname(common.__file__), "..", "..", ".env")
    if os.path.exists(env):
        for line in open(env):
            if line.strip().startswith("HF_TOKEN"):
                tok = line.split("=", 1)[1].strip().strip('"').strip("'")
                os.environ["HF_TOKEN"] = tok
                os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", tok)


def run():
    _load_token()
    from datasets import load_dataset
    from tokenizer.core import KreyolBPE
    kb = KreyolBPE.load_pkl(os.path.join(
        os.path.dirname(common.__file__), "..", "tokenizer", "kreyol-bpe",
        "tokenizer.pkl"))
    o2 = common.RefTokenizer()

    common.log("[j0.fineweb] loading corpus v0.1 minhash index...")
    lsh, mh = corpus_index.load()

    common.log(f"[j0.fineweb] streaming {FINEWEB_REPO}:{FINEWEB_CONFIG} ...")
    ds = load_dataset(FINEWEB_REPO, FINEWEB_CONFIG, split="train", streaming=True)

    n = 0
    raw_kb = raw_o2 = 0
    junk_docs = junk_kb = 0
    langid_docs = 0
    dup_docs = dup_kb = 0
    net_docs = net_kb = net_o2 = 0
    junk_reasons: dict[str, int] = {}

    for r in ds:
        text = r.get("text") or ""
        if not text.strip():
            continue
        n += 1
        ktok = kb.count(text)
        raw_kb += ktok
        raw_o2 += o2.count(text)
        # langid gate (fineweb's own score)
        try:
            score = float(r.get("language_score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        if r.get("language") != "hat" or score < LANGID_MIN:
            langid_docs += 1
            continue
        # junk gate
        reason = junk.junk_reason(text, "crawl")
        if reason:
            junk_docs += 1
            junk_kb += ktok
            junk_reasons[reason] = junk_reasons.get(reason, 0) + 1
            continue
        # dedup gate
        if corpus_index.query(lsh, text):
            dup_docs += 1
            dup_kb += ktok
            continue
        net_docs += 1
        net_kb += ktok
        net_o2 += o2.count(text)
        if n % 10000 == 0:
            common.log(f"    {n:,} docs | net-new so far {net_kb:,} kb-tok")

    agg = {
        "repo": FINEWEB_REPO, "config": FINEWEB_CONFIG,
        "raw_docs": n,
        "raw_tokens_kb": raw_kb, "raw_tokens_o200k": raw_o2,
        "langid_dropped_docs": langid_docs,
        "junk_dropped_docs": junk_docs, "junk_dropped_tokens_kb": junk_kb,
        "junk_reasons": junk_reasons,
        "dedup_dropped_docs": dup_docs, "dedup_dropped_tokens_kb": dup_kb,
        "net_new_docs": net_docs,
        "net_new_tokens_kb": net_kb, "net_new_tokens_o200k": net_o2,
        "net_new_frac_of_raw_tokens": (net_kb / raw_kb) if raw_kb else 0.0,
    }
    j0_scoping._save("fineweb2.json", agg)
    common.log(f"[j0.fineweb] raw={n:,} docs / {raw_kb:,} kb-tok | "
               f"langid-drop={langid_docs} junk-drop={junk_docs} dedup-drop={dup_docs} | "
               f"NET-NEW {net_docs:,} docs / {net_kb:,} kb-tok "
               f"({agg['net_new_frac_of_raw_tokens']:.1%} of raw)")
    return agg
