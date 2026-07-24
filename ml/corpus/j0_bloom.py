"""J0 — Bloom Library (sil-ai/bloom-lm) hat: per-entry license split.

260 hat stories (208 train / 26 val / 26 test). Each carries a per-entry `license`
(cc-by-4.0, cc-by-nc-4.0, cc-by-nd-4.0, cc-by-sa, ...). KEEP BY / BY-SA; DROP any
NC or ND (docs/data.md §1). Reports keeper count + kreyol-bpe tokens.

Access: gated but AUTO-GRANTING ("agree to share your contact information").
The HF account must have accepted the gate once (corpus/hf_gate.py).

Run:  uv run python -m corpus.j0_scoping bloom
"""

from __future__ import annotations

import os

from . import common, j0_scoping

BLOOM_REPO = "sil-ai/bloom-lm"
BLOOM_CONFIG = "hat"
SPLITS = ["train", "validation", "test"]


def is_keeper(license_str: str) -> bool:
    """Keep CC-BY / CC-BY-SA; drop anything with NC or ND."""
    s = (license_str or "").lower().replace("_", "-").strip()
    if not s.startswith("cc-by"):
        return False
    return ("-nc" not in s) and ("-nd" not in s)


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

    by_license: dict[str, dict] = {}
    keeper_docs = keeper_tokens = 0
    total_docs = total_tokens = 0
    per_split = {}

    for split in SPLITS:
        try:
            ds = load_dataset(BLOOM_REPO, BLOOM_CONFIG, split=split,
                              token=os.environ.get("HF_TOKEN"))
        except Exception as e:
            common.log(f"[j0.bloom] split {split} load failed: {type(e).__name__}: {str(e)[:160]}")
            per_split[split] = {"error": f"{type(e).__name__}: {str(e)[:160]}"}
            continue
        s_docs = s_keep = 0
        for r in ds:
            lic = (r.get("license") or "unknown").lower().replace("_", "-").strip()
            text = r.get("text") or ""
            tok = kb.count(text)
            total_docs += 1
            total_tokens += tok
            s_docs += 1
            g = by_license.setdefault(lic, {"docs": 0, "tokens": 0, "keeper": is_keeper(lic)})
            g["docs"] += 1
            g["tokens"] += tok
            if is_keeper(lic):
                keeper_docs += 1
                keeper_tokens += tok
                s_keep += 1
        per_split[split] = {"docs": s_docs, "keeper_docs": s_keep}

    agg = {
        "repo": BLOOM_REPO, "config": BLOOM_CONFIG,
        "total_docs": total_docs, "total_tokens_kb": total_tokens,
        "keeper_docs": keeper_docs, "keeper_tokens_kb": keeper_tokens,
        "by_license": dict(sorted(by_license.items(), key=lambda kv: -kv[1]["tokens"])),
        "per_split": per_split,
    }
    j0_scoping._save("bloom.json", agg)
    common.log(f"[j0.bloom] {total_docs} docs; keepers (BY/BY-SA) = {keeper_docs} docs / "
               f"{keeper_tokens:,} kb-tok")
    for lic, g in agg["by_license"].items():
        common.log(f"    {lic:18s} docs={g['docs']:4d} tok={g['tokens']:7,d} "
                   f"{'KEEP' if g['keeper'] else 'drop'}")
    return agg
