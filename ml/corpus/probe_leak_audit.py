"""Part 0b — probe-proverb leak provenance audit (Workstream H).

J's v0.2 build removed 19 probe-leak docs (11 net-new in fineweb-2 + 8 pre-existing
in v0.1's MADLAD, which Workstream E's absence claim missed). The build stats record
the COUNTS but not WHICH of the 15 held-out probe proverbs matched, nor whether probe
**#31 ("Lè chat pa la, rat pran kay.")** — the one Workstream E verified at 0 train
hits and that footnotes the Model C v0 exhibit claim — was among the v0.1 leaks.

This reproduces the build's exact detector (`build_v0_2._pnorm` / `_has_probe`:
NFC + lowercase + whitespace-collapse, substring match, probe-text len>15 filter) and
scans each shard (v0, v0.1, v0.2.1), reporting per-probe doc hits with example doc_ids
+ sources. Pure CPU. Writes a JSON sidecar (git-ignored) the fleet report reads.

Run:  cd ml && uv run python -m corpus.probe_leak_audit
"""

from __future__ import annotations

import json
import os
import re
import unicodedata

from . import config


PROBE_FILE = os.path.join(config.DATA, "eval", "proverbs_probe.jsonl")
SHARDS = {
    "v0":     os.path.join(config.CLEAN, "corpus_v0-full.jsonl"),
    "v0.1":   os.path.join(config.CLEAN, "corpus_v0_1-full.jsonl"),
    "v0.2.1": os.path.join(config.CLEAN, "corpus_v0_2_1-full.jsonl"),
}
OUT = os.path.join(config.DATA, "interim", "probe_leak_audit.json")


def _pnorm(s: str) -> str:
    # EXACT copy of build_v0_2._pnorm — the detector we are auditing.
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s).lower()).strip()


def load_probes():
    """The 15 held-out probe proverbs. `checked` mirrors the build's len>15 gate."""
    probes = []
    for line in open(PROBE_FILE, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        t = (d.get("text") or d.get("kreyol") or "").strip()
        probes.append({
            "num": d.get("proverb_num"),
            "text": t,
            "norm": _pnorm(t),
            "checked": len(t) > 15,   # build only scans probes with raw len > 15
        })
    return probes


def scan(shard_path: str, probes: list) -> dict:
    checked = [p for p in probes if p["checked"]]
    hits = {p["num"]: [] for p in probes}   # proverb_num -> list of {doc_id, source}
    n_docs = 0
    n_leak_docs = 0
    for line in open(shard_path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        n_docs += 1
        norm = _pnorm(d["text"])
        matched = [p for p in checked if p["norm"] in norm]
        if matched:
            n_leak_docs += 1
            acq = d.get("acquisition", {})
            for p in matched:
                if len(hits[p["num"]]) < 8:   # keep a few examples per probe
                    hits[p["num"]].append({
                        "doc_id": acq.get("doc_id"),
                        "source": acq.get("source"),
                    })
    return {
        "docs": n_docs,
        "leak_docs": n_leak_docs,                        # docs matching >=1 probe (build metric)
        "per_probe": {num: v for num, v in hits.items() if v},
        "per_probe_counts": {num: len(v) for num, v in hits.items() if v},
    }


def main():
    probes = load_probes()
    print(f"[probe-audit] loaded {len(probes)} probes; "
          f"{sum(p['checked'] for p in probes)} pass the build's len>15 gate")
    excluded = [(p["num"], p["text"]) for p in probes if not p["checked"]]
    if excluded:
        print(f"[probe-audit] NOT scanned (len<=15, build skips): {excluded}")

    out = {"probes": [{"num": p["num"], "text": p["text"], "checked": p["checked"]}
                      for p in probes],
           "shards": {}}
    for name, path in SHARDS.items():
        if not os.path.exists(path):
            print(f"[probe-audit] {name}: SHARD MISSING ({path})")
            continue
        print(f"[probe-audit] scanning {name} …")
        res = scan(path, probes)
        out["shards"][name] = res
        leaked_nums = sorted(res["per_probe_counts"].keys())
        print(f"  {name}: {res['docs']:,} docs, {res['leak_docs']} leak-docs, "
              f"probes matched (by num): {leaked_nums}")
        for num in leaked_nums:
            txt = next(p["text"] for p in probes if p["num"] == num)
            print(f"     #{num} «{txt}» -> {res['per_probe_counts'][num]} doc(s), "
                  f"e.g. {res['per_probe'][num][:2]}")

    # The load-bearing question: was #31 among the v0.1 leaks?
    v01 = out["shards"].get("v0.1", {})
    p31 = v01.get("per_probe_counts", {}).get(31, 0)
    out["probe_31_in_v0_1"] = p31 > 0
    out["probe_31_v0_1_hits"] = p31
    v021 = out["shards"].get("v0.2.1", {})
    out["v0_2_1_clean"] = v021.get("leak_docs", -1) == 0
    print(f"\n[probe-audit] #31 'Lè chat pa la' in v0.1 train? "
          f"{'YES — FLAG' if p31 else 'no (0 hits) — E was right'}")
    print(f"[probe-audit] v0.2.1 shard clean of probe leaks? {out['v0_2_1_clean']}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"[probe-audit] wrote {OUT}")
    return out


if __name__ == "__main__":
    main()
