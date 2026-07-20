"""Data acquisition — Petrov replication inputs and the FLORES+ measurement set.

Both land in the git-ignored ml/data/ tree; neither corpus is committed or
re-hosted (FLORES+ is eval-only by its terms; Petrov's bundle is CC-BY-SA
FLORES-200 that we read but do not redistribute).
"""

from __future__ import annotations

import csv
import json
import os
import urllib.request

import tiktoken

from . import config
from .counting import nfc


# --- Petrov et al. 2023 (pipeline validation) ---------------------------------

def _download(url: str, dest: str):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        return dest
    urllib.request.urlretrieve(url, dest)
    return dest


def fetch_petrov(data_dir: str, log) -> dict:
    """Download his per-language token CSV + FLORES-200 devtest sentences."""
    base = os.path.join(data_dir, "raw", "petrov")
    paths = {}
    csv_dest = os.path.join(base, "tokenization_lengths.csv")
    _download(config.PETROV_RAW_BASE + config.PETROV_LENGTHS_CSV, csv_dest)
    paths["csv"] = csv_dest
    for code, rel in config.PETROV_DEVTEST.items():
        dest = os.path.join(base, os.path.basename(rel))
        _download(config.PETROV_RAW_BASE + rel, dest)
        paths[code] = dest
    log(f"  petrov data in {base} (commit {config.PETROV_COMMIT[:12]})")
    return paths


def _read_lines(path: str):
    with open(path, encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]


def petrov_replication(data_dir: str, log) -> dict:
    """Reproduce his ht/en cl100k parity two ways:

      his_csv_parity  — straight from his released per-language totals.
      our_parity      — OUR tiktoken cl100k code re-tokenizing his sentences.

    The second is the real pipeline check: our counting code on his data must
    land on his ~1.74x.
    """
    paths = fetch_petrov(data_dir, log)

    # (a) his released totals
    with open(paths["csv"], encoding="utf-8") as f:
        rows = {r["Language"]: r for r in csv.DictReader(f)}
    his_ht = int(rows[config.PETROV_CSV_LANG["ht"]][config.PETROV_CL100K_COL])
    his_en = int(rows[config.PETROV_CSV_LANG["en"]][config.PETROV_CL100K_COL])
    his_csv_parity = his_ht / his_en

    # (b) our counting code on his sentences
    enc = tiktoken.get_encoding("cl100k_base")
    ht_lines = _read_lines(paths["ht"])
    en_lines = _read_lines(paths["en"])
    our_ht = sum(len(enc.encode(x, disallowed_special=())) for x in ht_lines)
    our_en = sum(len(enc.encode(x, disallowed_special=())) for x in en_lines)
    our_parity = our_ht / our_en

    return {
        "his_ht_tokens": his_ht,
        "his_en_tokens": his_en,
        "his_csv_parity": his_csv_parity,
        "our_ht_tokens": our_ht,
        "our_en_tokens": our_en,
        "our_parity": our_parity,
        "n_sentences_our": len(ht_lines),
        "note": ("his totals span dev+devtest (2009 sents); our recompute is "
                 "devtest-only (1012) — parity is scale-invariant, so ratios match"),
    }


# --- FLORES+ (our own measurement) --------------------------------------------

def load_flores(hf_token: str, log) -> dict:
    """Download the pinned devtest JSONL for ht/en/fr, NFC-normalize, join by id.

    datasets>=4 no longer executes dataset loading scripts, and flores_plus ships
    plain JSONL data files, so we fetch those directly at the pinned revision —
    which also keeps the exact version explicit.
    """
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
        with open(path, encoding="utf-8") as f:
            rows = {}
            for line in f:
                r = json.loads(line)
                rows[str(r["id"])] = nfc(r["text"])
        by_lang[code] = rows

    common = set.intersection(*(set(rows) for rows in by_lang.values()))
    ids = sorted(common, key=int)
    aligned = {code: [by_lang[code][i] for i in ids] for code in config.FLORES_LANGS}
    log(f"  FLORES+ {config.FLORES_REPO}@{config.FLORES_REVISION[:12]} "
        f"{config.FLORES_SPLIT}: joined {len(ids)} sentences by (split, id)")
    return {"ids": ids, "n": len(ids), "texts": aligned}
