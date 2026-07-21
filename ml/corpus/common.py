"""Shared helpers for the corpus pipeline: IO, hashing, sizing, rights lookup.

Kept dependency-light and deterministic. No stage logic here — just the plumbing
every stage reuses so behaviour (hashing, sampling, size units) is identical
across ingest/normalize/filter/dedup/audit/report.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import sys
import unicodedata
from datetime import datetime, timezone

import yaml

from . import config
from .schema import Rights

# --- logging ------------------------------------------------------------------

def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# --- hashing + ids ------------------------------------------------------------

def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def content_hash(text: str) -> str:
    """Short, stable content hash (first 16 hex of sha256)."""
    return sha256_hex(text)[:16]


def deterministic_keep(doc_id: str, frac: float, seed: int) -> bool:
    """Deterministic per-document sampler: keep iff hash(doc_id, seed) < frac.

    Independent of order/parallelism and reproducible across runs — the same
    doc_id is always in or out of the 1% sample.
    """
    if frac >= 1.0:
        return True
    h = hashlib.sha256(f"{seed}:{doc_id}".encode("utf-8")).hexdigest()
    bucket = int(h[:8], 16) / 0xFFFFFFFF
    return bucket < frac


# --- JSONL IO -----------------------------------------------------------------

def read_jsonl(path: str):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str, records) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def count_jsonl(path: str) -> int:
    return sum(1 for _ in read_jsonl(path))


# --- size units (docs/phase-0.md A6: bytes / chars / whitespace words / tokens)-

def n_bytes(text: str) -> int:
    return len(text.encode("utf-8"))


def n_chars(text: str) -> int:
    return len(text)


def n_ws_words(text: str) -> int:
    """Whitespace-split word count (the 'whitespace words' unit)."""
    return len(text.split())


class RefTokenizer:
    """The named reference tokenizer (o200k_base) for report token counts."""

    def __init__(self):
        import tiktoken

        self.name = config.REFERENCE_TOKENIZER
        self._enc = tiktoken.get_encoding(self.name)

    def count(self, text: str) -> int:
        return len(self._enc.encode(text, disallowed_special=()))


# --- text helpers -------------------------------------------------------------

def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def file_mtime_iso(path: str) -> str:
    ts = os.path.getmtime(path)
    return datetime.fromtimestamp(ts, tz=timezone.utc).replace(microsecond=0).isoformat()


# --- rights.yaml lookup -------------------------------------------------------

_RIGHTS_CACHE = None


def _load_rights_yaml() -> dict:
    global _RIGHTS_CACHE
    if _RIGHTS_CACHE is None:
        path = os.path.join(os.path.dirname(__file__), "rights.yaml")
        with open(path, "rt", encoding="utf-8") as f:
            _RIGHTS_CACHE = yaml.safe_load(f)
    return _RIGHTS_CACHE


def rights_for(source_key: str) -> Rights:
    """Build the schema `Rights` block for a source from rights.yaml."""
    doc = _load_rights_yaml()
    entry = next((s for s in doc["sources"] if s["key"] == source_key), None)
    if entry is None:
        raise KeyError(f"source {source_key!r} not in rights.yaml")
    lic = entry["license"]
    uses = entry["uses"]
    return Rights(
        license_id=lic["id"],
        license_url=lic.get("url"),
        analysis=uses["analysis"],
        tokenizer_training=uses["tokenizer_training"],
        model_training=uses["model_training"],
        redistribution=uses["redistribution"],
    )


def priority_class(source_key: str) -> str:
    doc = _load_rights_yaml()
    entry = next((s for s in doc["sources"] if s["key"] == source_key), None)
    return entry["priority_class"] if entry else "crawl"


# --- run tags + stage paths ---------------------------------------------------

def run_tag(sample: bool) -> str:
    return "sample" if sample else "full"


def stage_path(tag: str, stage: str, name: str) -> str:
    return os.path.join(config.INTERIM, tag, stage, f"{name}.jsonl")


# --- downloads (re-runnable; skip if present) ---------------------------------

def ensure_madlad() -> str:
    if os.path.exists(config.MADLAD_LOCAL):
        return config.MADLAD_LOCAL
    from huggingface_hub import hf_hub_download

    log(f"  downloading MADLAD {config.MADLAD_HT_CLEAN_REPO_FILE} @ {config.MADLAD_REVISION[:12]}")
    os.makedirs(config.DOWNLOADS, exist_ok=True)
    path = hf_hub_download(
        repo_id=config.MADLAD_REPO,
        filename=config.MADLAD_HT_CLEAN_REPO_FILE,
        revision=config.MADLAD_REVISION,
        repo_type="dataset",
        token=os.environ.get("HF_TOKEN"),
        local_dir=config.DOWNLOADS,
    )
    # hf_hub_download preserves the repo path; symlink/rename to the flat name.
    if path != config.MADLAD_LOCAL and os.path.exists(path):
        os.replace(path, config.MADLAD_LOCAL)
    return config.MADLAD_LOCAL


def ensure_url(url: str, local: str) -> str:
    if os.path.exists(local):
        return local
    import urllib.request

    log(f"  downloading {url}")
    os.makedirs(os.path.dirname(local), exist_ok=True)
    tmp = local + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, local)
    return local


def ensure_htwiki() -> str:
    return ensure_url(config.HTWIKI_URL, config.HTWIKI_LOCAL)


def ensure_lid_model() -> str:
    return ensure_url(config.LID_MODEL_URL, config.LID_MODEL_LOCAL)
