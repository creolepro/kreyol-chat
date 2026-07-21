"""Pinned configuration for Workstream A (corpus v0) + Workstream 0 registries.

Everything that must be recorded for reproducibility lives here: source
revisions/dump dates, filter thresholds, dedup parameters, the audit sample
plan, and the reference tokenizer used for token counts in the report. See
docs/phase-0.md (Workstreams 0 + A) and docs/plan.md §5.2.
"""

from __future__ import annotations

import os

# Build snapshot date. Passed explicitly (not read from the clock) so a re-run
# reproduces the same stamped report.
SNAPSHOT_DATE = "2026-07-20"

# Repo-root-relative data tree (all git-ignored). Resolved against REPO_ROOT.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../ml
DATA = os.path.join(REPO_ROOT, "data")
DOWNLOADS = os.path.join(DATA, "raw", "downloads")     # immutable source downloads
INTERIM = os.path.join(DATA, "interim")                # per-stage JSONL (per run tag)
CLEAN = os.path.join(DATA, "clean")                    # final corpus v0 shards
REPORTS = os.path.join(REPO_ROOT, "reports")

# --- Source: MADLAD-400 ht clean (rights: redistribution UNRESOLVED) ----------
MADLAD_REPO = "allenai/MADLAD-400"
MADLAD_REVISION = "9d886a76bd8fa69b294f2dd3843dacb8388ee5a5"
MADLAD_HT_CLEAN_REPO_FILE = "data/ht/ht_clean_0000.jsonl.gz"
MADLAD_LOCAL = os.path.join(DOWNLOADS, "madlad_ht_clean_0000.jsonl.gz")
MADLAD_SOURCE_KEY = "madlad_400_ht_clean"

# --- Source: Haitian Creole Wikipedia (rights: CC-BY-SA, clear) ---------------
HTWIKI_DUMP_DATE = "20260701"
HTWIKI_URL = (
    f"https://dumps.wikimedia.org/htwiki/{HTWIKI_DUMP_DATE}/"
    f"htwiki-{HTWIKI_DUMP_DATE}-pages-articles.xml.bz2"
)
HTWIKI_LOCAL = os.path.join(
    DOWNLOADS, f"htwiki-{HTWIKI_DUMP_DATE}-pages-articles.xml.bz2"
)
HTWIKI_SOURCE_KEY = "ht_wikipedia"

# --- Source: owned proverbs (CreolePro blog; rights: owned, redistributable) ---
# CreolePro's own curated list of 50 traditional Haitian Creole proverbs. Owned
# outright (rights.yaml owned_proverbs). Split teachable (into train) vs probe
# (held-out, never trained) per splits.yaml.
PROVERBS_SOURCE_KEY = "owned_proverbs"
PROVERBS_URL = "https://www.creolepro.com/blog/haitian-creole-proverbs-cultural-wisdom"
PROVERBS_FETCH_DATE = "2026-07-20"
PROVERBS_LOCAL = os.path.join(DATA, "eval", "proverbs.jsonl")          # git-ignored
PROVERBS_PROBE_LOCAL = os.path.join(DATA, "eval", "proverbs_probe.jsonl")  # held-out
PROVERBS_EXPECTED_COUNT = 50
PROVERBS_PROBE_N = 15          # held-out probe items (never in training/tokenizer)
PROVERBS_SPLIT_SEED = 20260720

# Pipeline source tags -> rights.yaml source key + the sources the pipeline walks.
SOURCE_KEYS = {
    "madlad": MADLAD_SOURCE_KEY,
    "htwiki": HTWIKI_SOURCE_KEY,
    "proverbs": PROVERBS_SOURCE_KEY,
}
PIPELINE_SOURCES = ["madlad", "htwiki", "proverbs"]

# --- Language id (quality audit) ----------------------------------------------
LID_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
LID_MODEL_LOCAL = os.path.join(DOWNLOADS, "lid.176.ftz")
TARGET_LANG = "ht"  # fasttext label we expect for good Kreyòl docs

# --- Reference tokenizer for report token counts ------------------------------
# Named per docs/phase-0.md A6 ("tokens under a named reference tokenizer").
# o200k_base (the GPT-4o-era tokenizer) — the same tiktoken encoding Workstream C
# reports, so corpus + fertility numbers use a consistent unit.
REFERENCE_TOKENIZER = "o200k_base"

# --- 1%-sample run (docs/phase-0.md: run the full pipeline on 1% first) --------
SAMPLE_FRAC = 0.01
SAMPLE_SEED = 20260720  # seeds the deterministic per-doc sample selector

# --- Normalization ------------------------------------------------------------
# Paragraph boundary token in the normalized text (a blank line == \n\n).
PARAGRAPH_SEP = "\n\n"

# --- Filter thresholds (SOURCE-SPECIFIC — a single floor would delete proverbs
#     and conversational text while under-filtering crawl) ----------------------
FILTER = {
    # crawl (MADLAD): stricter — it can afford it, and it needs it.
    "crawl": {
        "min_chars": 200,
        "min_words": 30,
        "max_symbol_ratio": 0.30,     # non-alnum, non-space share of chars
        "max_digit_ratio": 0.20,
        "max_replacement_char_ratio": 0.005,  # mojibake (U+FFFD) guard
        "min_mean_word_len": 2.0,
    },
    # wikipedia: gentler floor (short real stubs exist); bot-stubs are FLAGGED,
    # not dropped (docs/phase-0.md).
    "wikipedia": {
        "min_chars": 80,
        "min_words": 10,
        "max_symbol_ratio": 0.40,
        "max_digit_ratio": 0.35,
        "max_replacement_char_ratio": 0.02,
        "min_mean_word_len": 1.8,
    },
    # owned (proverbs/authored): tiny floor — proverbs are short by nature.
    "owned": {
        "min_chars": 8,
        "min_words": 2,
        "max_symbol_ratio": 0.50,
        "max_digit_ratio": 0.50,
        "max_replacement_char_ratio": 0.05,
        "min_mean_word_len": 1.5,
    },
}

# --- Wikipedia bot-stub heuristic ---------------------------------------------
# htwiki is heavily bot-generated (geo/species stubs). Flagged if the article is
# short AND looks template-driven. Tuned to be simple + documented, not perfect.
BOTSTUB = {
    "max_plaintext_chars": 600,       # stubs are short
    "min_template_ratio": 0.0008,     # {{...}} count per plaintext char
    "geo_stub_patterns": [            # common Kreyòl bot-stub opening shapes
        r"\bse yon (vil|komin|seksyon kominal|depatman|katye|rivyè|zile|komin nan)\b",
        r"\bse yon (espès|plant|bèt|zwazo|pwason|ensèk)\b",
        r"\bse yon (ane|dat)\b",
    ],
}

# --- Dedup --------------------------------------------------------------------
MINHASH_NUM_PERM = 128
MINHASH_THRESHOLD = 0.8          # Jaccard ~0.8 near-dup
SHINGLE_N = 5                    # 5-gram (word) shingles
DEDUP_MIN_PARAGRAPH_CHARS = 40   # paragraph-level dedup ignores tiny fragments
# Dedup survivor priority: LOWER wins. owned > wikipedia > crawl (docs/phase-0.md).
SURVIVOR_PRIORITY = {"owned": 0, "wikipedia": 1, "crawl": 2, "eval": 9}

# --- Quality audit ------------------------------------------------------------
AUDIT_SAMPLE_SIZE = 200
AUDIT_SEED = 20260720
# Char-count length bands used to stratify the audit sample.
AUDIT_LENGTH_BANDS = [
    ("xs", 0, 200),
    ("s", 200, 1000),
    ("m", 1000, 4000),
    ("l", 4000, 10 ** 12),
]
# Boilerplate/unreadable heuristics for the machine first-pass audit.
AUDIT = {
    "min_langid_conf_for_ht": 0.50,   # below this, "ht" call is low-confidence
    "boilerplate_repeat_line_ratio": 0.30,  # >30% duplicate lines == boilerplate-ish
    "unreadable_symbol_ratio": 0.35,
    "unreadable_replacement_ratio": 0.01,
}


def sources_meta():
    """(key, priority_class, revision) tuples recorded in the report header."""
    return [
        (MADLAD_SOURCE_KEY, "crawl", MADLAD_REVISION),
        (HTWIKI_SOURCE_KEY, "wikipedia", f"dump {HTWIKI_DUMP_DATE}"),
        (PROVERBS_SOURCE_KEY, "owned", f"creolepro blog {PROVERBS_FETCH_DATE}"),
    ]
