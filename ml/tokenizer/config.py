"""Pinned configuration for Workstream B (Kreyòl tokenizer v0).

Everything reproducibility-critical: the pinned nanochat commit whose rustbpe +
split-pattern + special-token conventions we match, the vocab sweep, seeds, the
training-sample source weighting, and the model-width constant used for the
embedding-cost table. See docs/phase-0.md Workstream B (incl. B0) and
docs/plan.md §3.2.
"""

from __future__ import annotations

import os

SNAPSHOT_DATE = "2026-07-20"

# --- pinned nanochat (rustbpe + conventions) ----------------------------------
# github.com/karpathy/nanochat @ this commit. rustbpe itself is the standalone
# PyPI package `rustbpe` (>=0.1.0, pinned in pyproject); this SHA pins the
# SPLIT_PATTERN / SPECIAL_TOKENS / train recipe we copy from nanochat/tokenizer.py.
NANOCHAT_COMMIT = "92d63d4e8bb4df75c3b71618f31ddde2378b2bcd"

# nanochat's 9 special tokens, inserted AFTER training (so rustbpe trains
# vocab_size - len(SPECIAL_TOKENS)). Copied verbatim from nanochat/tokenizer.py.
SPECIAL_TOKENS = [
    "<|bos|>",
    "<|user_start|>", "<|user_end|>",
    "<|assistant_start|>", "<|assistant_end|>",
    "<|python_start|>", "<|python_end|>",
    "<|output_start|>", "<|output_end|>",
]

# nanochat's GPT-4-style pre-tokenization regex (\p{N}{1,2}, not {1,3}).
# The `'(?i:[sdmt]|ll|ve|re)` clause is English contraction handling — the
# subject of the B0 apostrophe probe.
STOCK_SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)

# Kreyòl-aware variant: DROP the English-contraction clause so an apostrophe is
# handled uniformly by the general letter clause (attaches to the following
# letters). Fixes the `'t`/`'s`/`'d`/`'m` misfire on Kreyòl forms like m'te /
# n'ta while leaving m'ap-family and English single-letter contractions
# identical. The B0 spike measures this and CHOSEN_SPLIT_PATTERN records the pick.
KREYOL_SPLIT_PATTERN = (
    r"""[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)

# Set by the B0 decision (see rustbpe_spike.md). Used by the sweep + exports.
CHOSEN_SPLIT_PATTERN = KREYOL_SPLIT_PATTERN
CHOSEN_PATTERN_NAME = "kreyol_aware"   # "stock" | "kreyol_aware"

# --- vocab sweep --------------------------------------------------------------
VOCAB_SWEEP = [8192, 16384, 24576, 32768]   # 8k, 16k, 24k, 32k
CHOSEN_VOCAB = None                          # set after the sweep (report.py reads results)

# --- model width for the embedding-cost table (read from the pinned commit) ---
# nanochat sizes width from --depth as model_dim = depth * 64 (see
# nanochat/gpt.py at NANOCHAT_COMMIT). d12 => 768. lm_head is UNTIED from the
# embedding (separate matrices), so the token-embedding parameter cost is
# vocab * dim * 2.
NANOCHAT_D12_DEPTH = 12
NANOCHAT_DIM_PER_DEPTH = 64
D12_MODEL_DIM = NANOCHAT_D12_DEPTH * NANOCHAT_DIM_PER_DEPTH   # 768
EMBED_MATRICES = 2   # wte + lm_head (untied)

# --- data / sampling ----------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../ml
DATA = os.path.join(REPO_ROOT, "data")
CORPUS = os.path.join(DATA, "clean", "corpus_v0-full.jsonl")
PROVERBS_PROBE = os.path.join(DATA, "eval", "proverbs_probe.jsonl")   # NEVER train/eval on this
REPORTS = os.path.join(REPO_ROOT, "reports")
TOK_DIR = os.path.dirname(os.path.abspath(__file__))                  # ml/tokenizer
ARTIFACTS = os.path.join(TOK_DIR, "kreyol-bpe")                       # committed chosen tokenizer
WORK = os.path.join(DATA, "tokenizer_work")                          # git-ignored sweep scratch

# Deterministic split of corpus-v0 train docs into a tokenizer_eval holdout and
# the training pool. Seeded; holdout is NEVER trained on.
HOLDOUT_FRAC = 0.015          # ~1.5% of docs, stratified by source
SPLIT_SEED = 20260720
SAMPLE_SEED = 20260720

# Training-sample size (chars) fed to the BPE trainer per run. BPE needs far less
# than the full 484M-char corpus; a seeded sample keeps sweep runs fast + equal.
TRAIN_SAMPLE_CHARS = 120_000_000    # ~120M chars (~25% of corpus)

# Explicit source weighting for the PRIMARY training sample.
# "natural": sample in the corpus's own token proportions (the explicit primary
# choice — a Kreyòl tokenizer trained on what the corpus actually is). Recorded
# in the report with rationale.
PRIMARY_WEIGHTING = "natural"
# Sensitivity variant (16k only): cap crawl at ~60% of sampled chars, upweight
# Wikipedia + proverbs. If metrics barely move, the weighting question is closed.
SENSITIVITY_WEIGHTS = {"crawl": 0.60, "wikipedia": 0.39, "owned": 0.01}

# --- exhibit: whole-word survival --------------------------------------------
# Core grammar list (same as Workstream C) + top-N corpus-frequency words.
CORE_WORDS = [
    "te", "ta", "ap", "pral", "va",
    "mwen", "m", "ou", "w", "li", "l",
    "nou", "n", "yo", "y",
    "pa",
    "la", "a", "an", "lan", "nan",
    "sa", "ki", "gen", "fè",
]
TOP_WORDS_N = 500   # top-500 corpus-frequency words (train split, probe excluded)

# --- FLORES+ for out-of-domain / regression compression (measurement only) ----
# Reuse the exact pins from Workstream C's fertility config.
FLORES_REPO = "openlanguagedata/flores_plus"
FLORES_REVISION = "b3a5298db5721c8a682e7ef00a37fcc9ab522757"
FLORES_SPLIT = "devtest"
FLORES_LANGS = {"ht": "hat_Latn", "en": "eng_Latn", "fr": "fra_Latn"}

# --- English-control ablation -------------------------------------------------
# wikitext-103-raw-v1 (CC-BY-SA 3.0) — a size-matched English sample, one 16k
# tokenizer, to show which Kreyòl words shatter under an English tokenizer.
WIKITEXT_REPO = "Salesforce/wikitext"
WIKITEXT_CONFIG = "wikitext-103-raw-v1"
WIKITEXT_SPLIT = "train"
ENGLISH_ABLATION_VOCAB = 16384
