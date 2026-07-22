"""Pinned configuration for Workstream F — nanochat-on-Modal training + conversion.

Everything reproducibility-critical: the pinned nanochat commit (same as the
tokenizer), the Modal image/GPU/Volume knobs, the base-dir layout nanochat expects,
the throwaway-run hyperparameters, and the log-then-linear checkpoint schedule
(Workstream G's token points). See docs/phase-1.md Workstream F and docs/plan.md §3.2.
"""

from __future__ import annotations

import os

SNAPSHOT_DATE = "2026-07-21"

# --- pinned nanochat (SAME commit as the Workstream-B tokenizer) --------------
NANOCHAT_REPO = "https://github.com/karpathy/nanochat.git"
NANOCHAT_COMMIT = "92d63d4e8bb4df75c3b71618f31ddde2378b2bcd"

# --- repo paths ---------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../ml
DATA = os.path.join(REPO_ROOT, "data")
# Committed kreyol-bpe tokenizer artifacts (Workstream B). We rebuild nanochat's
# tokenizer bundle (a pickled tiktoken Encoding + token_bytes.pt) from these.
KREYOL_BPE_DIR = os.path.join(REPO_ROOT, "tokenizer", "kreyol-bpe")
KREYOL_BPE_PKL = os.path.join(KREYOL_BPE_DIR, "tokenizer.pkl")     # {mergeable_ranks, pattern}
KREYOL_BPE_META = os.path.join(KREYOL_BPE_DIR, "meta.json")
KREYOL_BPE_HF_JSON = os.path.join(KREYOL_BPE_DIR, "tokenizer.json")  # HF bridge (for the convert probe)
# corpus v0.1 (Workstream E) + the held-out sets to exclude from training.
CORPUS_V0_1 = os.path.join(DATA, "clean", "corpus_v0_1-{tag}.jsonl")
EVAL_DIR = os.path.join(DATA, "eval")
AUTHORED_EVAL = os.path.join(EVAL_DIR, "authored_eval.jsonl")
TRANSLATION_SHAPED_EVAL = os.path.join(EVAL_DIR, "translation_shaped_eval.jsonl")
PROVERBS_PROBE = os.path.join(EVAL_DIR, "proverbs_probe.jsonl")

# Local scratch for the nanochat data/tokenizer bundle we upload to the Volume.
WORK = os.path.join(DATA, "train_work")            # git-ignored
BUNDLE_TOKENIZER = os.path.join(WORK, "tokenizer")  # tokenizer.pkl + token_bytes.pt
BUNDLE_DATA = os.path.join(WORK, "base_data_climbmix")  # parquet shards

# nanochat's 9 special tokens (verbatim; appended after the 24,567 content ranks).
SPECIAL_TOKENS = [
    "<|bos|>",
    "<|user_start|>", "<|user_end|>",
    "<|assistant_start|>", "<|assistant_end|>",
    "<|python_start|>", "<|python_end|>",
    "<|output_start|>", "<|output_end|>",
]
VOCAB_SIZE = 24576          # 24,567 content + 9 special. 24576 = 384*64 -> nanochat's
                            # pad_vocab_size_to=64 is a NO-OP (no padding). Verified in F1.

# --- tokenizer_eval holdout (must match tokenizer/config.py so we exclude the
#     SAME docs the tokenizer held out) --------------------------------------
HOLDOUT_FRAC = 0.015
HOLDOUT_SPLIT_SEED = 20260720

# --- parquet materialization --------------------------------------------------
# Smoke: cap the training text so the Volume upload + on-the-fly tokenization stay
# fast. The full corpus (~450 MB text) is used for the real Workstream-G run
# (`--max-mb -1`). nanochat's dataloader treats the LAST parquet (sorted) as the val
# split, all others as train.
SMOKE_MAX_MB = 60
VAL_MAX_MB = 4
PARQUET_ROW_GROUP = 256      # docs per row group (DDP/resume granularity)
DATA_SEED = 20260721

# --- Modal harness ------------------------------------------------------------
MODAL_APP_NAME = "kreyol-train"
MODAL_GPU = "H100"           # the Workstream-G d12 target GPU (plan §3.2/§7.2)
MODAL_VOLUME = "kreyol-train-cache"
MODAL_BASE_DIR = "/cache/nanochat"     # NANOCHAT_BASE_DIR on the Volume
MODAL_TIMEOUT_S = 60 * 60
MODAL_H100_USD_PER_HR = 3.95           # list price for the $ estimate (dashboard is truth)

# --- throwaway training run (the smoke) --------------------------------------
# depth=12 so the measured tok/s is representative of the real Model-C d12 target.
# Small batch + few steps keep it to a couple of GPU-minutes. window-pattern L (full
# causal) is robust with or without Flash-Attention-3 (SDPA has no sliding-window
# support); G can switch to the default SSSL sliding-window pattern under FA3.
SMOKE = {
    "depth": 12,
    "max_seq_len": 2048,
    "device_batch_size": 32,
    "total_batch_size": 65536,       # 1 grad-accum step at bs32*2048 on 1 GPU
    "window_pattern": "L",
    "warmup_steps": 5,               # short warmup so the loss drop is clean over ~40 steps
    "eval_every": 10,
    "eval_tokens": 131072,
    "core_metric_every": -1,         # skip CORE (English tasks + downloads)
    "sample_every": -1,              # skip English sample prompts; we sample Kreyòl ourselves
    "num_iterations_a": 20,          # call A: 0 -> 20
    "save_steps_a": "3,10",          # irregular saves (proves the log-then-linear schedule)
    "resume_from": 20,               # call B resumes the step-20 checkpoint (fresh container)
    "num_iterations_b": 40,          # call B: 20 -> 40
    "model_tag": "kreyol-d12-smoke",
}

# --- Workstream-G checkpoint schedule (log-then-linear, Pythia-style) ----------
# The token points at which G saves a checkpoint (+ runs the frozen prompt list and
# BPB curves). Converted to STEP indices for --save-steps at G's batch size.
G_CHECKPOINT_TOKENS = [0, 10_000_000, 25_000_000, 50_000_000,
                       100_000_000, 250_000_000, 500_000_000, 1_000_000_000]
G_DEFAULT_TOTAL_BATCH = 524288      # d12 default auto batch (2^19 tokens/step)


def tokens_to_steps(token_points, total_batch_size):
    """Map token checkpoint points -> optimizer step indices (round)."""
    return [round(t / total_batch_size) for t in token_points]


# --- Kreyòl generation probe (proves the trained model emits Kreyòl) ----------
# Short completion seeds (NOT the exhibit prompt list, and NO probe proverbs).
GEN_PROMPTS = [
    "Bonjou, kijan ou ye",
    "Ayiti se yon peyi",
    "Mwen renmen manje",
    "Pitit la ap",
]
GEN_MAX_TOKENS = 32
