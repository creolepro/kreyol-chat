"""Workstream H — micro-model fleet configuration.

Identical-twin experiments at ~30M params (the SAME standard-Llama arch as Workstream
G, scaled down), ~200M tokens/run (near-Chinchilla at this size — not confounded by
undertraining). Each question has a DECISION attached (docs/phase-1.md Workstream H).

The micro arch is G's `LlamaForCausalLM` contract at width 384 / depth 6 (verified
29.50M params: 18.87M untied 24k embeddings + 6×1.77M layers). It is embedding-
dominated (64%) — inherent to a 24,576 vocab at width 384 — so BPB deltas between
conditions come almost entirely from the 10.6M non-embedding transformer, which sees
~19 tokens/param at the 200M budget (Chinchilla-ish). Documented, not hidden.

Reuses G's model (`llama_model.build_model`), dataloader (`data_g`), and BPB
(`bpb_g`); only the arch dims, the data variants, and the eval-slice set change.
"""

from __future__ import annotations

import os

from . import config as F          # Workstream-F paths / Modal knobs / VOCAB_SIZE

SNAPSHOT_DATE = "2026-07-25"

# ---------------------------------------------------------------------------
# Micro architecture — SAME contract as G.ARCH, scaled to ~30M params.
# ---------------------------------------------------------------------------
MICRO_ARCH = {
    "hidden_size": 384,
    "intermediate_size": 1024,          # 8/3 * 384 = 1024 (Llama SwiGLU convention)
    "num_attention_heads": 6,           # head_dim = 384/6 = 64
    "num_key_value_heads": 6,           # full MHA
    "hidden_act": "silu",
    "max_position_embeddings": 2048,
    "rope_theta": 10000.0,
    "rms_norm_eps": 1e-5,
    "attention_bias": False,
    "mlp_bias": False,
    "tie_word_embeddings": False,        # untied — matches G exactly (the 24k embedding tax is real)
    "vocab_size": F.VOCAB_SIZE,          # 24,576
}
MICRO_DEPTH = 6                          # -> 29.50M params (verified on Modal, verify_params)

# ---------------------------------------------------------------------------
# Training recipe — uniform across ALL fleet runs so every comparison is fair.
# 2^19 tokens/optimizer step (== G) keeps the token->step map identical.
# ---------------------------------------------------------------------------
TRAIN = {
    "max_seq_len": 2048,
    "device_batch_size": 64,             # 64*2048 = 131,072 tok/microbatch (fp32 logits ~13GB — fits H100)
    "total_batch_size": 524288,          # 2^19 tokens / optimizer step (4 grad-accum microbatches)
    "peak_lr": 3.0e-3,                   # scale-appropriate for ~30M (higher than the 123M flagship's 1.5e-3)
    "min_lr_frac": 0.1,                  # cosine floor
    "warmup_frac": 0.1,                  # warmup = max(20, round(0.1*num_iter)) — PROPORTIONAL so the
                                         # Q5 epoch-stretch runs (191/381/572 steps) each get a clean shape
    "weight_decay": 0.1,
    "adam_beta1": 0.9,
    "adam_beta2": 0.95,
    "grad_clip": 1.0,
    "compile": False,                    # eager — 13 micro model-builds in one warm container
                                         # makes repeated torch.compile a needless risk; a 30M model
                                         # is fast enough eager (confirmed by the smoke). Precheck
                                         # (full-size, 2 runs) keeps G's compile=True.
    "attn_impl": "sdpa",
    "dtype": "bfloat16",
}
TOKENS_PER_STEP = TRAIN["total_batch_size"]     # 524,288
FLEET_TOKENS = 200_000_000                       # ~200M tokens/run (near-Chinchilla)
FLEET_STEPS = round(FLEET_TOKENS / TOKENS_PER_STEP)   # 381 -> 199.8M

# seeds vary weight init AND the dataloader's random-offset sampling; the .bin doc
# ORDER is fixed per variant (DOC_SHUFFLE_SEED) so seed spread isolates init+sampling
# variance, and Q1's two tokenizers share doc order (only the vocabulary differs).
SEEDS = {"A": 20260725, "B": 20260726}
DOC_SHUFFLE_SEED = 20260725
Q5_SUBSAMPLE_TOKENS = 25_000_000                 # fixed unique subsample; 4/8/12x -> 100/200/300M

# ---------------------------------------------------------------------------
# Tokenizers (both are 24,576-vocab, kreyol_aware pattern; only the merges differ).
# ---------------------------------------------------------------------------
TOKENIZERS = {
    "kreyol-bpe":  os.path.join(F.REPO_ROOT, "tokenizer", "kreyol-bpe", "tokenizer.pkl"),
    "english-24k": os.path.join(F.REPO_ROOT, "tokenizer", "english-24k", "tokenizer.pkl"),
}

# ---------------------------------------------------------------------------
# Corpus shards (git-ignored under data/clean/).
# ---------------------------------------------------------------------------
CORPUS_SHARDS = {
    "v0":     os.path.join(F.DATA, "clean", "corpus_v0-full.jsonl"),
    "v0.1":   os.path.join(F.DATA, "clean", "corpus_v0_1-full.jsonl"),
    "v0.2.1": os.path.join(F.DATA, "clean", "corpus_v0_2_1-full.jsonl"),
}

# register mix weights for the "weighted" sampling variant (Q2). Copied from
# corpus/config_v0_2.MIX_WEIGHTS — the exact weights G-v1 would sample with.
MIX_WEIGHTS = {
    "journalism": 4.0, "government": 3.0, "financial": 3.0, "children": 3.0,
    "legal": 3.0, "encyclopedic": 2.0, "religious": 1.0, "web_crawl": 1.0,
    "news_translated": 1.0, "health_translated": 1.0, "historical_literary": 1.0,
    "historical_pedagogy": 1.0, "tax": 1.0, "immigration": 1.0, "health": 1.0,
    "disaster": 1.0, "proverb": 1.0,
}

# ---------------------------------------------------------------------------
# Data variants — (corpus, tokenizer, sampling) recipes the tokenizer step builds.
# ---------------------------------------------------------------------------
VARIANTS = {
    "v0_kbpe":            {"corpus": "v0",     "tok": "kreyol-bpe",  "sampling": "natural"},
    "v01_kbpe":           {"corpus": "v0.1",   "tok": "kreyol-bpe",  "sampling": "natural"},
    "v01_nostub_kbpe":    {"corpus": "v0.1",   "tok": "kreyol-bpe",  "sampling": "natural",
                           "drop_bot_stubs": True},
    "v021_kbpe":          {"corpus": "v0.2.1", "tok": "kreyol-bpe",  "sampling": "natural"},
    "v021_weighted_kbpe": {"corpus": "v0.2.1", "tok": "kreyol-bpe",  "sampling": "weighted"},
    "v021_en24k":         {"corpus": "v0.2.1", "tok": "english-24k", "sampling": "natural"},
    "q5sub_kbpe":         {"corpus": "v0.2.1", "tok": "kreyol-bpe",  "sampling": "natural",
                           "subsample_tokens": Q5_SUBSAMPLE_TOKENS},
}

# ---------------------------------------------------------------------------
# The 13 canonical runs (deduplicated — shared runs train once, e.g. v021_kbpe.sA
# feeds Q1, Q2 and Q7; v01_kbpe.sA feeds Q3, Q4 and Q7).
# Each: {variant, seed_key, num_iterations}. Q5 step counts filled at prep time from
# the measured subsample size (kept here as epoch multiples).
# ---------------------------------------------------------------------------
def _r(variant, seed, steps):
    return {"variant": variant, "seed": seed, "num_iterations": steps}

RUNS = {
    "v021_kbpe.sA":     _r("v021_kbpe", "A", FLEET_STEPS),
    "v021_kbpe.sB":     _r("v021_kbpe", "B", FLEET_STEPS),
    "v021_en24k.sA":    _r("v021_en24k", "A", FLEET_STEPS),
    "v021_en24k.sB":    _r("v021_en24k", "B", FLEET_STEPS),
    "v021_weighted.sA": _r("v021_weighted_kbpe", "A", FLEET_STEPS),
    "v021_weighted.sB": _r("v021_weighted_kbpe", "B", FLEET_STEPS),
    "v01_kbpe.sA":      _r("v01_kbpe", "A", FLEET_STEPS),
    "v01_kbpe.sB":      _r("v01_kbpe", "B", FLEET_STEPS),
    "v0_kbpe.sA":       _r("v0_kbpe", "A", FLEET_STEPS),
    "v01_nostub.sA":    _r("v01_nostub_kbpe", "A", FLEET_STEPS),
    # Q5 epoch stretch — steps set at prep time (epoch multiples of the subsample).
    "q5sub.e4.sA":      _r("q5sub_kbpe", "A", None),
    "q5sub.e8.sA":      _r("q5sub_kbpe", "A", None),
    "q5sub.e12.sA":     _r("q5sub_kbpe", "A", None),
}
Q5_EPOCH_RUNS = {"q5sub.e4.sA": 4, "q5sub.e8.sA": 8, "q5sub.e12.sA": 12}

# ---------------------------------------------------------------------------
# The questions — each names the runs it pairs, what varies, and the decision.
# ---------------------------------------------------------------------------
QUESTIONS = {
    "Q1": {
        "title": "Tokenizer — does the Kreyòl vocabulary improve LEARNING (BPB), not just cost?",
        "varies": "tokenizer (kreyol-bpe vs english-24k); identical v0.2.1 data/order/steps",
        "runs": {"kreyol-bpe": ["v021_kbpe.sA", "v021_kbpe.sB"],
                 "english-24k": ["v021_en24k.sA", "v021_en24k.sB"]},
        "headline_slice": "general_holdout",
        "seeds": 2,
        "decides": "Station 1's tokenizer-thesis claim strength (causal, not just fertility)",
    },
    "Q2": {
        "title": "Mix weights — does authored-upweighting shift the model's voice?",
        "varies": "sampling (natural vs config_v0_2.MIX_WEIGHTS-weighted) on v0.2.1",
        "runs": {"natural": ["v021_kbpe.sA", "v021_kbpe.sB"],
                 "weighted": ["v021_weighted.sA", "v021_weighted.sB"]},
        "headline_slice": "authored_eval",
        "seeds": 2,
        "decides": "G-v1's sampling config + late-curriculum design",
    },
    "Q7": {
        "title": "Corpus v0.1 vs v0.2.1 — does fineweb-2's bulk help or dilute?",
        "varies": "corpus (v0.1 vs v0.2.1); same tokenizer, fixed compute (381 steps)",
        "runs": {"v0.1": ["v01_kbpe.sA", "v01_kbpe.sB"],
                 "v0.2.1": ["v021_kbpe.sA", "v021_kbpe.sB"]},
        "headline_slice": "general_holdout",
        "seeds": 2,
        "decides": "whether G-v1 trains on v0.2.1 at all (biggest decision in the fleet)",
    },
    "Q3": {
        "title": "Junk filtering — does v0.1's de-junk win at fixed compute?",
        "varies": "corpus (v0 vs v0.1); same tokenizer, fixed compute",
        "runs": {"v0": ["v0_kbpe.sA"], "v0.1": ["v01_kbpe.sA"]},
        "headline_slice": "general_holdout",
        "seeds": 1,
        "decides": "how aggressive E's future filters should get",
    },
    "Q4": {
        "title": "Bot-stubs — food or filler?",
        "varies": "wikipedia bot-stubs in vs out (v0.1)",
        "runs": {"stubs_in": ["v01_kbpe.sA"], "stubs_out": ["v01_nostub.sA"]},
        "headline_slice": "authored_eval",
        "seeds": 1,
        "decides": "corpus policy v0.3",
    },
    "Q5": {
        "title": "Epoch stretch — how far does repetition stretch our corpus?",
        "varies": "4 vs 8 vs 12 epochs of a fixed 25M-token unique subsample",
        "runs": {"4ep": ["q5sub.e4.sA"], "8ep": ["q5sub.e8.sA"], "12ep": ["q5sub.e12.sA"]},
        "headline_slice": "general_holdout",
        "seeds": 1,
        "decides": "G-v1's token schedule; re-validates the Muennighoff repetition band at our scale",
    },
}

# ---------------------------------------------------------------------------
# Eval slices (BPB, byte-normalized). FIXED for every run so BPB is comparable
# across corpora AND across tokenizers. FLORES hat is measurement-only.
# ---------------------------------------------------------------------------
AUTHORED_EVAL_V2 = os.path.join(F.EVAL_DIR, "authored_eval_v2.jsonl")
EVAL_SLICES = ["general_holdout", "authored_eval", "translation_shaped_eval",
               "authored_eval_v2", "flores_hat"]
GENERAL_HOLDOUT_BYTES = 700_000                  # match G / the base-model scorecard budget

# ---------------------------------------------------------------------------
# Part 3 — flagship depth pre-check (NOT fleet-scale): full-size d12 vs d16 on
# v0.2.1, ~175M tokens each (in G's 150-200M sweep band), G's EXACT recipe/seed.
# ---------------------------------------------------------------------------
DEPTH_PRECHECK = {
    "depths": [12, 16],
    "num_iterations": 334,               # 334 * 524,288 = 175.1M tokens (150-200M band)
    "corpus_variant": "v021_kbpe",       # v0.2.1, kreyol-bpe, natural
    "model_tag": "precheck-v021-d{depth}",
    "seed": 20260722,                    # G's depth-sweep seed (same protocol)
}

# ---------------------------------------------------------------------------
# Modal Volume layout for the fleet (separate from G's /cache/g).
# ---------------------------------------------------------------------------
FLEET_DIR = "/cache/fleet"
FLEET_DATA_DIR = FLEET_DIR + "/data"             # {variant}.bin + eval_texts.json + manifests
FLEET_TOK_DIR = FLEET_DIR + "/tokenizers"        # {kreyol-bpe,english-24k}.pkl
FLEET_RESULTS = FLEET_DIR + "/results"           # self-persisted per-run result JSONs

# local scratch (git-ignored)
FLEET_WORK = os.path.join(F.DATA, "train_work", "fleet")
FLEET_BUNDLE = os.path.join(FLEET_WORK, "data")  # bins staged before upload
