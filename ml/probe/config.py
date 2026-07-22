"""Pinned configuration for the Workstream D base-model probe (Phase 0b).

Everything that must be recorded for reproducibility lives here: candidate model
repos, decoding settings, the fixed prompt templates, seeds, the BPB / doc-boundary
policy, and the Modal harness knobs. See ../../docs/phase-0.md Workstream D and
../../docs/plan.md §3.1 for the protocol.

Precision is "unquantized bf16" everywhere (transformers, torch_dtype=bfloat16).
"""

from __future__ import annotations

# The probe snapshot date. Stamped rather than read from the clock so a re-run
# reproduces the same claims. HF model revisions are resolved to exact commit
# SHAs at load time (recorded per-model in the results / report).
SNAPSHOT_DATE = "2026-07-21"

# --- candidate base checkpoints (PRETRAINED bases, never instruct variants) ----
# gated=True => download may 401/403 if the HF account hasn't accepted the gate;
# we skip such a model gracefully and record why. `control` marks the expected
# floor (Llama 3.2: 8 official languages, HT not among them).
CANDIDATES = [
    {"key": "qwen3-1.7b",  "repo": "Qwen/Qwen3-1.7B-Base",        "gated": False},
    {"key": "qwen3-4b",    "repo": "Qwen/Qwen3-4B-Base",          "gated": False},
    {"key": "smollm3-3b",  "repo": "HuggingFaceTB/SmolLM3-3B-Base", "gated": False},
    {"key": "gemma3-4b",   "repo": "google/gemma-3-4b-pt",        "gated": True},
    {"key": "llama3.2-3b", "repo": "meta-llama/Llama-3.2-3B",     "gated": True,
     "control": True},
]

# --- FLORES+ selection corpus: DEV ONLY (devtest is reserved for final report) -
# Pins reused verbatim from Workstream C so the join is identical.
FLORES_REPO = "openlanguagedata/flores_plus"
FLORES_REVISION = "b3a5298db5721c8a682e7ef00a37fcc9ab522757"
FLORES_SPLIT = "dev"                       # <- NOT devtest
FLORES_LANGS = {"ht": "hat_Latn", "en": "eng_Latn", "fr": "fra_Latn"}

# --- staged funnel sizes ------------------------------------------------------
SMOKE_N = 20            # stage 1: catch template/decoding breakage cheaply
EVAL_SUBSET_N = 250     # stage 2: fixed dev subset scored on every model
N_SHOT = 5              # few-shot MT exemplars (disjoint from the eval subset)
# stage 3 (full dev MT) runs on the top TOP_K models by the automated scorecard.
TOP_K = 2
# Gemma-3 loads as the multimodal Gemma3ForConditionalGeneration; its generate
# path is ~20x slower than the plain causal LMs. If a slow model reaches the
# full-dev head-to-head, we bound that stage to a seeded FULLDEV_CAP_SLOW-sentence
# sample (still 2x the selection subset, scored identically for both finalists)
# so the run stays within the $5-10 budget. Both fast -> full dev (992).
SLOW_MODELS = {"gemma3-4b"}
FULLDEV_CAP_SLOW = 500

# Seeds (fixed). One shuffle of the dev ids drives exemplar + subset selection so
# the two sets are provably disjoint and reproducible.
SELECT_SEED = 20260721
NAT_SHUFFLE_SEED = 20260721   # de-identification order for the naturalness sheet
BPB_SUBSAMPLE_SEED = 20260721

# --- decoding (greedy everywhere) ---------------------------------------------
# do_sample=False, num_beams=1 -> deterministic greedy. Stop strings truncate the
# completion; max_new_tokens bounds cost. Recorded verbatim in the report.
GREEDY = {"do_sample": False, "num_beams": 1}
# 128 comfortably covers FLORES sentence lengths (~50-90 target tokens) while
# capping the cost of ramble-to-max stragglers uniformly across models; a batch
# runs until every member emits the stop string or hits this cap.
MT_MAX_NEW = 128
PROVERB_MAX_NEW = 32
NAT_MAX_NEW = 80
MT_STOP = "\n"
PROVERB_STOP = "\n"
NAT_STOP = "\n\n"
GEN_BATCH_SIZE = 16

# --- fixed prompt templates ---------------------------------------------------
# Language names are the English exonyms (base models key on those). The exact
# strings below are the record; do not paraphrase when reporting.
MT_LANG_NAME = {"en": "English", "ht": "Haitian Creole"}


def mt_line(lang_code: str, text: str) -> str:
    return f"{MT_LANG_NAME[lang_code]}: {text}"


def build_mt_prompt(shots, src_code, tgt_code, src_text):
    """5-shot completion prompt. `shots` = list of (src_text, tgt_text) in the
    same direction. The model completes the final `{TgtName}: ` line."""
    blocks = []
    for s_src, s_tgt in shots:
        blocks.append(mt_line(src_code, s_src) + "\n" + mt_line(tgt_code, s_tgt))
    blocks.append(mt_line(src_code, src_text) + "\n" + MT_LANG_NAME[tgt_code] + ":")
    return "\n\n".join(blocks)


# --- BPB (bits-per-byte) policy ----------------------------------------------
# Primary measure. BPB = total_nll_bits / total_utf8_bytes, cross-tokenizer
# comparable because the denominator is bytes, not tokens.
#   * doc boundaries: each holdout doc is scored INDEPENDENTLY (no cross-doc
#     context, so nothing leaks between documents).
#   * BOS: we prepend the tokenizer's BOS id, or its EOS id when no BOS is
#     defined (GPT-2 document-start convention). The start token is context only
#     and is NEVER scored; every real token IS scored.
#   * long docs: split into consecutive windows of BPB_MAX_LEN real tokens, each
#     window re-seeded with the start token. Denominator = the doc's UTF-8 bytes
#     (windows partition the token stream, which detokenizes losslessly).
BPB_MAX_LEN = 2048
# Full holdout slice is ~7 MB; seed-subsample docs to a byte budget for cost.
# The authored-only slice (~0.43 MB) is scored in full.
BPB_FULL_BYTE_BUDGET = 700_000

# --- naturalness mini-review (10 fixed Kreyòl completion prompts) -------------
# Base models don't follow instructions; these are completion seeds spanning
# greeting / simple-question / short-translation / register-shift / continuation.
# A fluent speaker scores the shuffled, de-identified outputs — we DO NOT score.
NATURALNESS_PROMPTS = [
    {"id": "greet_1",   "category": "greeting",
     "prompt": "Bonjou, zanmi m! Jodi a"},
    {"id": "greet_2",   "category": "greeting",
     "prompt": "Alo, koman ou ye? Mwen menm, mwen"},
    {"id": "quest_1",   "category": "simple question",
     "prompt": "Kisa ou renmen manje? Mwen renmen"},
    {"id": "quest_2",   "category": "simple question",
     "prompt": "Poukisa lapli tonbe? Paske"},
    {"id": "trans_1",   "category": "short translation",
     "prompt": 'An kreyòl, "Good morning, how are you?" vle di:'},
    {"id": "trans_2",   "category": "short translation",
     "prompt": 'Fraz sa a: "I love my country very much" vin di an kreyòl:'},
    {"id": "regis_1",   "category": "register shift (formal)",
     "prompt": "Mesyedam, se yon gwo onè pou mwen prezante"},
    {"id": "regis_2",   "category": "register shift (casual)",
     "prompt": "Frè m, kite m di w yon bagay:"},
    {"id": "cont_1",    "category": "continuation (narrative)",
     "prompt": "Te gen yon fwa, yon ti gason yo te rele Ti Jan. Chak maten, li"},
    {"id": "cont_2",    "category": "continuation (expository)",
     "prompt": "Ayiti se yon peyi ki gen yon istwa rich. Kilti li"},
]

# --- Modal harness knobs ------------------------------------------------------
MODAL_APP_NAME = "kreyol-base-probe"
MODAL_GPU = "L40S"           # 48 GB, ~$2/GPU-h; ample for 4B bf16
MODAL_VOLUME = "kreyol-probe-hf-cache"   # persists downloaded weights across runs
MODAL_SECRET = "kreyol-hf"    # Modal secret carrying HF_TOKEN
MODAL_TIMEOUT_S = 60 * 60     # per-call ceiling
# For a $ ESTIMATE only (GPU-seconds x rate). Modal's dashboard is authoritative;
# re-verify the rate on modal.com/pricing before quoting. L40S list ~ $1.95/GPU-h.
MODAL_L40S_USD_PER_HR = 1.95

# --- external comparison (context only, NOT directly comparable) --------------
# Robinson et al. 2023 "ChatGPT MT: Competitive for High- but Not Low-Resource
# Languages" reports eng->hat chrF; different prompts/pipeline/dataset version, so
# we contextualize but never claim comparability.
ROBINSON_2023_NOTE = (
    "Robinson et al. 2023 (arXiv:2309.07423) report eng->hat MT quality for "
    "several systems on FLORES-style data; their numbers use a different "
    "prompt, pipeline, and dataset version and are NOT directly comparable to "
    "these few-shot base-completion scores — cited only for rough orientation."
)
