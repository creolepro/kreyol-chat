"""Workstream I configuration — midtraining + SFT → Model C chat.

The three-layer SFT data stack (docs/data.md §3, binding):
  Layer 1 — midtraining (format-teaching bulk): kakugo-hat (cleaned) + a capped,
            deduped minority of aya_collection/xP3x + translation turns from the
            PD glossary + CMU lexicon. FULL-sequence LM loss.
  Layer 2 — corpus-grounded generation (the quality core): MURI-style reverse
            instructions + doc→dialogue over AUTHORED corpus passages, teacher =
            claude-opus-4-8. PILOT-GATED (Part 2). Trained in the SFT stage.
  Layer 3 — SFT cap (small + excellent): muri-it hat + aya gold + best Layer-2 +
            glossary/lexicon QA. Loss MASKED to assistant responses.

Format: the nanochat chat template over kreyol-bpe's reserved special tokens
(<|user_start|> … already in the 24,576 vocab). BOS is NOT in the template — the
Part-0 add_bos_token fix makes the runtime prepend <|bos|> (24567), matching training.

Reuses the Workstream-F/G config (paths, tokenizer bundle, Modal knobs, the v1 base
checkpoint). Nothing under ml/data/ is committed.
"""

from __future__ import annotations

import os

from . import config as F
from . import llama_config as G

SNAPSHOT_DATE = "2026-07-26"

# --- the Model C v1 base checkpoint the chat model continues from --------------
V1_BASE_TAG = G.FLAGSHIP_V1["model_tag"].format(depth=12)   # modelc-v1-d12
V1_BASE_STEP = G.FLAGSHIP_V1["num_iterations"]              # 1907
DEPTH = 12

# --- reserved special tokens (subset of F.SPECIAL_TOKENS actually used here) ---
BOS = "<|bos|>"
USER_START, USER_END = "<|user_start|>", "<|user_end|>"
ASSISTANT_START, ASSISTANT_END = "<|assistant_start|>", "<|assistant_end|>"

# --- the nanochat chat template (deployment: HF apply_chat_template + GGUF KV) --
# NO leading <|bos|>: the Part-0 add_bos_token=true makes the runtime prepend it, so
# the template must not (else double-BOS). user AND system roles → user_start block;
# assistant → assistant_start block. add_generation_prompt opens an assistant turn.
CHAT_TEMPLATE = (
    "{%- for message in messages -%}"
    "{%- if message['role'] == 'assistant' -%}"
    "<|assistant_start|>{{ message['content'] }}<|assistant_end|>"
    "{%- else -%}"
    "<|user_start|>{{ message['content'] }}<|user_end|>"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}<|assistant_start|>{%- endif -%}"
)

# --- local layout (git-ignored) ------------------------------------------------
CHAT_RAW = os.path.join(F.DATA, "interim", "chat")        # per-source cleaned jsonl
CHAT_WORK = os.path.join(F.DATA, "train_work", "chat")    # bins + manifests + results
CHAT_BUNDLE = os.path.join(CHAT_WORK, "data")             # midtrain/sft bins uploaded to Volume

# per-source cleaned conversation files (unified schema: see chat_data)
KAKUGO_CLEAN = os.path.join(CHAT_RAW, "kakugo_clean.jsonl")
AYA_COLLECTION_CLEAN = os.path.join(CHAT_RAW, "aya_collection_clean.jsonl")
XP3X_CLEAN = os.path.join(CHAT_RAW, "xp3x_clean.jsonl")
TRANSLATION_TURNS = os.path.join(CHAT_RAW, "translation_turns.jsonl")
MURI_IT_HAT = os.path.join(CHAT_RAW, "muri_it_hat.jsonl")
AYA_GOLD = os.path.join(CHAT_RAW, "aya_gold.jsonl")
LAYER2_GEN = os.path.join(CHAT_RAW, "layer2_generated.jsonl")   # Part 2 output
LAYER2_PILOT = os.path.join(CHAT_RAW, "layer2_pilot.jsonl")     # Part 2 pilot output

# assembled layers (what the tokenizer packs)
LAYER1_JSONL = os.path.join(CHAT_RAW, "layer1_midtrain.jsonl")
LAYER3_JSONL = os.path.join(CHAT_RAW, "layer3_sft.jsonl")

# --- HF dataset specs ----------------------------------------------------------
DS_KAKUGO = {"repo": "Kreyol/kakugo-hat"}
DS_MURI_IT = {"repo": "akoksal/muri-it", "config": "default", "lang_field": "language", "lang": "hat"}
DS_AYA_DATASET = {"repo": "CohereForAI/aya_dataset", "lang_field": "language_code", "lang": "hat"}
DS_AYA_COLLECTION = {"repo": "CohereForAI/aya_collection_language_split", "config": "haitian"}
DS_XP3X = {"repo": "CohereForAI/xP3x", "config": "hat_Latn"}

# --- filtering / cleaning knobs ------------------------------------------------
MIN_TURN_CHARS = 2               # drop empty/near-empty turns
MIN_CONTENT_CHARS = 8            # a message shorter than this is degenerate
LANGID_DROP_CONF = 0.60          # drop a Kreyòl-expected turn only if clearly foreign at ≥ this
# fastText labels that mean "not Kreyòl and not tolerable code-switch". English is
# tolerated in USER turns only up to a point (many real prompts mix in an English term);
# a clearly-foreign ASSISTANT turn (the Kreyòl output) is the disqualifier.
FOREIGN_LANGS = {"es", "pt", "it", "de", "nl", "ca", "gl", "ro", "id", "af", "sw",
                 "so", "ru", "tr", "vi", "tl", "ceb", "war", "fr", "en"}
# eval carve-out: never let FLORES (our standing MT eval) leak into training
FLORES_MARKERS = ("flores", "facebook/flores", "openlanguagedata")
EVAL_SPLITS = {"dev", "devtest", "validation", "valid", "test"}

# caps (Layer-1 templated bulk must stay a MINORITY next to kakugo; format variety only)
AYA_COLLECTION_CAP = 6000        # after dedup + FLORES drop + train-split filter
XP3X_CAP = 4000                  # after the hard FLORES carve-out
TRANSLATION_TURNS_CAP = 4000     # glossary (1,955) + a capped CMU-lexicon sample
CMU_LEXICON_SAMPLE = 2000        # capped sample of the 32,231-pair lexicon (longer entries first)
NEAR_DUP_KEY_CHARS = 160         # normalized prefix length for cheap near-dup dedup

DL_SEED = 20260726

# --- midtraining (Layer 1+2), continued LM training from the v1 base -----------
# Small next to pretraining (real-data-majority anchor). Full-sequence loss teaches the
# chat FORMAT + keeps the language model. Finer step size (2^17) since the set is small.
MIDTRAIN = {
    "model_tag": "modelc-chat-mid-d12",
    "resume_tag": V1_BASE_TAG, "resume_step": V1_BASE_STEP,
    "epochs": 3.0,
    "total_batch_size": 131072,      # 2^17 tokens / optimizer step
    "device_batch_size": 16,
    "max_seq_len": 2048,
    "peak_lr": 3.0e-4, "min_lr_frac": 0.1, "warmup_frac": 0.03,
    "weight_decay": 0.1, "adam_beta1": 0.9, "adam_beta2": 0.95, "grad_clip": 1.0,
    "loss_mask": "full",             # LM loss over the whole formatted conversation
    "seed": 20260726,
}

# --- SFT (Layer 3 + best Layer-2), response-masked, from the midtrained ckpt ----
SFT = {
    "model_tag": "modelc-chat-sft-d12",
    "resume_tag": MIDTRAIN["model_tag"],   # resume step filled in at run time (mid final)
    "epochs": 3.0,
    "total_batch_size": 65536,       # 2^16 tokens / optimizer step
    "device_batch_size": 16,
    "max_seq_len": 2048,
    "peak_lr": 1.0e-4, "min_lr_frac": 0.1, "warmup_frac": 0.05,
    "weight_decay": 0.1, "adam_beta1": 0.9, "adam_beta2": 0.95, "grad_clip": 1.0,
    "loss_mask": "response",         # loss ONLY on assistant-response tokens + <|assistant_end|>
    "seed": 20260727,
}

# --- Modal Volume layout for the chat run --------------------------------------
CHAT_DIR = "/cache/chat"
CHAT_DATA_DIR = CHAT_DIR + "/data"       # midtrain.{bin,mask.bin}, sft.{bin,mask.bin}, eval
CHAT_CKPT_DIR = G.G_CKPT_DIR             # reuse G's checkpoint dir (same Volume tree)
CHAT_ARTIFACT_DIR = CHAT_DIR + "/artifacts"

# --- chat-mode eval ------------------------------------------------------------
CHAT_GEN_MAX_TOKENS = 200        # frozen prompts answered in chat mode
CHAT_EXHIBIT_TEMP = 0.7
CHAT_EXHIBIT_TOP_P = 0.95
NATURALNESS_SHEET_N = 30         # blinded chat outputs for the 2nd native review
