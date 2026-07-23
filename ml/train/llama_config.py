"""Pinned configuration for Workstream G — Model C as a STANDARD Llama architecture.

Per the Architecture Decision block (docs/phase-1.md Workstream G, 2026-07-22, binding):
Model C trains as a standard, explicitly-specified Llama architecture with HF
`LlamaForCausalLM` semantics as the canonical model contract — learned RMSNorm weights,
standard RoPE, standard causal attention, SwiGLU MLP, ordinary residuals — and NONE of
the pinned nanochat speedrun features (value embeddings, smear, backout, per-layer
lambdas, QK multiplier, logit softcap, custom sliding-window). This is a model-
implementation swap, not a config flag: we train the actual HF `LlamaForCausalLM`, so
export (`save_pretrained`) is zero-divergence and the F2 conversion gates become
tractable.

Width is fixed at 768; depth is parameterized (12/16/20) and chosen by the Part-3
mini-sweep. kreyol-bpe (24,576, `kreyol_aware` pattern) is used end-to-end.

Reuses paths / tokenizer-bundle logic from the Workstream-F config (`train/config.py`);
the F modules stay intact and reproducible.
"""

from __future__ import annotations

import os

from . import config as F  # Workstream-F config: paths, tokenizer bundle, Modal knobs

SNAPSHOT_DATE = "2026-07-22"

# --- standard Llama architecture (width 768; HF LlamaForCausalLM contract) -----
# Param counts (verified on Modal in Part 1): untied embeddings 2 * 24576*768 = 37.75M
# plus per-layer 7.08M (attn 4*768^2 = 2.36M + SwiGLU 3*768*2048 = 4.72M + 2 norms).
#   d12 -> ~123M, d16 -> ~151M, d20 -> ~179M  (matches the decision block exactly).
ARCH = {
    "hidden_size": 768,
    "intermediate_size": 2048,          # 8/3 * 768 = 2048 (Llama SwiGLU convention)
    "num_attention_heads": 6,           # head_dim = 768/6 = 128
    "num_key_value_heads": 6,           # full MHA (GQA allowed by the block, not needed at 123M)
    "hidden_act": "silu",               # SwiGLU
    "max_position_embeddings": 2048,
    "rope_theta": 10000.0,              # standard RoPE base (2048 ctx; no length-extrapolation need)
    "rms_norm_eps": 1e-5,
    "attention_bias": False,
    "mlp_bias": False,
    "tie_word_embeddings": False,       # untied (matches the 37.75M embedding mass in the count)
    "vocab_size": F.VOCAB_SIZE,         # 24,576 (24,567 content + 9 special)
}
DEPTHS = [12, 16, 20]

# --- optimizer / schedule (shared across the sweep AND the flagship, so sweep runs
#     are a fair comparison and the flagship inherits the proven recipe) ----------
TRAIN = {
    "max_seq_len": 2048,
    "device_batch_size": 32,            # 32 * 2048 = 65,536 tokens / microbatch
    "total_batch_size": 524288,         # 2^19 tokens / optimizer step (8 grad-accum microbatches)
    "peak_lr": 1.5e-3,                  # AdamW; safe/standard for ~150M at this token budget
    "min_lr_frac": 0.1,                 # cosine floor = 10% of peak
    "warmup_steps": 100,
    "weight_decay": 0.1,
    "adam_beta1": 0.9,
    "adam_beta2": 0.95,
    "grad_clip": 1.0,
    "seed": 20260722,                   # data order + init; identical across depth-sweep runs
    "compile": True,                    # torch.compile the model for throughput
    "attn_impl": "sdpa",                # PyTorch SDPA (H100 flash kernels); no flash-attn build
    "dtype": "bfloat16",
}

# One optimizer step = TRAIN["total_batch_size"] tokens. This makes the token->step
# mapping exact and identical to the Workstream-F schedule table.
TOKENS_PER_STEP = TRAIN["total_batch_size"]      # 524,288


def tokens_to_steps(token_points):
    return [round(t / TOKENS_PER_STEP) for t in token_points]


# --- Part 2: F2 conversion-gate throwaway (d16, ~$1) --------------------------
GATE = {
    "depth": 16,
    "num_iterations": 60,               # ~31M tokens — enough for non-random Kreyòl generation
    "model_tag": "modelc-d16-gate",
}

# --- Part 3: depth mini-sweep (d12/d16/d20, identical data order + budget) ------
SWEEP = {
    "depths": [12, 16, 20],
    "num_iterations": 300,              # 300 * 524,288 = ~157M tokens each (in the 150-200M band)
    "model_tag": "modelc-sweep-d{depth}",
}

# --- Part 4: flagship (chosen depth; multi-epoch toward 0.6-1.2B effective) -----
FLAGSHIP = {
    "num_iterations": 1431,             # 1431 * 524,288 = ~750M effective tokens (ceiling; may stop earlier)
    "model_tag": "modelc-v0-d{depth}",
    # log-then-linear (Pythia-style) checkpoint token points; -> steps via tokens_to_steps
    "checkpoint_tokens": [0, 10_000_000, 25_000_000, 50_000_000,
                          100_000_000, 250_000_000, 500_000_000, 750_000_000],
}

# --- pinned llama.cpp (same commit the Workstream-F convert probe used) ---------
# The F2 gate registers our `kreyol_aware` pre-tokenizer against THIS source tree.
LLAMA_CPP_REPO = "https://github.com/ggml-org/llama.cpp.git"
LLAMA_CPP_COMMIT = "67b9b0e7f6ce45d929a4411907d3c48ec719e81c"  # same tree the F convert probe used
PRETOKENIZER_NAME = "kreyol-bpe"                   # the name registered in llama.cpp source

# --- Ollama (stock, for gate 3) ------------------------------------------------
OLLAMA_INSTALL = "https://ollama.com/install.sh"

# --- eval slices for BPB (Part 4 learning curves) ------------------------------
# authored_eval / translation_shaped_eval come from Workstream E; FLORES hat devtest
# is measurement-only (the final_devtest "measurement, not training" carve-out; the
# task lists it explicitly). general holdout = a seeded sample of the tokenizer_eval
# holdout (the SAME docs the base-model scorecard used, so the vs-bases table lines up).
FLORES_HAT_DEVTEST = os.path.join(F.DATA, "raw", "petrov", "hat_Latn.devtest")
BPB_GENERAL_HOLDOUT_BYTES = 700_000                # match base_model_probe.md's ~700kB budget

# --- Modal Volume layout for Workstream G --------------------------------------
G_DIR = "/cache/g"                                 # on the Modal Volume
G_DATA_DIR = G_DIR + "/data"                       # train.bin / val.bin / eval_texts.json / parity_probe.json
G_TOKENIZER_DIR = G_DIR + "/tokenizer"             # tokenizer.pkl (tiktoken) + tokenizer.json (HF)
G_CKPT_DIR = G_DIR + "/checkpoints"                # HF checkpoint dirs per (tag, step)
G_ARTIFACT_DIR = G_DIR + "/artifacts"              # GGUF / ONNX / gate outputs
PARITY_PROBE = G_DATA_DIR + "/parity_probe.json"   # ~1k-line source-mixed probe + fixtures

# --- local scratch (git-ignored) ----------------------------------------------
G_WORK = os.path.join(F.DATA, "train_work", "g")   # bins + manifests before upload
G_BUNDLE_DATA = os.path.join(G_WORK, "data")

# --- frozen exhibit prompt list (Part 4 per-checkpoint generations) ------------
# The FROZEN 10-prompt list (corpus/checkpoint_prompts.json) — used verbatim, never
# edited. Loaded at generation time only; probe proverb #31 enters ONLY here.
# CHECKPOINT_PROMPTS = local source (uploaded); G_CHECKPOINT_PROMPTS = the Volume copy the
# Modal functions read (the train package mounts at /root/train, so REPO_ROOT-relative
# sibling dirs like corpus/ are NOT present in the container).
CHECKPOINT_PROMPTS = os.path.join(F.REPO_ROOT, "corpus", "checkpoint_prompts.json")
G_CHECKPOINT_PROMPTS = G_DIR + "/checkpoint_prompts.json"
GEN_MAX_TOKENS = 48

# --- apostrophe / clitic parity fixtures (gate 4) ------------------------------
# Kreyòl clitics + accents + straight AND curly apostrophes + punctuation — the exact
# cases where a wrong pre-tokenizer (GPT-2 fallback) mis-splits vs `kreyol_aware`.
PARITY_FIXTURES = [
    "m'te di w sa",            # straight apostrophe clitic (m' = mwen)
    "m’te di w sa",       # curly apostrophe (U+2019) — same phrase
    "n'ta renmen ale",         # n' clitic, straight
    "n’ta renmen ale",    # n' clitic, curly
    "l'ap vini kounye a",      # l' clitic
    "Dèyè mòn gen mòn.",       # è / ò accents + period
    "Ki jan ou rele? Mwen byen!",  # ? and ! punctuation
    "Ann ale — se lè a!",      # em dash + accent
    "1804: Ayiti te vin endepandan.",  # digits + colon
    "«Bonjou», li di.",        # guillemets + comma
]
