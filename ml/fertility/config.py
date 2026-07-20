"""Pinned configuration for the Workstream C fertility measurement.

Everything that must be recorded for reproducibility lives here: dataset and
tokenizer revisions, the Claude model id, the core Kreyòl word list, and the
date-stamped (operator-supplied) API price snapshot. See ../../docs/phase-0.md
Workstream C and ../../docs/plan.md §3.3 for the protocol.
"""

# The measurement snapshot date. Passed explicitly rather than read from the
# clock so a re-run reproduces the same stamped claims/prices.
SNAPSHOT_DATE = "2026-07-19"

# --- FLORES+ (our own measurement corpus; eval-only, never re-hosted) ---------
FLORES_REPO = "openlanguagedata/flores_plus"
# Pinned dataset revision (commit sha resolved 2026-07-19). Recorded in the report.
FLORES_REVISION = "b3a5298db5721c8a682e7ef00a37fcc9ab522757"
FLORES_SPLIT = "devtest"
# canonical short code -> FLORES+ language file stem
FLORES_LANGS = {"ht": "hat_Latn", "en": "eng_Latn", "fr": "fra_Latn"}

# --- Petrov et al. 2023 released data (pipeline-validation ground truth) -------
# github.com/aleksandarpetrov/tokenization-fairness, pinned commit.
PETROV_COMMIT = "365c3f85bdb302fdcf6d389f5f1651e4ef041741"
PETROV_RAW_BASE = (
    "https://raw.githubusercontent.com/aleksandarpetrov/"
    f"tokenization-fairness/{PETROV_COMMIT}/"
)
# His per-language token totals (rows=languages, cols=tokenizers).
PETROV_LENGTHS_CSV = "assets/tokenization_lengths.csv"
# His FLORES-200 devtest sentences (what we re-tokenize with our own code).
PETROV_DEVTEST = {
    "ht": "flores200_dataset/devtest/hat_Latn.devtest",
    "en": "flores200_dataset/devtest/eng_Latn.devtest",
    "fr": "flores200_dataset/devtest/fra_Latn.devtest",
}
PETROV_CSV_LANG = {"ht": "Haitian Creole", "en": "English", "fr": "French"}
PETROV_CL100K_COL = "cl100k_base"
# Gate: our recomputed ht/en cl100k parity on his sentences must land here.
PETROV_TARGET = 1.739  # his 91870/52835
PETROV_TOLERANCE = 0.03

# --- tiktoken encodings (labeled as tokenizers, not products) -----------------
# (encoding_name, display_label)
TIKTOKEN_ENCODINGS = [
    ("cl100k_base", "cl100k (GPT-4 / GPT-3.5-era)"),
    ("o200k_base", "o200k (GPT-4o-era)"),
]

# --- HuggingFace tokenizers (revisions resolved + recorded at run time) --------
# (repo_id, display_label)
HF_TOKENIZERS = [
    ("google/gemma-3-4b-pt", "Gemma-3 (google/gemma-3-4b-pt)"),
    ("Qwen/Qwen3-1.7B", "Qwen3 (Qwen/Qwen3-1.7B)"),
    ("facebook/nllb-200-distilled-600M", "NLLB (facebook/nllb-200-distilled-600M)"),
    ("HuggingFaceTB/SmolLM3-3B", "SmolLM3 (HuggingFaceTB/SmolLM3-3B)"),
    ("meta-llama/Llama-3.2-3B", "Llama-3 (meta-llama/Llama-3.2-3B)"),
]

# Our own Kreyòl BPE tokenizer (Workstream B output). Included automatically if
# present; absent for now (Workstream B not yet run).
OUR_TOKENIZER_PATH = "tokenizer/tokenizer.json"
OUR_TOKENIZER_LABEL = "kreyol-bpe (ours, Workstream B)"

# --- Claude API measurement (separately labeled; NOT a tokenizer count) -------
CLAUDE_MODEL = "claude-opus-4-8"
CLAUDE_BATCH_SIZE = 200  # sentences per count_tokens message
CLAUDE_LABEL = f"claude_api_input_parity ({CLAUDE_MODEL})"
CLAUDE_OVERHEAD_CONTENT = "."  # minimal message; its count is subtracted per call

# --- Core Kreyòl word list for whole-word survival (from B3) -------------------
# TMA markers, pronouns/clitics, determiners, negation, high-frequency function
# words. Each checked bare AND with a leading space; single-token == survives.
CORE_WORDS = [
    "te", "ta", "ap", "pral", "va",           # TMA markers
    "mwen", "m", "ou", "w", "li", "l",         # pronouns + clitic forms
    "nou", "n", "yo", "y",
    "pa",                                       # negation
    "la", "a", "an", "lan", "nan",             # determiners / postpositions
    "sa", "ki", "gen", "fè",                    # high-frequency
]

# --- Statistics knobs ---------------------------------------------------------
BOOTSTRAP_REPS = 2000       # >= 1000 required
BOOTSTRAP_SEED = 20260719   # fixed for reproducibility
CONTEXT_BUDGET = 8192       # "sentences per 8k budget"

# --- Date-stamped $ premium snapshot (OPERATOR-SUPPLIED — VERIFY) -------------
# Only tokenizers tied to an actually-priced API get a $ premium. These are
# illustrative published list prices as configured on SNAPSHOT_DATE and MUST be
# re-verified against each provider's pricing page before being quoted; prices
# drift. The % premium (parity - 1) is exact and needs no price.
PRICE_DATE = SNAPSHOT_DATE
API_INPUT_PRICES_USD_PER_MTOK = {
    # display_label : (usd_per_million_input_tokens, source_note)
    "cl100k (GPT-4 / GPT-3.5-era)": (
        10.00, "openai.com/api/pricing — gpt-4-turbo input; ILLUSTRATIVE, VERIFY"),
    "o200k (GPT-4o-era)": (
        2.50, "openai.com/api/pricing — gpt-4o input; ILLUSTRATIVE, VERIFY"),
    CLAUDE_LABEL: (
        15.00, "anthropic.com/pricing — Opus input; ILLUSTRATIVE, VERIFY"),
}
