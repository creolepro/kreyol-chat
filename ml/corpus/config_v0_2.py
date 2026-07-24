"""Pinned configuration for corpus v0.2 (Workstream J).

Kept SEPARATE from config.py / config_v0_1.py so corpus v0 and v0.1 stay frozen
and reproducible. v0.2 is a NEW layer: v0.1 shard + net-new J sources, all
register-tagged, cross-source-deduped. Authored emphasis is a TRAIN-MIX knob
(MIX_WEIGHTS below), not a corpus-composition claim (J0 chose "control via mix").
"""

from __future__ import annotations

import os

from . import config

# --- data locations (all git-ignored under ml/data/) -----------------------
VOA_DIR = os.path.join(config.DATA, "voa")                 # raw html + extracted
VOA_STATE = os.path.join(VOA_DIR, "crawl_state.json")      # resumable checkpoint
VOA_ARTICLES = os.path.join(VOA_DIR, "articles.jsonl")     # extracted, PD-clean
FAMILY_SRC = "/Users/patricedouge/Desktop/haitian-creole-data"
FAMILY_DIR = os.path.join(config.DATA, "local", "family-v1")
FEDERAL_DIR = os.path.join(config.DATA, "federal")
SMALLWINS_DIR = os.path.join(config.DATA, "smallwins")
V0_2_INGEST = os.path.join(config.DATA, "interim", "v0_2_ingest")   # per-source jsonl
CORPUS_V0_2 = os.path.join(config.CLEAN, "corpus_v0_2-{tag}.jsonl")
V0_2_STATS = os.path.join(config.CLEAN, "corpus_v0_2-{tag}.build_stats.json")

# --- dedup survivor priority (extends config.SURVIVOR_PRIORITY) -------------
# Lower wins. authored/PD material beats the crawl copy of the same text so the
# well-provenanced version survives (e.g. VOA article vs its MADLAD scrape).
SURVIVOR_PRIORITY_V0_2 = {"owned": 0, "authored": 1, "wikipedia": 2,
                          "crawl": 3, "eval": 9}

# --- per-source origin / genre / register tags -----------------------------
# origin/genre are schema.py enums; register is a free tag used for mix control
# and the nutrition label. (VOA 2026 articles are re-routed to authored_eval_v2.)
SOURCE_TAGS = {
    "voa_nouvel":        {"origin": "authored_kreyol",      "genre": "news",         "register": "journalism"},
    "fineweb2_hat":      {"origin": "web_crawl",            "genre": "web",          "register": "web_crawl"},
    "us_federal_pdfs":   {"origin": "human_translation",    "genre": "educational",  "register": "government"},
    "cfpb_glossary_family": {"origin": "human_translation", "genre": "dictionary",   "register": "financial"},
    "bib_la_1985":       {"origin": "human_translation",    "genre": "religious",    "register": "religious"},
    "konstitisyon_1987": {"origin": "human_translation",    "genre": "historical",   "register": "legal"},
    "bloom_lm_hat":      {"origin": "authored_kreyol",      "genre": "educational",  "register": "children"},
    "storybooks_haiti":  {"origin": "human_translation",    "genre": "educational",  "register": "children"},
}

# --- register mix weights (TRAIN-time sampling guidance, reported not baked) ---
# The corpus holds everything; these are the *relative* per-register sampling
# weights a training run would use to lift authored/register signal above its raw
# token share. web_crawl is the bulk pool held at weight 1.0; authored registers
# are upweighted. Actual epoch budget + repetition caps are the model's call.
MIX_WEIGHTS = {
    "journalism": 4.0,
    "government": 3.0,
    "financial": 3.0,
    "children": 3.0,
    "legal": 3.0,
    "encyclopedic": 2.0,     # htwiki authored
    "religious": 1.0,        # capped by RELIGIOUS_CAP_FRAC regardless
    "web_crawl": 1.0,        # MADLAD + fineweb-2 bulk
}
RELIGIOUS_CAP_FRAC = 0.02    # hard cap: religious <= 2% of corpus tokens (Bib La)

# --- source pins -----------------------------------------------------------
FINEWEB_REPO = "HuggingFaceFW/fineweb-2"
FINEWEB_CONFIG = "hat_Latn"
BLOOM_REPO = "sil-ai/bloom-lm"
BLOOM_CONFIG = "hat"

BIB_LA_USFM_URL = "https://ebible.org/Scriptures/hat_usfm.zip"
KONSTITISYON_WIKISOURCE = "https://ht.wikisource.org/wiki/Konstitisyon_1987"
STORYBOOKS_REPO = "https://github.com/global-asp/storybooks-haiti"

# Federal PDF producers (J2). Each is 17 U.S.C. §105 PD; per-doc provenance
# carries the agency + register. Entry points are landing pages we enumerate.
FEDERAL_PRODUCERS = {
    "cdc":        {"register": "health",     "base": "https://stacks.cdc.gov/"},
    "uscis":      {"register": "immigration","base": "https://www.uscis.gov/tools/multilingual-resource-center"},
    "irs":        {"register": "tax",        "base": "https://www.irs.gov/"},
    "ssa":        {"register": "government",  "base": "https://www.ssa.gov/"},
    "fema_ready": {"register": "disaster",    "base": "https://www.ready.gov/ht"},
    "epa":        {"register": "health",      "base": "https://www.epa.gov/"},
    "hhs_osha":   {"register": "health",      "base": "https://www.osha.gov/"},
    "cfpb":       {"register": "financial",   "base": "https://www.consumerfinance.gov/"},
    "medlineplus":{"register": "health",      "base": "https://medlineplus.gov/languages/haitiancreolefrenchcreole.html"},
}

GLOSSARY_PAIRS_FEDERAL = os.path.join(config.REPO_ROOT, "corpus", "glossary_pairs_federal.json")
CONTRIBUTORS_DIR = os.path.join(config.REPO_ROOT, "corpus", "contributors")


def survivor_priority(priority_class: str) -> int:
    return SURVIVOR_PRIORITY_V0_2.get(priority_class, 5)
