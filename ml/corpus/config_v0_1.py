"""Pinned configuration for Phase-1 Workstream E — corpus v0.1 + standing eval slices.

Kept SEPARATE from the frozen phase-0 `config.py` so corpus v0 stays byte-for-byte
reproducible: v0.1 is a *new pipeline stage* layered on top of the v0 shard, not a
change to the v0 build. See docs/phase-1.md Workstream E and docs/plan.md §3.2/§5.3.

The junk-filter thresholds below were CALIBRATED against the 200 human-verified audit
labels (ml/reports/audit_model_labels.csv, human-concurred in audit_model_summary.md):
every rule here fired on ZERO of the 42 human-labeled *natural* crawl docs while
catching junk / commercial-translationese. The audit's ~17% crawl junk is mostly
*semantic* (fluent-looking machine-translated commercial text) and is NOT mechanically
fingerprintable — so these deterministic filters are deliberately high-precision /
low-recall. The residual translation-shaped material stays in v0.1 (flagged via the
translation_shaped_eval slice; its removal is fleet Q3's decision, not assumed here).
"""

from __future__ import annotations

import os

from . import config as C  # phase-0 paths/units (REPO_ROOT, CLEAN, DATA, REPORTS, ...)

# Stamped, not read from the clock, so a re-run reproduces the same report.
SNAPSHOT_DATE = "2026-07-21"

# --- inputs / outputs (all git-ignored under ml/data) -------------------------
# v0.1 is a strict document-SUBSET of v0 (drop-only): every surviving doc is
# byte-identical to its v0 form, so provenance + content hashes carry over and the
# size deltas are exact.
CORPUS_V0 = os.path.join(C.CLEAN, "corpus_v0-{tag}.jsonl")
CORPUS_V0_1 = os.path.join(C.CLEAN, "corpus_v0_1-{tag}.jsonl")
JUNK_STATS = os.path.join(C.CLEAN, "corpus_v0_1-{tag}.junk_stats.json")

# --- junk filters (deterministic; first trip attributes the drop) -------------
# Enable/disable + thresholds. "sources" limits a rule to certain priority classes
# ("crawl"/"wikipedia"/"owned") where it is safe; None = all sources.
JUNK = {
    # MT/CMS number-obfuscation placeholder (WordPress/Babylon artifact). The literal
    # string XNUMX/XNMX is never real Kreyòl/English, so >=2 occurrences is unambiguous.
    "mt_placeholder": {"min_count": 2, "sources": None},
    # >=3 distinct commercial/gambling/pharma spam markers (catalogs, casino, crypto,
    # steroid/pharma, wholesale/supplier SEO). Distinct-marker count, not raw hits, so a
    # single "bitcoin" in a news story does not trip it.
    "commercial_spam": {"min_distinct": 3, "sources": None},
    # >=4 price patterns ($/€/£/USD/HTG/Gourdes + digits): product/price-list pages.
    "price_listing": {"min_count": 4, "sources": None},
    # HTML entities left in the text (&#39; / &quot; / &#123; ...). Density per 1000
    # chars AND a floor count. Near-absent in the MADLAD "clean" split (documented), but
    # kept as a rule because it is the audit's canonical CMS fingerprint.
    "html_entity": {"min_density_per_1k": 2.0, "min_count": 3, "sources": None},
    # Non-Latin alphabetic ratio (Kreyòl is Latin-script). Catches Thai/Chinese-dominant
    # scraped spam. Crawl-only: Kreyòl Wikipedia legitimately has foreign-script names in
    # authored stubs about foreign places/people, which are NOT junk (audit finding).
    "foreign_script": {"min_ratio": 0.20, "min_chars": 100, "scan_chars": 4000,
                       "sources": ["crawl"]},
}

# untranslated-English-fragment density was CALIBRATED and dropped as a rule: it fired on
# ~0 cleaned crawl docs and would false-positive on authored Kreyòl-Wikipedia bios of
# foreign figures (English/French proper-noun bodies). Documented, not applied.
ENGLISH_FRAGMENT_EVALUATED = True

# Curated marker lists (lowercased; matched as whole words, case-insensitive).
SPAM_MARKERS = [
    # gambling
    "casino", "gambling", "betting", "roulette", "poker", "slots", "jackpot", "wager",
    "blackjack", "baccarat", "sportsbook", "betway", "bookmaker", "bonus", "payout", "odds",
    # crypto / finance spam
    "bitcoin", "crypto", "cryptocurrency", "forex", "binance", "ethereum",
    # pharma
    "steroid", "steroids", "anabolic", "dianabol", "clenbuterol", "viagra", "cialis",
    "tadalafil", "sildenafil", "pharmacy",
    # adult / dating spam
    "escort", "escorts", "porn", "xxx", "webcam", "hookup",
    # SEO / catalog
    "seo", "backlink", "backlinks", "affiliate", "wholesale", "supplier", "manufacturer",
]

# --- standing eval slices (registered in splits.yaml, never trained on) -------
# Membership: a human-verified TIER-A core from the audit labels + a conservative,
# well-defined TIER-B heuristic expansion (labelled separately in the manifest).
AUDIT_MODEL_LABELS = os.path.join(C.REPORTS, "audit_model_labels.csv")
EVAL_DIR = os.path.join(C.DATA, "eval")           # git-ignored manifests live here
# Manifest paths take a {suffix}: "" for the canonical full build (what splits.yaml
# and every training/eval consumer reference), "-sample" for the 1% smoke — so a
# sample run never clobbers the real slices.
AUTHORED_EVAL = os.path.join(EVAL_DIR, "authored_eval{suffix}.jsonl")
TRANSLATION_SHAPED_EVAL = os.path.join(EVAL_DIR, "translation_shaped_eval{suffix}.jsonl")
EVAL_SLICE_STATS = os.path.join(C.CLEAN, "eval_slices-{tag}.stats.json")


def eval_suffix(sample: bool) -> str:
    return "-sample" if sample else ""

EVAL_SEED = 20260721                # seeds the deterministic tier-B samples
AUTHORED_WIKI_CAP = 300             # tier-B: non-bot-stub Wikipedia authored sample cap
TRANS_HEURISTIC_CAP = 300          # tier-B: residual-fingerprint crawl translationese cap
# Tier-B translationese fingerprint: crawl docs carrying a residual MT/CMS signal that is
# NOT strong enough to be dropped as junk (>=1 HTML entity OR exactly one XNUMX) — a
# documented, precise translation artifact, seeded-sampled and capped.

# --- checkpoint prompt DRAFT (Workstream E3) ----------------------------------
# DRAFTED for the human to sign off — NOT frozen, NOT used for any metric. The
# proverb-completion slot is intentionally a placeholder: no probe-split proverb text
# is committed in any prompt set we build (Station-2 honesty constraint).
PROMPT_DRAFT_PATH = os.path.join(C.REPO_ROOT, "corpus", "checkpoint_prompts_DRAFT.json")
