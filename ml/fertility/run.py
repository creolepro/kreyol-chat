"""Workstream C entry point.

Order of operations (docs/phase-0.md Workstream C, docs/plan.md §3.3):

  1. PIPELINE VALIDATION — reproduce Petrov's ht/en cl100k ~1.74x from his
     released data with our counting code. Printed prominently; the run ABORTS
     if it does not land on his number.
  2. FLORES+ measurement — pinned devtest, joined by (split, id), NFC-normalized.
  3. Tokenizers — tiktoken cl100k/o200k + HF (Gemma/Qwen3/NLLB/SmolLM3/Llama-3)
     + our BPE if present.
  4. Claude — separately-labeled `count_tokens` API measurement.

Outputs: fertility/results.csv, reports/fertility_parity.png, reports/fertility.md.

Run:  cd ml && uv run python -m fertility.run
"""

from __future__ import annotations

import csv
import os
import pathlib
import sys

import numpy as np

from . import config, counters
from . import counting as C
from . import data_sources as D
from . import report as R

ML_DIR = pathlib.Path(__file__).resolve().parent.parent      # .../ml
REPO_ROOT = ML_DIR.parent                                    # repo root
DATA_DIR = ML_DIR / "data"
RESULTS_CSV = ML_DIR / "fertility" / "results.csv"
REPORTS_DIR = ML_DIR / "reports"
PNG_PATH = REPORTS_DIR / "fertility_parity.png"
MD_PATH = REPORTS_DIR / "fertility.md"


def log(msg=""):
    print(msg, flush=True)


def load_env():
    """Load HF_TOKEN / ANTHROPIC_API_KEY from the repo-root .env (never committed)."""
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k, v)
    return os.environ.get("HF_TOKEN"), os.environ.get("ANTHROPIC_API_KEY")


# --- step 1: Petrov gate ------------------------------------------------------

def run_petrov_gate() -> dict:
    log("=" * 72)
    log("STEP 1 — PIPELINE VALIDATION: Petrov et al. 2023 cl100k replication")
    log("=" * 72)
    rep = D.petrov_replication(str(DATA_DIR), log)
    log(f"  his released totals : ht={rep['his_ht_tokens']}  en={rep['his_en_tokens']}"
        f"  ->  ht/en = {rep['his_csv_parity']:.4f}")
    log(f"  our recomputed      : ht={rep['our_ht_tokens']}  en={rep['our_en_tokens']}"
        f"  ->  ht/en = {rep['our_parity']:.4f}   (n={rep['n_sentences_our']} devtest)")
    log("")
    log(f"  >>> cl100k Haitian/English parity (OUR code on HIS data): "
        f"{rep['our_parity']:.4f}  (Petrov: {rep['his_csv_parity']:.4f}) <<<")
    ok = abs(rep["our_parity"] - config.PETROV_TARGET) <= config.PETROV_TOLERANCE
    if not ok:
        log("")
        log("  !! REPLICATION FAILED — parity outside tolerance; ABORTING. !!")
        sys.exit(1)
    log(f"  PASS (within ±{config.PETROV_TOLERANCE} of {config.PETROV_TARGET}). "
        f"Pipeline validated; proceeding.\n")
    return rep


# --- metrics ------------------------------------------------------------------

def _priced(label):
    return config.API_INPUT_PRICES_USD_PER_MTOK.get(label)


def metrics_for_tokenizer(counter, texts, words) -> dict:
    """Full per-tokenizer metrics from per-sentence counts."""
    ht = [counter.count(t) for t in texts["ht"]]
    en = [counter.count(t) for t in texts["en"]]
    fr = [counter.count(t) for t in texts["fr"]]
    n = len(ht)

    parity_he = C.corpus_parity(ht, en)
    lo_he, hi_he = C.paired_bootstrap_ci(ht, en, config.BOOTSTRAP_REPS, config.BOOTSTRAP_SEED)
    parity_hf = C.corpus_parity(ht, fr)
    lo_hf, hi_hf = C.paired_bootstrap_ci(ht, fr, config.BOOTSTRAP_REPS, config.BOOTSTRAP_SEED)
    q = C.per_sentence_ratio_quantiles(ht, en)

    n_single, n_total, surv_rows = C.survival(counter.count, config.CORE_WORDS)

    row = {
        "label": counter.label,
        "kind": counter.kind,
        "repo": counter.repo,
        "revision": counter.revision,
        "n_sentences": n,
        "ht_tokens": int(np.sum(ht)),
        "en_tokens": int(np.sum(en)),
        "fr_tokens": int(np.sum(fr)),
        "parity_ht_en": round(parity_he, 4),
        "ci95_lo_ht_en": round(lo_he, 4),
        "ci95_hi_ht_en": round(hi_he, 4),
        "parity_ht_fr": round(parity_hf, 4),
        "ci95_lo_ht_fr": round(lo_hf, 4),
        "ci95_hi_ht_fr": round(hi_hf, 4),
        "p10_ht_en": round(q[10], 4),
        "p50_ht_en": round(q[50], 4),
        "p90_ht_en": round(q[90], 4),
        "ht_tokens_per_word": round(C.tokens_per_word(int(np.sum(ht)), words["ht"]), 4),
        "en_tokens_per_word": round(C.tokens_per_word(int(np.sum(en)), words["en"]), 4),
        "fr_tokens_per_word": round(C.tokens_per_word(int(np.sum(fr)), words["fr"]), 4),
        "survival_single": n_single,
        "survival_total": n_total,
        "survival_rate": round(n_single / n_total, 4),
        "sentences_per_8k_ht": round(C.sentences_per_budget(int(np.sum(ht)), n, config.CONTEXT_BUDGET), 1),
        "sentences_per_8k_en": round(C.sentences_per_budget(int(np.sum(en)), n, config.CONTEXT_BUDGET), 1),
        "pct_premium_vs_en": round((parity_he - 1) * 100, 1),
        "ci_note": "",
        "_survival_rows": surv_rows,
    }
    _attach_price(row, counter.label, parity_he)
    return row


def metrics_for_claude(counter, texts, words) -> dict:
    """Claude: batched corpus totals; per-sentence metrics are N/A by design."""
    ht_total, ht_batches = counter.count_corpus(texts["ht"])
    en_total, en_batches = counter.count_corpus(texts["en"])
    fr_total, fr_batches = counter.count_corpus(texts["fr"])
    n = len(texts["ht"])

    parity_he = ht_total / en_total
    parity_hf = ht_total / fr_total
    # coarse batch-level paired bootstrap (few batches — labeled as such)
    lo_he, hi_he = C.paired_bootstrap_ci(ht_batches, en_batches, config.BOOTSTRAP_REPS, config.BOOTSTRAP_SEED)
    lo_hf, hi_hf = C.paired_bootstrap_ci(ht_batches, fr_batches, config.BOOTSTRAP_REPS, config.BOOTSTRAP_SEED)

    row = {
        "label": counter.label,
        "kind": counter.kind,
        "repo": counter.repo,
        "revision": counter.revision,
        "n_sentences": n,
        "ht_tokens": ht_total,
        "en_tokens": en_total,
        "fr_tokens": fr_total,
        "parity_ht_en": round(parity_he, 4),
        "ci95_lo_ht_en": round(lo_he, 4),
        "ci95_hi_ht_en": round(hi_he, 4),
        "parity_ht_fr": round(parity_hf, 4),
        "ci95_lo_ht_fr": round(lo_hf, 4),
        "ci95_hi_ht_fr": round(hi_hf, 4),
        "p10_ht_en": "", "p50_ht_en": "", "p90_ht_en": "",
        "ht_tokens_per_word": round(C.tokens_per_word(ht_total, words["ht"]), 4),
        "en_tokens_per_word": round(C.tokens_per_word(en_total, words["en"]), 4),
        "fr_tokens_per_word": round(C.tokens_per_word(fr_total, words["fr"]), 4),
        "survival_single": "", "survival_total": "", "survival_rate": "",
        "sentences_per_8k_ht": round(C.sentences_per_budget(ht_total, n, config.CONTEXT_BUDGET), 1),
        "sentences_per_8k_en": round(C.sentences_per_budget(en_total, n, config.CONTEXT_BUDGET), 1),
        "pct_premium_vs_en": round((parity_he - 1) * 100, 1),
        "ci_note": f"batch-level bootstrap ({len(ht_batches)} batches of "
                   f"~{config.CLAUDE_BATCH_SIZE}); overhead {counter.overhead} tok/msg subtracted; "
                   f"per-sentence quantiles/survival N/A (API measurement)",
        "_survival_rows": None,
    }
    _attach_price(row, counter.label, parity_he)
    return row


def _attach_price(row, label, parity_he):
    priced = _priced(label)
    if priced:
        usd, src = priced
        row["api_input_usd_per_mtok"] = usd
        row["price_snapshot_date"] = config.PRICE_DATE
        row["price_source"] = src
        # illustrative extra $ to process the ht side of this 1012-sentence set
        row["usd_premium_flores_ht_vs_en"] = round(
            (row["ht_tokens"] - row["en_tokens"]) / 1e6 * usd, 6)
    else:
        row["api_input_usd_per_mtok"] = ""
        row["price_snapshot_date"] = ""
        row["price_source"] = ""
        row["usd_premium_flores_ht_vs_en"] = ""


CSV_FIELDS = [
    "label", "kind", "repo", "revision", "n_sentences",
    "ht_tokens", "en_tokens", "fr_tokens",
    "parity_ht_en", "ci95_lo_ht_en", "ci95_hi_ht_en",
    "parity_ht_fr", "ci95_lo_ht_fr", "ci95_hi_ht_fr",
    "p10_ht_en", "p50_ht_en", "p90_ht_en",
    "ht_tokens_per_word", "en_tokens_per_word", "fr_tokens_per_word",
    "survival_single", "survival_total", "survival_rate",
    "sentences_per_8k_ht", "sentences_per_8k_en",
    "pct_premium_vs_en",
    "api_input_usd_per_mtok", "price_snapshot_date", "price_source",
    "usd_premium_flores_ht_vs_en",
    "ci_note",
]


def write_csv(rows):
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    log(f"  wrote {RESULTS_CSV.relative_to(REPO_ROOT)}")


# --- main ---------------------------------------------------------------------

def main():
    hf_token, anthropic_key = load_env()

    petrov = run_petrov_gate()

    log("=" * 72)
    log("STEP 2 — FLORES+ measurement set")
    log("=" * 72)
    flores = D.load_flores(hf_token, log)
    texts = flores["texts"]
    words = {code: C.total_words(texts[code]) for code in ("ht", "en", "fr")}
    log(f"  words (segmentation rule): ht={words['ht']} en={words['en']} fr={words['fr']}\n")

    log("=" * 72)
    log("STEP 3 — Tokenizers")
    log("=" * 72)
    tok_counters = list(counters.build_tiktoken_counters())
    hf_counters, skipped = counters.build_hf_counters(hf_token, log)
    tok_counters += hf_counters
    ours = counters.build_our_counter(str(ML_DIR), log)
    if ours:
        tok_counters.append(ours)

    rows = []
    for c in tok_counters:
        log(f"  measuring {c.label} ...")
        rows.append(metrics_for_tokenizer(c, texts, words))

    log("")
    log("=" * 72)
    log("STEP 4 — Claude API input-parity measurement")
    log("=" * 72)
    claude_row = None
    if anthropic_key:
        try:
            claude = counters.ClaudeApiCounter(log)
            log(f"  measuring {claude.label} (batched) ...")
            claude_row = metrics_for_claude(claude, texts, words)
            rows.append(claude_row)
        except Exception as e:  # noqa: BLE001
            reason = f"{type(e).__name__}: {' '.join(str(e).split())[:160]}"
            skipped.append({"label": config.CLAUDE_LABEL, "repo": config.CLAUDE_MODEL, "reason": reason})
            log(f"  SKIP Claude: {reason}")
    else:
        skipped.append({"label": config.CLAUDE_LABEL, "repo": config.CLAUDE_MODEL,
                        "reason": "ANTHROPIC_API_KEY not set"})
        log("  SKIP Claude: ANTHROPIC_API_KEY not set")

    log("")
    log("=" * 72)
    log("Writing outputs")
    log("=" * 72)
    write_csv(rows)
    R.write_png(rows, str(PNG_PATH), REPO_ROOT, log)
    R.write_markdown(rows, petrov, flores, words, skipped, str(MD_PATH), REPO_ROOT, log)

    # console summary
    log("")
    log("PARITY SUMMARY (ht/en, sum-based, 95% CI):")
    for r in sorted(rows, key=lambda x: x["parity_ht_en"]):
        log(f"  {r['label']:<42} {r['parity_ht_en']:.3f}  "
            f"[{r['ci95_lo_ht_en']:.3f}, {r['ci95_hi_ht_en']:.3f}]")
    log("\nDone.")


if __name__ == "__main__":
    main()
