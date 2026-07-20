"""Pure counting + statistics — "our counting code".

No I/O, no network: given per-sentence token counts, produce the metrics defined
in docs/plan.md §3.3. Reused unchanged for the Petrov replication and the
FLORES+ measurement so the pipeline that validates is the pipeline that reports.
"""

from __future__ import annotations

import re
import unicodedata

import numpy as np

# --- text normalization + word segmentation -----------------------------------

def nfc(text: str) -> str:
    """NFC-normalize (è/ò consistency). No lowercasing, no accent stripping."""
    return unicodedata.normalize("NFC", text)


# Segmentation rule (documented in the report):
#   A "word" is a maximal run of Unicode letters, allowing an internal apostrophe
#   or hyphen that joins two letter-runs. Apostrophe-clitic forms such as `m'ap`,
#   `l'ap`, `n'ap` and hyphen compounds such as `pitit-pitit` therefore count as a
#   SINGLE word (we do not split on internal ' or -). Pure digits/punctuation are
#   not words. `[^\W\d_]` matches accented letters (è, ò, …); the pattern is
#   applied to NFC text.
_WORD_RE = re.compile(r"[^\W\d_]+(?:['’\-][^\W\d_]+)*", re.UNICODE)


def word_count(text: str) -> int:
    return len(_WORD_RE.findall(text))


def total_words(texts) -> int:
    return sum(word_count(t) for t in texts)


# --- parity + uncertainty ------------------------------------------------------

def corpus_parity(a_counts, b_counts) -> float:
    """Primary parity statistic: sum(a) / sum(b) over the whole corpus.

    This is NOT the mean of per-sentence ratios (a different statistic); we never
    silently substitute one for the other.
    """
    sb = float(np.sum(b_counts))
    if sb == 0:
        return float("nan")
    return float(np.sum(a_counts)) / sb


def paired_bootstrap_ci(a_counts, b_counts, reps: int, seed: int, alpha: float = 0.05):
    """Paired bootstrap 95% CI for sum(a)/sum(b).

    Resamples SENTENCE INDICES with replacement (a and b resampled together — the
    pairing is what makes the ratio meaningful), recomputes the sum-ratio each
    rep, and returns the (100*alpha/2, 100*(1-alpha/2)) percentiles.
    """
    a = np.asarray(a_counts, dtype=np.float64)
    b = np.asarray(b_counts, dtype=np.float64)
    n = len(a)
    rng = np.random.default_rng(seed)
    ratios = np.empty(reps, dtype=np.float64)
    for i in range(reps):
        idx = rng.integers(0, n, size=n)
        ratios[i] = a[idx].sum() / b[idx].sum()
    lo = float(np.percentile(ratios, 100 * alpha / 2))
    hi = float(np.percentile(ratios, 100 * (1 - alpha / 2)))
    return lo, hi


def per_sentence_ratio_quantiles(a_counts, b_counts, quantiles=(10, 50, 90)):
    """p10/p50/p90 of the per-sentence ratio a_i / b_i (b_i > 0 only)."""
    a = np.asarray(a_counts, dtype=np.float64)
    b = np.asarray(b_counts, dtype=np.float64)
    mask = b > 0
    r = a[mask] / b[mask]
    return {q: float(np.percentile(r, q)) for q in quantiles}


def tokens_per_word(total_tokens: int, total_word_count: int) -> float:
    if total_word_count == 0:
        return float("nan")
    return total_tokens / total_word_count


def sentences_per_budget(total_tokens: int, n_sentences: int, budget: int) -> float:
    """8192 ÷ mean tokens/sentence (order-independent). How many sentences of this
    language fit in a fixed context budget."""
    if total_tokens == 0:
        return float("nan")
    mean_per_sentence = total_tokens / n_sentences
    return budget / mean_per_sentence


# --- whole-word survival -------------------------------------------------------

def survival(count_fn, words):
    """Fraction of {word, ' '+word} that tokenize to a single token.

    `count_fn(str) -> int` returns the number of tokens for a raw string
    (content only). single-token == survives. Returns (n_single, n_total, rows)
    where rows is a per-word dict for the report.
    """
    n_single = 0
    n_total = 0
    rows = []
    for w in words:
        bare = count_fn(w) == 1
        lead = count_fn(" " + w) == 1
        n_single += int(bare) + int(lead)
        n_total += 2
        rows.append({"word": w, "bare_single": bare, "leading_space_single": lead})
    return n_single, n_total, rows
