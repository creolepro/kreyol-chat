"""Scoring for the base-model probe (all LOCAL, CPU).

  * BPB — bits-per-byte is computed on-GPU (modal_app._score_bpb); here we only
    surface the already-aggregated total_nll_bits / total_utf8_bytes.
  * MT — sacreBLEU spBLEU (flores200 tokenizer) + chrF2++ (word_order=2, beta=2).
    The signatures are captured so the exact metric config is on the record.
  * Proverbs — exact-continuation hit + chrF near-miss against the gold second
    half of each probe proverb.
"""

from __future__ import annotations

import re
import unicodedata

import sacrebleu


# --- MT: spBLEU + chrF2++ -----------------------------------------------------

def _bleu_metric():
    # spBLEU = BLEU with the FLORES-200 SentencePiece tokenizer.
    return sacrebleu.BLEU(tokenize="flores200")


def _chrf_metric():
    # chrF2++ = chrF with word bigrams (word_order=2), char_order 6, beta 2.
    return sacrebleu.CHRF(char_order=6, word_order=2, beta=2)


def score_mt(hyps, refs):
    """Corpus spBLEU + chrF2++ for one direction. Returns scores + signatures."""
    bleu, chrf = _bleu_metric(), _chrf_metric()
    b = bleu.corpus_score(hyps, [refs])
    c = chrf.corpus_score(hyps, [refs])
    return {
        "spbleu": round(b.score, 2),
        "chrf2pp": round(c.score, 2),
        "n": len(hyps),
        "spbleu_sig": bleu.get_signature().format(short=False),
        "chrf2pp_sig": chrf.get_signature().format(short=False),
    }


# --- proverbs: exact-continuation hit / near-miss -----------------------------

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s).lower().strip()
    s = re.sub(r"[^\w\s]", "", s, flags=re.UNICODE)   # drop punctuation
    s = re.sub(r"\s+", " ", s).strip()
    return s


def score_proverbs(items, completions, near_miss_chrf=50.0):
    """Per-item: exact hit on the normalized continuation + chrF vs gold.

    A 'hit' means the model's continuation reproduces the gold second half
    exactly (normalized). 'near' means chrF >= near_miss_chrf but not exact.
    """
    chrf = sacrebleu.CHRF(char_order=6, word_order=2, beta=2)
    rows, n_hit, n_near = [], 0, 0
    for it, comp in zip(items, completions):
        gold, comp = it["gold"], (comp or "")
        exact = _norm(comp) == _norm(gold) and _norm(gold) != ""
        # also credit a hit if the gold appears verbatim at the start of output
        starts = _norm(comp).startswith(_norm(gold)) and _norm(gold) != ""
        hit = bool(exact or starts)
        cf = chrf.sentence_score(comp, [gold]).score
        near = (not hit) and cf >= near_miss_chrf
        n_hit += int(hit)
        n_near += int(near)
        rows.append({"num": it["num"], "full": it["full"],
                     "prompt_half": it["prompt_half"], "gold": gold,
                     "completion": comp, "chrf": round(cf, 1),
                     "hit": hit, "near": near, "english": it["english"]})
    return {"n": len(items), "n_hit": n_hit, "n_near": n_near, "rows": rows}
