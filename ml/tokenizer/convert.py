"""Format bridges: rustbpe/tiktoken -> HF tokenizer.json + browser {vocab,merges}.

The tiktoken encoding built from rustbpe's mergeable_ranks is the SOURCE OF TRUTH
(it's what nanochat/Phase 1 uses). This module derives:
  (a) an HF `tokenizers` tokenizer.json (loadable by tokenizers/transformers, and
      by the fertility script), and
  (b) a plain {vocab, merges, pattern, special_tokens} JSON for the browser,
plus a tiktoken-native base64 dump (js-tiktoken, exact by construction).

`parity` re-encodes a probe set through the HF bridge and compares token IDs to
tiktoken, so any divergence (usually pre-tokenizer regex-engine differences) is
measured, not assumed.
"""

from __future__ import annotations

import base64
import json
import os

from . import config


# --- GPT-2 byte<->unicode (the exact map HF ByteLevel uses) -------------------

def bytes_to_unicode():
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


_B2U = bytes_to_unicode()


def _tok_to_str(tb: bytes) -> str:
    return "".join(_B2U[b] for b in tb)


# --- merge reconstruction from mergeable_ranks --------------------------------

def reconstruct_merges(mergeable_ranks: dict):
    """Recover ordered BPE merges (byte-level-unicode string pairs) from ranks."""
    ranks = mergeable_ranks

    def split(tb: bytes, max_rank: int):
        parts = [bytes([b]) for b in tb]
        while len(parts) > 1:
            best_i, best_rank = None, None
            for i in range(len(parts) - 1):
                r = ranks.get(parts[i] + parts[i + 1])
                if r is not None and r < max_rank and (best_rank is None or r < best_rank):
                    best_i, best_rank = i, r
            if best_i is None:
                break
            parts = parts[:best_i] + [parts[best_i] + parts[best_i + 1]] + parts[best_i + 2:]
        return parts

    merges = []
    for tb, rank in sorted(ranks.items(), key=lambda x: x[1]):
        if len(tb) < 2:
            continue
        parts = split(tb, rank)
        if len(parts) == 2:
            merges.append((_tok_to_str(parts[0]), _tok_to_str(parts[1])))
        else:
            merges.append(None)  # unreconstructable (should not happen for BPE)
    return merges


# --- HF tokenizer.json --------------------------------------------------------

def build_hf_tokenizer(kbpe):
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, Regex

    vocab = {_tok_to_str(tb): rank for tb, rank in kbpe.mergeable_ranks.items()}
    merges = [m for m in reconstruct_merges(kbpe.mergeable_ranks) if m is not None]
    tok = Tokenizer(models.BPE(vocab=vocab, merges=merges, fuse_unk=False, byte_fallback=False))
    # Replicate tiktoken: regex findall (Split/isolated) then byte-level mapping.
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(kbpe.pattern), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    tok.add_special_tokens(config.SPECIAL_TOKENS)   # ids continue after content vocab
    return tok


def export_hf(kbpe, path: str):
    tok = build_hf_tokenizer(kbpe)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tok.save(path)
    return tok


# --- browser exports ----------------------------------------------------------

def export_browser(kbpe, path_vocab_merges: str, path_tiktoken: str):
    """(b) plain {vocab, merges} JSON + a tiktoken-native base64 dump (js-tiktoken)."""
    vocab = {_tok_to_str(tb): rank for tb, rank in kbpe.mergeable_ranks.items()}
    merges = [f"{a} {b}" for m in reconstruct_merges(kbpe.mergeable_ranks) if m for a, b in [m]]
    offset = len(kbpe.mergeable_ranks)
    specials = {name: offset + i for i, name in enumerate(config.SPECIAL_TOKENS)}
    os.makedirs(os.path.dirname(path_vocab_merges), exist_ok=True)
    with open(path_vocab_merges, "w", encoding="utf-8") as f:
        json.dump({"vocab": vocab, "merges": merges, "pattern": kbpe.pattern,
                   "special_tokens": specials, "byte_encoder": "gpt2"}, f, ensure_ascii=False)
    # tiktoken-native (base64(bytes) -> rank), exact for js-tiktoken
    tk = {base64.b64encode(tb).decode("ascii"): rank
          for tb, rank in kbpe.mergeable_ranks.items()}
    with open(path_tiktoken, "w", encoding="utf-8") as f:
        json.dump({"bpe_ranks_b64": tk, "pat_str": kbpe.pattern,
                   "special_tokens": specials}, f, ensure_ascii=False)


# --- parity check -------------------------------------------------------------

def parity(kbpe, hf_tok, sentences):
    """Exact-token-ID parity of the HF bridge vs tiktoken on a probe set."""
    same_sents = 0
    tok_total = 0
    tok_match = 0
    first_divergence = None
    for s in sentences:
        a = kbpe.encode_ordinary(s)
        b = hf_tok.encode(s, add_special_tokens=False).ids
        tok_total += max(len(a), len(b))
        m = sum(1 for x, y in zip(a, b) if x == y)
        tok_match += m
        if a == b:
            same_sents += 1
        elif first_divergence is None:
            first_divergence = {"text": s[:120], "tiktoken": a[:20], "hf": b[:20]}
    n = len(sentences)
    return {
        "n_sentences": n,
        "sentence_exact_match": same_sents,
        "sentence_exact_frac": same_sents / n if n else 0.0,
        "token_match_frac": tok_match / tok_total if tok_total else 0.0,
        "first_divergence": first_divergence,
    }
