"""KreyolBPE — train with rustbpe, inference with tiktoken.

A minimal wrapper that mirrors nanochat/tokenizer.py (pinned NANOCHAT_COMMIT) but
without pulling in nanochat's torch stack: we only need rustbpe + tiktoken. The
SPLIT_PATTERN and the 9 SPECIAL_TOKENS are copied from that commit; the special
tokens are appended AFTER training, so rustbpe trains `vocab_size - 9` merges.

Byte-level BPE: content tokens are counted with `encode_ordinary` (no special
tokens), matching Workstream C's `add_special_tokens=False` convention.
"""

from __future__ import annotations

import json
import os
import pickle

import rustbpe
import tiktoken

from . import config


class KreyolBPE:
    def __init__(self, enc: tiktoken.Encoding, mergeable_ranks: dict, pattern: str):
        self.enc = enc
        self.mergeable_ranks = mergeable_ranks   # dict[bytes, int]
        self.pattern = pattern

    # --- training -------------------------------------------------------------

    @classmethod
    def train(cls, text_iterator, vocab_size: int, pattern: str):
        """Train a byte-level BPE. `vocab_size` INCLUDES the special tokens."""
        n_special = len(config.SPECIAL_TOKENS)
        vocab_no_special = vocab_size - n_special
        assert vocab_no_special >= 256, f"vocab too small: {vocab_no_special}"
        rt = rustbpe.Tokenizer()
        rt.train_from_iterator(text_iterator, vocab_no_special, pattern=pattern)
        trained_pattern = rt.get_pattern()
        mergeable_ranks = {bytes(k): v for k, v in rt.get_mergeable_ranks()}
        enc = cls._build_enc(mergeable_ranks, trained_pattern)
        return cls(enc, mergeable_ranks, trained_pattern)

    @staticmethod
    def _build_enc(mergeable_ranks: dict, pattern: str) -> tiktoken.Encoding:
        offset = len(mergeable_ranks)
        special = {name: offset + i for i, name in enumerate(config.SPECIAL_TOKENS)}
        return tiktoken.Encoding(
            name="kreyol-bpe",
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks,
            special_tokens=special,
        )

    # --- inference (content tokens only) --------------------------------------

    def encode_ordinary(self, text: str):
        return self.enc.encode_ordinary(text)

    def count(self, text: str) -> int:
        return len(self.enc.encode_ordinary(text))

    def decode(self, ids):
        return self.enc.decode(ids)

    def id_to_bytes(self, tid: int) -> bytes:
        return self.enc.decode_single_token_bytes(tid)

    @property
    def vocab_size(self) -> int:
        return self.enc.n_vocab

    @property
    def n_content_tokens(self) -> int:
        return len(self.mergeable_ranks)

    # --- persistence ----------------------------------------------------------

    def save_pkl(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"mergeable_ranks": self.mergeable_ranks, "pattern": self.pattern}, f)

    @classmethod
    def load_pkl(cls, path: str):
        with open(path, "rb") as f:
            d = pickle.load(f)
        enc = cls._build_enc(d["mergeable_ranks"], d["pattern"])
        return cls(enc, d["mergeable_ranks"], d["pattern"])

    def save_meta(self, path: str, extra: dict | None = None):
        meta = {
            "name": "kreyol-bpe",
            "nanochat_commit": config.NANOCHAT_COMMIT,
            "pattern": self.pattern,
            "pattern_name": config.CHOSEN_PATTERN_NAME,
            "vocab_size": self.vocab_size,
            "n_content_tokens": self.n_content_tokens,
            "special_tokens": config.SPECIAL_TOKENS,
            "snapshot_date": config.SNAPSHOT_DATE,
        }
        if extra:
            meta.update(extra)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
