"""Token counters.

Two kinds, kept deliberately distinct:

* TokenizerCounter  — a real tokenizer we run locally (tiktoken / HF / ours).
                      Counts CONTENT tokens only (no special tokens) and can
                      answer whole-word survival.
* ClaudeApiCounter  — the Anthropic `count_tokens` endpoint. This is an API
                      *estimate* that includes request scaffolding, so it is
                      measured in batches with the empty-request overhead
                      subtracted, and labeled as an API measurement — never a
                      raw tokenizer count.
"""

from __future__ import annotations

import time
from importlib.metadata import version

import tiktoken

from . import config

_TIKTOKEN_VER = f"tiktoken=={version('tiktoken')}"


class TokenizerCounter:
    """Wraps a local tokenizer. `count(text)` = number of content tokens."""

    kind = "tokenizer"
    supports_survival = True

    def __init__(self, label, count_fn, repo=None, revision=None):
        self.label = label
        self._count = count_fn
        self.repo = repo
        self.revision = revision

    def count(self, text: str) -> int:
        return self._count(text)


# --- builders -----------------------------------------------------------------

def build_tiktoken_counters():
    counters = []
    for name, label in config.TIKTOKEN_ENCODINGS:
        enc = tiktoken.get_encoding(name)
        # disallowed_special=() -> treat any special-token-looking substring as
        # ordinary text instead of raising.
        counters.append(
            TokenizerCounter(
                label,
                (lambda e: lambda t: len(e.encode(t, disallowed_special=())))(enc),
                repo=f"tiktoken:{name}",
                revision=_TIKTOKEN_VER,
            )
        )
    return counters


def build_hf_counters(hf_token, log):
    """One counter per HF repo. Resolves + pins + records the commit sha.

    Any failure (gated/auth/network/load) is logged and skipped so the run
    proceeds; the script is re-runnable to fill the row in later.
    """
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    api = HfApi(token=hf_token)
    counters, skipped = [], []
    for repo, label in config.HF_TOKENIZERS:
        try:
            sha = api.model_info(repo, token=hf_token).sha
            tok = AutoTokenizer.from_pretrained(
                repo, revision=sha, token=hf_token, trust_remote_code=False
            )
            counters.append(
                TokenizerCounter(
                    label,
                    (lambda tk: lambda t: len(tk.encode(t, add_special_tokens=False)))(tok),
                    repo=repo,
                    revision=sha,
                )
            )
            log(f"  loaded {label}  rev={sha[:12]}")
        except Exception as e:  # noqa: BLE001 — robustness is the point here
            reason = f"{type(e).__name__}: {' '.join(str(e).split())[:160]}"
            skipped.append({"label": label, "repo": repo, "reason": reason})
            log(f"  SKIP {label}: {reason}")
    return counters, skipped


def build_our_counter(repo_root, log):
    """Include our Kreyòl BPE tokenizer if Workstream B has produced it."""
    import os

    path = os.path.join(repo_root, config.OUR_TOKENIZER_PATH)
    if not os.path.exists(path):
        log(f"  our tokenizer not found at {config.OUR_TOKENIZER_PATH} "
            f"(expected — Workstream B not yet run); skipping")
        return None
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(path)
    return TokenizerCounter(
        config.OUR_TOKENIZER_LABEL,
        lambda t: len(tok.encode(t, add_special_tokens=False).ids),
        repo=config.OUR_TOKENIZER_PATH,
        revision="local",
    )


class ClaudeApiCounter:
    """Anthropic `count_tokens` batched corpus measurement.

    Not a per-sentence tokenizer: it returns total CONTENT tokens for a language
    (batches joined by newline, per-message overhead subtracted) and the
    per-batch counts (for a coarse batch-level bootstrap CI).
    """

    kind = "api"
    supports_survival = False

    def __init__(self, log):
        import anthropic

        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.model = config.CLAUDE_MODEL
        self.label = config.CLAUDE_LABEL
        self.batch_size = config.CLAUDE_BATCH_SIZE
        self._log = log
        self.repo = self.model
        self.revision = self.model
        self.overhead = self._count_message(config.CLAUDE_OVERHEAD_CONTENT)
        log(f"  Claude count_tokens overhead (content={config.CLAUDE_OVERHEAD_CONTENT!r}) "
            f"= {self.overhead} tokens (subtracted per batch)")

    def _count_message(self, content: str) -> int:
        """One count_tokens call with exponential backoff on 429/transient."""
        import anthropic

        delay = 1.0
        for attempt in range(7):
            try:
                r = self.client.messages.count_tokens(
                    model=self.model, messages=[{"role": "user", "content": content}]
                )
                return r.input_tokens
            except anthropic.RateLimitError:
                self._log(f"    429 — backoff {delay:.0f}s")
                time.sleep(delay)
                delay = min(delay * 2, 60)
            except (anthropic.APIConnectionError, anthropic.InternalServerError):
                time.sleep(delay)
                delay = min(delay * 2, 60)
        # final attempt without catching
        r = self.client.messages.count_tokens(
            model=self.model, messages=[{"role": "user", "content": content}]
        )
        return r.input_tokens

    def count_corpus(self, sentences):
        """Return (total_content_tokens, per_batch_content_tokens list)."""
        batch_tokens = []
        for start in range(0, len(sentences), self.batch_size):
            chunk = sentences[start:start + self.batch_size]
            raw = self._count_message("\n".join(chunk))
            batch_tokens.append(raw - self.overhead)
        return sum(batch_tokens), batch_tokens
