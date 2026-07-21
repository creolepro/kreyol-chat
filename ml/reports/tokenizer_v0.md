# Kreyòl tokenizer v0 — Workstream B

*Snapshot 2026-07-20. Byte-level BPE trained with **rustbpe** (nanochat pinned `92d63d4e8bb4`), inference via tiktoken. Pre-tokenization: **kreyol_aware** pattern (B0 decision — see [rustbpe_spike.md](rustbpe_spike.md)). See [../../docs/phase-0.md](../../docs/phase-0.md) Workstream B and [../../docs/plan.md](../../docs/plan.md) §3.2.*

## Pick

**Chosen vocab: 24,576.** next size (32,768) improves bytes/token by only 1.9% (< 3%) — compression has flattened. At 24,576, whole-word survival on the core Kreyòl list is **100%** and on the top-500 corpus words **78%**; the token-embedding table costs **37.7M** params at nanochat d12 width (768). On FLORES+ our tokenizer already compresses Kreyòl (**4.237** bytes/token) better than English (**3.029**) — a preview of the fertility flip (Workstream C).

## Training data + source weighting

Deterministic, seeded sample (seed 20260720) of the **train split only** (the tokenizer_eval holdout and the 15 probe proverbs are excluded). **Primary weighting = `natural`** (the corpus's own source proportions): a Kreyòl tokenizer should reflect what the corpus actually *is*, and re-weighting is a modeling choice we test separately rather than bake into v0.

- **natural** sample: 35,586 docs / 120M chars — crawl 92%, wikipedia 7%, owned 0%.
- **sensitivity** variant (16k only): crawl downweighted to ~60%, Wikipedia upweighted — 48,820 docs / 95M chars — crawl 60%, wikipedia 40%, owned 0%.

**Sensitivity result (16k):** holdout bytes/token 3.945 (natural) vs 3.898 (crawl-downweighted) — Δ=-0.048; top-500 survival 73% vs 70% (Δ=-3.1%). Downweighting crawl **does not beat** natural on either metric (both move by a hair, and if anything natural is slightly better), so **the weighting question is closed for v0** — natural stands. (A different *content* mix is a Phase-1 modeling choice, separate from the tokenizer.)

## Vocab sweep

Compression = **bytes per token on the held-out tokenizer_eval slice** (higher = better; fewer tokens for the same text). FLORES+ columns are measurement-only (Kreyòl out-of-domain; English/French are the code-switch regression check — reported, not gated).

Holdout: **2,109 docs** (owned 1, wikipedia 537, crawl 1,571). FLORES+ devtest: 1012 sentences/lang.

| vocab | bytes/token (holdout) | FLORES ht | FLORES en | FLORES fr | embed params (d12) | round-trip | survival core | survival top-500 | train |
|--:|--:|--:|--:|--:|--:|:--:|--:|--:|--:|
| **8,192** | 3.666 | 3.818 | 2.545 | 2.623 | 12.6M | ✓ | 98% | 66% | 4.1s |
| **16,384** | 3.945 | 4.104 | 2.820 | 2.939 | 25.2M | ✓ | 98% | 73% | 4.1s |
| **24,576** ⬅ | 4.073 | 4.237 | 3.029 | 3.159 | 37.7M | ✓ | 100% | 78% | 4.1s |
| **32,768** | 4.151 | 4.302 | 3.189 | 3.319 | 50.3M | ✓ | 100% | 84% | 4.2s |

### Compression curve (holdout bytes/token)

| vocab | bytes/token | Δ vs previous |
|--:|--:|--:|
| 8,192 | 3.666 | — |
| 16,384 | 3.945 | +7.6% |
| 24,576 | 4.073 | +3.2% |
| 32,768 | 4.151 | +1.9% |

**Pick rule:** smallest vocab where the next size adds < 3% bytes/token. → **24,576**. Bigger vocab keeps compressing but the marginal gain flattens while the embedding table grows linearly (below) — at ~200M params (d12) that trade stops being worth it.

## Embedding-table cost (nanochat d12, dim 768, untied wte+lm_head)

| vocab | embed params | share of a ~200M d12 model |
|--:|--:|--:|
| 8,192 | 12.6M | ~6% |
| 16,384 | 25.2M | ~13% |
| 24,576 | 37.7M | ~19% |
| 32,768 | 50.3M | ~25% |

(`vocab × 768 × 2`; nanochat pads vocab up to a multiple for kernels, so the real table is slightly larger. This is why we don't just take 32k — the biggest vocab nearly doubles the embedding cost for a few % compression.)

## Whole-word survival at 24,576 (exhibit metric)

Each word checked **bare and with a leading space**; single-token = survives. Core grammar list: **50/50 = 100%**. Top-500 corpus-frequency words (train split, probe excluded): **780/1000 = 78%**.

Core list, per word (bare / leading-space):

| word | bare | ` word` |    | word | bare | ` word` |
|---|:--:|:--:|---|---|:--:|:--:|
| `te` | ✓ | ✓ | | `yo` | ✓ | ✓ |
| `ta` | ✓ | ✓ | | `y` | ✓ | ✓ |
| `ap` | ✓ | ✓ | | `pa` | ✓ | ✓ |
| `pral` | ✓ | ✓ | | `la` | ✓ | ✓ |
| `va` | ✓ | ✓ | | `a` | ✓ | ✓ |
| `mwen` | ✓ | ✓ | | `an` | ✓ | ✓ |
| `m` | ✓ | ✓ | | `lan` | ✓ | ✓ |
| `ou` | ✓ | ✓ | | `nan` | ✓ | ✓ |
| `w` | ✓ | ✓ | | `sa` | ✓ | ✓ |
| `li` | ✓ | ✓ | | `ki` | ✓ | ✓ |
| `l` | ✓ | ✓ | | `gen` | ✓ | ✓ |
| `nou` | ✓ | ✓ | | `fè` | ✓ | ✓ |
| `n` | ✓ | ✓ | |  |  |  |

> The top-500 words are computed from corpus-v0 whitespace-word frequencies over the **train split only** (holdout + probe proverbs excluded), lowercasing-free on NFC text.

## Robustness spot-checks at 24,576

**caps:** `Ayiti`→`Ayiti`; `AYITI`→`AYITI`; `ayiti`→`ayiti`; `PÒTOPRENS`→`P` `ÒT` `OP` `R` `ENS`; `Pòtoprens`→`Pòtoprens`

**numbers:** `1804`→`18` `04`; `2026`→`20` `26`; `3.14`→`3` `.` `14`; `100000`→`10` `00` `00`; `12h30`→`12` `h` `30`

**spelling variants:** `ou`→`ou`; `w`→`w`; `mwen`→`mwen`; `m`→`m`; `li`→`li`; `l`→`l`; `kreyòl`→`kre` `yòl`; `kreyol`→`kre` `yol`

**accents apostrophe:** `fè`→`fè`; `fe`→`fe`; `m'ap`→`m` `'ap`; `n'ta`→`n` `'` `ta`; `lòt`→`lòt`; `sè`→`sè`

## English-control ablation (exhibit)

A second 16k tokenizer trained with identical settings on a size-matched sample of **wikitext-103-raw** (English, CC-BY-SA 3.0), then both tokenizers run on the 35 teachable proverbs:

- Same proverbs cost **329 tokens** under the Kreyòl tokenizer vs **607** under the English one (**1.845× more** on English).
- Of 154 distinct proverb words, **83** survive whole under Kreyòl but shatter under English. Examples:

| word | Kreyòl | English | English pieces |
|---|--:|--:|---|
| `piti` | 1 | 2 | ` pit` `i` |
| `zwazo` | 1 | 4 | ` z` `w` `az` `o` |
| `fè` | 1 | 2 | ` f` `è` |
| `nich` | 1 | 2 | ` n` `ich` |
| `nan` | 1 | 2 | ` n` `an` |
| `dlo` | 1 | 2 | ` d` `lo` |
| `pa` | 1 | 2 | ` p` `a` |
| `konnen` | 1 | 3 | ` k` `onn` `en` |
| `doulè` | 1 | 3 | ` dou` `l` `è` |
| `wòch` | 1 | 4 | ` w` `�` `�` `ch` |
| `Lè` | 1 | 2 | ` L` `è` |
| `ou` | 1 | 2 | ` o` `u` |
| `gade` | 1 | 2 | ` g` `ade` |
| `lavi` | 1 | 2 | ` l` `avi` |
| `kat` | 1 | 2 | ` k` `at` |
| `li` | 1 | 2 | ` l` `i` |
| `ka` | 1 | 2 | ` k` `a` |
| `mache` | 1 | 2 | ` m` `ache` |
| `chemen` | 1 | 2 | ` chem` `en` |
| `menm` | 1 | 2 | ` men` `m` |
| `Bondye` | 1 | 2 | ` Bond` `ye` |
| `konn` | 1 | 2 | ` k` `onn` |
| `tanbou` | 1 | 4 | ` t` `an` `b` `ou` |
| `chay` | 1 | 2 | ` ch` `ay` |
| `Moun` | 1 | 2 | ` M` `oun` |

## Exported artifacts

The chosen tokenizer is committed under `ml/tokenizer/kreyol-bpe/` in three formats (+ the tiktoken pickle + meta):

- `tokenizer.json` — HF `tokenizers`/transformers (and the fertility script).
- `vocab_merges.json` — plain `{vocab, merges, pattern, special_tokens}` for the browser.
- `tokenizer_tiktoken.json` — tiktoken-native base64 (js-tiktoken, exact).
- Export parity: the HF `tokenizer.json` reproduces tiktoken IDs on **1035/1035 = 100.0%** of a 1k-line probe (source of truth = the rustbpe/tiktoken encoding).

## Rights note

The tokenizer is a **derived model artifact** trained on the corpus-v0 train split (MADLAD-400 ht, ht Wikipedia, teachable proverbs). `rights.yaml` allows **tokenizer training** on all three; the only open item is MADLAD *text redistribution*. The committed vocab is short byte-level subwords (longest entries are single words / morphological suffixes like `-syon`, punctuation runs — no verbatim phrases or PII), which is a standard publishable artifact and does not re-host source text. Quarantined sources (MIT-Haiti, the dictionary) and eval-only FLORES+ were **not** in training. If MADLAD's license resolves restrictively, revisit before any *commercial* redistribution.

## Reproduce

```bash
cd ml && uv sync
uv run python -m tokenizer.spike   # B0 go/no-go
uv run python -m tokenizer.run     # sweep + eval + export + this report
```
