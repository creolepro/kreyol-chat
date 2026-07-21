# tokenizer/ — Workstream B

Kreyòl **tokenizer v0**: byte-level BPE trained with nanochat's **rustbpe** (pinned commit),
inference via tiktoken. Vocab chosen from an {8k, 16k, 24k, 32k} sweep. Protocol:
[../../docs/phase-0.md](../../docs/phase-0.md) Workstream B (incl. the B0 spike),
[../../docs/plan.md](../../docs/plan.md) §3.2.

## Run

```bash
cd ml
uv sync
uv run python -m tokenizer.spike   # B0 integration spike -> ../reports/rustbpe_spike.md (go/no-go)
uv run python -m tokenizer.run     # sweep + eval + export -> ../reports/tokenizer_v0.md
```

Corpus v0 is read from the git-ignored `data/clean/`; sweep scratch lands in git-ignored
`data/tokenizer_work/`. HF_TOKEN / FLORES access are read from the repo-root `.env`.

## Chosen tokenizer (committed)

`kreyol-bpe/` — the chosen **24,576**-vocab tokenizer, in three formats + the tiktoken pickle:

| file | for |
|---|---|
| `tokenizer.json` | HF `tokenizers`/transformers (and the fertility script) |
| `vocab_merges.json` | plain `{vocab, merges, pattern, special_tokens}` for the browser |
| `tokenizer_tiktoken.json` | tiktoken-native base64 (js-tiktoken, exact) |
| `tokenizer.pkl` | rustbpe/tiktoken source-of-truth encoding |
| `meta.json` | pinned commit, pattern, vocab, training-sample composition |

All three text formats reproduce the rustbpe/tiktoken token IDs exactly (100% on a 1k-line probe).
On FLORES+ it lands at **ht/en 0.67× / ht/fr 0.57×** — it flips the token tax (Workstream C).

## Design decisions (see the reports)

- **rustbpe** is a standalone PyPI wheel — no Rust source build needed. B0 verdict: **GO**.
- **Kreyòl-aware pre-tokenization pattern**: nanochat's GPT-4 pattern hard-codes English
  contractions (`'(?i:[sdmt]|ll|ve|re)`), which shreds Kreyòl TMA forms like `m'te`/`n'ta`
  into `'t`+vowel. We drop that clause — `m'ap`-family and English single-letter contractions
  stay identical, `m'te`/`n'ta` are fixed. Full evidence in `../reports/rustbpe_spike.md`.
- **Vocab 24,576**: smallest size where holdout compression flattens (< 3% gain to the next
  size) while the embedding table stays affordable at d12 width. See `../reports/tokenizer_v0.md`.

## Modules

| file | role |
|---|---|
| `config.py` | pinned nanochat commit, split patterns, vocab sweep, seeds, weighting, d12 width |
| `core.py` | `KreyolBPE` — train (rustbpe) + inference (tiktoken), save/load |
| `convert.py` | tiktoken → HF `tokenizer.json` + browser JSON, with exact-ID parity check |
| `data.py` | corpus streaming, seeded holdout, weighted training samples, top-N word freqs |
| `external.py` | FLORES+ (compression/regression) + wikitext-103 (English-control ablation) |
| `sweep.py` | sample materialization, training, metrics (compression, survival, robustness) |
| `spike.py` | the B0 integration spike (`python -m tokenizer.spike`) |
| `report.py` / `run.py` | `tokenizer_v0.md` writer + the sweep orchestrator |
