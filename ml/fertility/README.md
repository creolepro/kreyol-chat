# fertility/ — Workstream C

Tokenizer **fertility** for Haitian Creole: how many tokens the same content costs in Kreyòl vs
English (and French) across ~8 tokenizers, with confidence intervals — plus the first published
Claude / o200k / Gemma-3 / Qwen3 / SmolLM3 numbers for HT that we could find. Protocol:
[../../docs/phase-0.md](../../docs/phase-0.md) (Workstream C) and
[../../docs/plan.md](../../docs/plan.md) §3.3.

## Run

```bash
cd ml
uv sync
# HF_TOKEN + ANTHROPIC_API_KEY are read from the repo-root .env (never committed)
uv run python -m fertility.run
```

The run **aborts** unless it first reproduces Petrov et al.'s cl100k Haitian/English ~1.74× on his
released data with our counting code (the pipeline gate).

## Outputs

- `results.csv` (here) — one row per tokenizer/API with parities, 95% CIs, per-sentence quantiles,
  tokens/word, whole-word survival, sentences-per-8k, and a date-stamped $ premium for priced APIs.
- `../reports/fertility_parity.png` — ht/en parity bar chart with CI error bars.
- `../reports/fertility.md` — methodology + results, including the Petrov replication, the pinned
  FLORES+ revision, and every tokenizer revision.

## Modules

| file | role |
|---|---|
| `config.py` | pinned revisions, model id, core word list, date-stamped price snapshot |
| `counting.py` | pure counting/stats: segmentation, sum-parity, paired-bootstrap CIs, quantiles, survival, 8k budget |
| `counters.py` | tiktoken + HF `AutoTokenizer` + Claude `count_tokens` counters |
| `data_sources.py` | Petrov download+replication; FLORES+ pinned download + join by (split, id) |
| `report.py` | the PNG chart and the `fertility.md` writer |
| `run.py` | the gate + orchestration + outputs (`python -m fertility.run`) |

## Data & licensing

FLORES+ is **eval-only by its terms** and Petrov's bundle is CC-BY-SA FLORES-200 — neither is
committed or re-hosted. Both download into the git-ignored `ml/data/` tree (FLORES+ via the HF
cache). Only aggregate numbers (`results.csv`, the report, the chart) are committed.

## Known gaps (see the report's "Skipped items & flags")

- **Llama-3** tokenizer row is empty when the HF gate is still pending — the script is re-runnable
  to fill it in once access is granted.
- **No authored-Kreyòl set** yet, so the translated-vs-authored (translationese) comparison is a
  TODO.
- **Our Kreyòl BPE** (`tokenizer/kreyol-bpe/tokenizer.json`, Workstream B) is now included — it
  lands at **ht/en 0.67× / ht/fr 0.57×**, flipping the token tax (Kreyòl costs fewer tokens than
  English). The script auto-detects it if the file exists.
