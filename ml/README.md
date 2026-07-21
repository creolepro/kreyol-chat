# ml/

Python workspace for all model, corpus, and tokenizer work. Managed with
[uv](https://docs.astral.sh/uv/); Python ≥ 3.12.

The current focus is **Phase 0** — assembling corpus v0, training the Kreyòl tokenizer, measuring
fertility across tokenizers, and probing candidate base models. The full runbook is
[../docs/phase-0.md](../docs/phase-0.md); the overall plan is [../docs/plan.md](../docs/plan.md).

## Setup

```bash
cd ml
uv sync          # create .venv and install dependencies from uv.lock
```

## Layout

```
corpus/      # Workstream A — ingest → normalize → filter → dedup → corpus v0
tokenizer/   # Workstream B — byte-level BPE sweep, whole-word survival eval, export
fertility/   # Workstream C — tokens-per-content across ~8 tokenizers → CSV + chart
probe/       # Workstream D — zero-shot base-model probe (Modal app)
reports/     # generated stats and markdown reports
data/        # git-ignored — downloaded sources (raw/), corpus shards (clean/), eval sets (eval/)
```

Implemented so far:

- **`corpus/`** — Workstreams 0 + A: rights matrix + split registry, and the corpus v0
  pipeline (ingest → normalize → filter → dedup → audit → report). Run `python -m corpus.run
  --sample` then `python -m corpus.run`; see [corpus/README.md](corpus/README.md).
- **`tokenizer/`** — Workstream B: Kreyòl byte-level BPE (rustbpe, vocab sweep → 24k), the B0
  integration spike, and the chosen tokenizer under `tokenizer/kreyol-bpe/`; see
  [tokenizer/README.md](tokenizer/README.md), [reports/tokenizer_v0.md](reports/tokenizer_v0.md),
  and [reports/rustbpe_spike.md](reports/rustbpe_spike.md).
- **`fertility/`** — Workstream C: tokenizer fertility across ~8 tokenizers (incl. ours at
  ht/en 0.67×); see [fertility/README.md](fertility/README.md) and
  [reports/fertility.md](reports/fertility.md).

`probe/` (Workstream D) is still scaffold. See the runbook before adding scripts.
