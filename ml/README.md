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

No implementation code lives here yet — this is the initial scaffold. See the runbook before
adding scripts.
