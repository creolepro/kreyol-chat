# probe/ — Workstream D (base-model probe, Phase 0b)

A small **Modal** app that probes candidate pretrained *base* checkpoints (Qwen3-1.7B/4B,
SmolLM3-3B, Gemma-3-4B, Llama-3.2-3B control) to pick Model B's continued-pretraining
base **with recorded evidence**, and to bank the "before" numbers for the later
before/after adaptation comparison. See [../../docs/phase-0.md](../../docs/phase-0.md)
Workstream D and [../reports/base_model_probe.md](../reports/base_model_probe.md).

Everything is unquantized **bf16** via `transformers`; weights cache in a Modal volume so
reruns skip re-download. Budget ≈ $5–10 of GPU (one L40S).

## Run

```bash
# Modal must be authenticated (uv run modal profile current). HF_TOKEN in repo-root .env.
uv run python -m probe.run --stage all      # prefetch → smoke → main → fulldev → report
# or step through the funnel:
uv run python -m probe.run --stage smoke     # 20 dev sents/model — catch template/decoding bugs
uv run python -m probe.run --stage main      # full battery on the 250-item dev subset
uv run python -m probe.run --stage fulldev   # full dev MT on the top-2
uv run python -m probe.run --report          # rebuild reports from saved JSON (no GPU)
```

## Measures

- **Bits-per-byte (primary)** on the corpus `tokenizer_eval` holdout (never-trained docs),
  reported on the **authored-only** slice (Wikipedia non-stub + owned) and the **full**
  holdout slice. `total_nll_bits / total_utf8_bytes` → cross-tokenizer comparable.
- **Few-shot MT completion** (5-shot, fixed template, greedy), eng→hat and hat→eng, scored
  spBLEU + chrF2++ (sacreBLEU; signatures recorded).
- **Proverb completion** on the 15-item probe split — exact-continuation hit / near-miss.
- **Blinded naturalness** — 10 fixed prompts, outputs shuffled + de-identified into
  [../reports/probe_naturalness_sheet.md](../reports/probe_naturalness_sheet.md); hidden key
  in `naturalness_key.json`. A fluent speaker scores it — the harness does **not**.

## Data discipline

Selection uses FLORES+ **dev** only (pinned revision, joined by `(split, id)`); `final_devtest`
is reserved. BPB text is never-trained holdout docs. The 15 probe proverbs appear in no
training set (held out in Workstream A) and are re-asserted absent from the BPB slices. Raw
results (which embed FLORES prompts/refs) are written under the git-ignored `ml/data/probe/`;
only the reports (scores + model-generated / owned text) are committed.

## Modules

| file | role |
|---|---|
| `config.py` | pins: candidates, revisions-resolved-at-load, templates, decoding, seeds, BPB policy, Modal knobs |
| `data.py` | FLORES+ dev loader, seeded exemplar/subset split, MT prompts, holdout BPB slices, proverb + naturalness prompts |
| `modal_app.py` | Modal image/volume/GPU; `prefetch` (CPU) + `run_model` (bf16 BPB + greedy generation) |
| `measures.py` | BPB surfacing, spBLEU + chrF2++, proverb hit/near-miss |
| `run.py` | staged-funnel orchestrator + scorecard + top-2 selection |
| `report.py` | `base_model_probe.md`, naturalness sheet + hidden key |
