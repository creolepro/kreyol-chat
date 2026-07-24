# ml/

Python workspace for all model, corpus, and tokenizer work. Managed with
[uv](https://docs.astral.sh/uv/); Python ≥ 3.12.

**Phase 0** is complete (corpus v0, Kreyòl tokenizer, fertility, base-model probe). **Phase 1**
is under way: **Workstreams E** (corpus v0.1 + eval slices), **F** (nanochat-on-Modal infra +
conversion proof), **G** (Model C v0, standard-Llama d12), and **J** (corpus v0.2 acquisition)
are done. Runbooks:
[../docs/phase-0.md](../docs/phase-0.md), [../docs/phase-1.md](../docs/phase-1.md); overall plan
[../docs/plan.md](../docs/plan.md).

## Setup

```bash
cd ml
uv sync          # create .venv and install dependencies from uv.lock
```

## Layout

```
corpus/      # Workstreams A + E — ingest → … → corpus v0; junk-filter → v0.1 + eval slices
tokenizer/   # Workstream B — byte-level BPE sweep, whole-word survival eval, export
fertility/   # Workstream C — tokens-per-content across ~8 tokenizers → CSV + chart
probe/       # Workstream D — zero-shot base-model probe (Modal app)
train/       # Workstream F — nanochat-on-Modal training infra + conversion proof
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
- **`probe/`** — Workstream D: base-model probe (Modal, bf16). BPB + few-shot MT scorecard
  over FLORES+ dev for 5 candidate bases; leader **`google/gemma-3-4b-pt`**. See
  [probe/README.md](probe/README.md) and [reports/base_model_probe.md](reports/base_model_probe.md).
- **`corpus/` (Workstream E)** — corpus **v0.1** (drop-only junk filter, −3,508 docs / −6.3%
  o200k tokens) + standing **eval slices** (`authored_eval`, `translation_shaped_eval`); prompt
  list **drafted** (pending human sign-off). See [reports/corpus_v0_1.md](reports/corpus_v0_1.md).
- **`train/`** — Workstream F: nanochat @ pinned commit on Modal H100 with kreyol-bpe swapped in.
  Vocab plumbs (no padding); loss↓ + checkpoint resume across Modal calls; **~469k tok/s** d12.
  Conversion chain **breaks** (custom pre-tokenizer + non-Llama arch) — the de-risking finding.
  See [train/README.md](train/README.md) and [reports/train_smoke.md](reports/train_smoke.md).
- **`train/` (Workstream G)** — **Model C v0**: a standard HF `LlamaForCausalLM` (the arch swap
  resolves F's conversion blocker). Depth sweep → **d12 (123M)**; F2 conversion gates all pass
  (GGUF/Ollama/ONNX, token-ID parity 1.000, `kreyol-bpe` pre-tokenizer registered in llama.cpp);
  flagship to 750M tokens **beats 3–4B Gemma/Llama/Qwen on 3 of 4 Kreyòl BPB slices**. See
  [reports/modelc_v0.md](reports/modelc_v0.md), [reports/depth_sweep.md](reports/depth_sweep.md),
  [reports/f2_gates.md](reports/f2_gates.md).
- **`corpus/` (Workstream J)** — corpus **v0.2** (measure-first). J0 scoping tripped the red-flag
  gate — **VOA net-new ≈2.25M, not 20–60M** (54% of URLs are audio/video, 34% wire, 45% MADLAD-dup);
  fineweb-2 was the opposite surprise (**~108M net-new**, not "heavy overlap"). Built under the
  "full v0.2 + mix control" decision: v0.1 frozen base + register-tagged net-new (VOA journalism,
  33 federal PD PDFs, Bib La/Konstitisyon/Storybooks, fineweb-2), authored-beats-crawl dedup,
  authored emphasis via **train-mix weights**. Committable: **1,955 federal EN↔HT glossary pairs**
  (`corpus/glossary_pairs_federal.json`) + a contributor record. New eval axes: `authored_eval_v2`
  (VOA temporal) + terminology probes. See [reports/corpus_v0_2_scoping.md](reports/corpus_v0_2_scoping.md),
  [reports/corpus_v0_2.md](reports/corpus_v0_2.md).
