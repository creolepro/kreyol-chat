# train/ — Workstream F (training infra + conversion proof)

nanochat wired onto Modal with the **kreyol-bpe** tokenizer swapped in end-to-end, a
throwaway d12 training run (loss / checkpoint-resume / schedule), and the
deployment **conversion chain** proven (or its break reported precisely). Protocol:
[../../docs/phase-1.md](../../docs/phase-1.md) Workstream F, [../../docs/plan.md](../../docs/plan.md)
§3.2/§7.2/§7.3. Report: [../reports/train_smoke.md](../reports/train_smoke.md).

## Run

```bash
cd ml && uv sync
uv run python -m train.prepare     # local: build tokenizer bundle + corpus-v0.1 parquet
uv run python -m train.run         # Modal: verify vocab → train A → resume B → generate → convert
uv run python -m train.report      # write ../reports/train_smoke.md
# Workstream G data (whole corpus): uv run python -m train.prepare --full
```

Modal must be authenticated (`modal profile current`). No secrets needed — the corpus
is local and no gated models are used.

## Modules

| file | role |
|---|---|
| `config.py` | pins: nanochat commit (same as the tokenizer), Modal app/GPU(H100)/Volume, base-dir layout, smoke hyperparameters, the log-then-linear checkpoint schedule (G's token points) |
| `prepare.py` | local: rebuild nanochat's tokenizer bundle (tiktoken `Encoding` pkl + `token_bytes.pt`) from committed kreyol-bpe; materialize corpus v0.1 → parquet shards, **excluding** the eval slices + tokenizer_eval holdout (probe proverbs already absent) |
| `apply_savesteps.py` | minimal, auditable patch to nanochat `base_train.py` adding `--save-steps` (explicit step indices for the log-then-linear schedule); applied at image-build time |
| `modal_app.py` | Modal image (nanochat @ pinned commit + patch) + functions: `setup` (finalize bundle, verify vocab plumbing/padding), `train` (run base_train, parse loss/tok-s/bpb), `generate` (greedy Kreyòl), `convert_probe` (nanochat→HF→GGUF, report the break) |
| `run.py` | orchestrator: prepare → upload bundle to Volume → setup → train A → train B (resume in a fresh container) → generate → convert; writes the results JSON |
| `report.py` | `../reports/train_smoke.md` from the results JSON |

## What the smoke establishes

- **Vocab plumbing:** the kreyol-bpe 24,576 vocab (`kreyol_aware` pattern) loads through
  nanochat; 24,576 = 384×64 so nanochat's `pad_vocab_size_to=64` is a **no-op** (no
  kernel padding).
- **Training:** loss decreases; **checkpoint save + resume works across Modal function
  calls** (call B is a fresh container that loads state from the Volume); the
  log-then-linear checkpoint schedule is configurable via `--save-steps`.
- **Conversion (the de-risking finding):** whether the custom-vocab checkpoint survives
  nanochat → HF → `convert_hf_to_gguf.py` → llama.cpp and the browser export — reported
  precisely in the train_smoke report.

## Data & rights

The parquet shards, tokenizer bundle, and checkpoints live under `data/train_work/`
(git-ignored) and the Modal Volume — nothing under `ml/data/` is committed. The three
held-out sets (tokenizer_eval holdout, the two Workstream-E eval slices) are excluded
from the training shards; the 15 probe proverbs are absent from the corpus entirely.
