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

## Workstream G — Model C as a standard Llama

The Workstream-F finding was that nanochat's speedrun architecture (48% nanochat-only
params) has no llama.cpp graph and its `kreyol_aware` pre-tokenizer isn't registered. The
binding 2026-07-22 decision: **Model C is a real `transformers.LlamaForCausalLM`** (learned
RMSNorm, standard RoPE, causal attention, SwiGLU, ordinary residuals — none of the speedrun
features), width 768, depth from a sweep. Because we train the HF class, `save_pretrained`
is architecturally lossless and the conversion chain becomes tractable.

| file | role |
|---|---|
| `llama_config.py` | G arch spec (width 768, depths 12/16/20 → 123M/151M/179M), optimizer/schedule, Volume layout, llama.cpp pin (`67b9b0e7f6ce`), apostrophe/clitic parity fixtures |
| `llama_model.py` | torch-free `param_count()` + HF `LlamaConfig`/`LlamaForCausalLM` factory |
| `tokenize_g.py` | local: corpus v0.1 → uint16 `train.bin`/`val.bin` (excl. eval slices + tokenizer holdout), 4 BPB slice texts, ~1k parity probe, provenance nutrition label |
| `data_g.py` | seeded depth-invariant dataloader (vectorized gather) + cosine LR |
| `bpb_g.py` | BPB with the byte-identical policy of the Workstream-D base-model probe |
| `patch_llamacpp_cpp.py` | register the `kreyol-bpe` pre-tokenizer in llama.cpp source (enum + name→pre_type + pre_type→regex, greedy mirror of the possessive `kreyol_aware` pattern) |
| `gates.py` | F2 gates 1–6 (native↔HF logits / GGUF+llama.cpp / stock-Ollama / token-ID parity / ONNX browser / cross-runtime greedy + Q4) |
| `llama_app.py` | Modal app: `verify_params`, `train` (HF ckpt resume + per-ckpt gens+BPB), `generate`, `bpb`, `convert_gates`, `base_bpb`; image = torch + transformers + patched-and-built llama.cpp + stock Ollama |
| `g_run.py` | orchestrator subcommands: `upload`/`verify`/`gate`/`sweep`/`flagship`/`base-bpb` |
| `g_report.py` | `../reports/{f2_gates,depth_sweep,modelc_v0}.md` + the committable per-checkpoint generations JSON |

```bash
cd ml && uv run python -m train.tokenize_g     # local: bins + parity probe + nutrition
uv run python -m train.g_run upload            # push to the Volume
uv run python -m train.g_run verify            # param counts == torch-free calc
uv run python -m train.g_run gate              # Part 2: d16 throwaway → F2 gates
uv run python -m train.g_run sweep             # Part 3: d12/d16/d20 BPB
uv run python -m train.g_run flagship --depth <d>   # Part 4: the flagship run
uv run python -m train.g_run base-bpb          # base-model BPB on the same slices
uv run python -m train.g_report all
```

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
