# corpus/ — Workstreams 0 + A (+ Phase-1 E)

Rights registries + the **corpus v0** pipeline: ingest → normalize → filter → dedup →
audit → report. Rights-clear sources only; provenance-tagged JSONL + a stats report with a
quality audit. Protocol: [../../docs/phase-0.md](../../docs/phase-0.md) (Workstreams 0 + A),
[../../docs/plan.md](../../docs/plan.md) §5.2.

Phase-1 **Workstream E** layers **corpus v0.1** on top (a drop-only junk-filter stage) and the
**standing eval slices** — see the modules + run below and
[../reports/corpus_v0_1.md](../reports/corpus_v0_1.md).

## Run

```bash
cd ml
uv sync
# HF_TOKEN read from the repo-root .env (MADLAD is public; token is optional).
uv run python -m corpus.run --sample   # 1% smoke test, end to end (run this first)
uv run python -m corpus.run            # full build + audit + report
```

Each stage is also runnable alone: `python -m corpus.<stage> [--sample]`. Raw downloads land
immutable in `data/raw/downloads/`; every stage reads them and is re-runnable. Interim JSONL
lives under `data/interim/<sample|full>/<stage>/`; the final corpus is `data/clean/corpus_v0-<tag>.jsonl`.

## Workstream 0 — registries (fixed before ingestion)

| file | role |
|---|---|
| `rights.yaml` | per-source rights matrix (analysis / tokenizer / model / redistribution), license ids/urls, and the MADLAD CC-BY-vs-ODC-BY discrepancy as an explicit unresolved item |
| `splits.yaml` | split registry (train / tokenizer_eval / model_selection_dev / final_devtest / exhibit_examples); FLORES+/CreoleVal excluded from training; proverbs teachable-vs-probe rule |
| `schema.py` | Pydantic §5.2 document metadata (origin, genre, acquisition anchors, rights, split); every record is validated against it |

## Workstream A — pipeline modules

| file | role |
|---|---|
| `config.py` | pins (MADLAD revision, htwiki dump date), filter thresholds, dedup params, audit plan, reference tokenizer |
| `common.py` | IO, hashing, deterministic sampler, size units, rights lookup, source downloads |
| `ingest.py` | MADLAD gz + htwiki bz2→wikitext→plaintext; full §5.2 metadata; deterministic 1% sampler |
| `normalize.py` | NFC + whitespace; **preserves paragraph boundaries**; no lowercasing/accent-stripping |
| `filter.py` | source-specific thresholds + boilerplate strip; **flags** Wikipedia bot-stubs (keeps them) |
| `dedup.py` | document exact + MinHash near-dup + paragraph exact-dup, across sources; survivor priority owned > wikipedia > crawl; writes a duplicate map |
| `audit.py` | deterministic stratified sample; fasttext lid.176 + boilerplate/unreadable heuristics; machine first pass |
| `report.py` | `../reports/corpus_v0.md` (sizes ×4 units, dedup rates, bot-stub share, audit, caveats) |
| `run.py` | orchestrator (`python -m corpus.run [--sample]`) |

## Workstream E — corpus v0.1 + eval slices (Phase 1)

A **new stage** on the frozen v0 shard (v0 stays byte-for-byte reproducible; v0.1 is a strict
drop-only document subset). Run: `uv run python -m corpus.run_v0_1 [--sample]`.

| file | role |
|---|---|
| `config_v0_1.py` | junk thresholds (calibrated to 0 natural false-positives on the audit labels), spam markers, eval-slice caps/seed, prompt-draft path — kept separate from the frozen phase-0 `config.py` |
| `junk.py` | 5 deterministic junk filters (`mt_placeholder`/`commercial_spam`/`price_listing`/`html_entity`/`foreign_script`) → `corpus_v0_1-<tag>.jsonl` + stats; first-trip attribution, 4-unit deltas |
| `evalslices.py` | `authored_eval` + `translation_shaped_eval` from human-verified audit labels (tier A) + conservative heuristics (tier B); git-ignored doc_id manifests under `data/eval/` |
| `report_v0_1.py` | `../reports/corpus_v0_1.md` (deltas, per-filter counts, calibration, slice sizes, prompt draft) |
| `checkpoint_prompts_DRAFT.json` | ~10 exhibit prompts — **DRAFT, pending human sign-off, not frozen**; no probe-split proverb text |
| `run_v0_1.py` | orchestrator (junk → evalslices → report) |

## Outputs

- `data/clean/corpus_v0-<tag>.jsonl` — provenance-tagged corpus (git-ignored: `*.jsonl`).
- `../reports/corpus_v0.md` — the stats report (Station 5 nutrition-label raw material).
- `../reports/audit_sample_review.md` — committable audit review sheet (MADLAD snippets redacted).
- `../reports/audit_sample.jsonl` — full audit sample incl. text (git-ignored).

## Data & licensing

Rights-clear sources only: **MADLAD-400 ht clean** (redistribution UNRESOLVED — see `rights.yaml`)
and **ht Wikipedia** (CC-BY-SA). Quarantined sources (MIT-Haiti, dictionary) and eval-only
**FLORES+** are never ingested. No corpus text is committed — `data/` and every `*.jsonl` are
git-ignored; only aggregate reports are committed. MADLAD text is not even excerpted in committed
files until its license resolves.
