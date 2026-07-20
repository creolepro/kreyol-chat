# Phase 0 — Rights, Corpus v0, Kreyòl Tokenizer, Fertility Numbers, Base-Model Probe

**Parent plan:** [plan.md](./plan.md) (§3.3 exp 1, §5.2, §11)
**Structure:** **Phase 0a** (rights → corpus → tokenizer → fertility; CPU-only, ≈$0) then **Phase 0b** (GPU base-model probe, ≈$5–10 on Modal). 0a is the 1–2-weekend target; 0b can trail it.

## Goals & deliverables

| #   | Deliverable                                                                                                                                     | Feeds                                                                        |
| --- | ----------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| 0   | **Rights matrix + split registry** — per-source permissions; train/eval/exhibit splits fixed before ingestion                                   | Everything downstream; the legal audit trail                                 |
| 1   | **Corpus v0** — deduplicated, provenance-tagged JSONL from rights-clear sources + a stats report with a quality audit                           | Model C pretraining (Phase 1), tokenizer training, Station 5 nutrition label |
| 2   | **Tokenizer v0** — byte-level BPE (nanochat rustbpe, pinned commit), vocab size chosen from a sweep                                             | Model C (Phase 1), Station 1                                                 |
| 3   | **Fertility report** — parity ratios for Kreyòl vs English vs French across ~8 tokenizers, with CIs; first *published* HT numbers we know of    | Publishable writeup, Station 1, Twitter series                               |
| 4   | **Base-model probe (0b)** — BPB + few-shot scorecard for the candidate *base* checkpoints on FLORES+ dev                                        | The Model B base-model decision (Phase 2)                                    |

Success = we can answer three questions with our own defensible numbers: _how much usable Kreyòl text exists (really)_, _what does the token tax measure on modern tokenizers_, and _which base checkpoint starts from the strongest Kreyòl position_.

---

## Setup

**Where the code lives:** `ml/` at the repo root — the Python workspace for all model/corpus/tokenizer work. Docs, code, and provenance live together in one repo; the data dir is git-ignored.

```
ml/
├── pyproject.toml          # uv-managed; python ≥3.12
├── data/                   # git-ignored
│   ├── raw/                # downloaded sources, untouched
│   ├── quarantine/         # rights-unresolved sources — never enters artifacts
│   ├── clean/              # corpus v0 JSONL shards
│   └── eval/               # flores_plus, proverbs, authored set
├── corpus/                 # Workstream A scripts (+ rights matrix, split registry)
├── tokenizer/              # Workstream B scripts + trained tokenizer.json
├── fertility/              # Workstream C script + results CSV
├── probe/                  # Workstream D (Modal app)
└── reports/                # generated stats/markdown
```

**Deps:** `datasets`, `tokenizers`, `tiktoken`, `transformers`, `sacrebleu`, `anthropic`, `modal`, `datasketch` (near-dup), `matplotlib`, `pydantic`.

**Accounts/gates to clear on day 1 (approval can lag — do these first):**

- HF account + accept terms for: `openlanguagedata/flores_plus`, a Llama 3.x repo (tokenizer only), `google/gemma-3-4b-pt`. (Qwen3, SmolLM3, NLLB, MADLAD are ungated.)
- `ANTHROPIC_API_KEY` for `count_tokens`; Modal account (free $30/mo credits cover all of Phase 0).

---

## Workstream 0 — Rights matrix & split registry (before any ingestion)

Two small files in `ml/corpus/`, written first, versioned forever:

**`rights.yaml`** — one entry per source: license ID + URL, and explicit allowed/denied/unresolved for each of: analysis, tokenizer-training, model-training, redistribution. Current state:

| Source                         | Analysis | Tokenizer-train | Model-train | Redistribute | Status                                                        |
| ------------------------------ | -------- | --------------- | ----------- | ------------ | ------------------------------------------------------------- |
| MADLAD-400 ht clean            | ✅       | ✅              | ✅          | ⚠️           | **Resolve CC-BY-4.0 (card prose) vs ODC-BY (repo metadata)** before publishing artifacts |
| ht Wikipedia (CC-BY-SA)        | ✅       | ✅              | ✅          | ✅ (SA)      | Clear                                                         |
| Owned proverbs/materials       | ✅       | ✅              | ✅          | ✅           | Clear                                                         |
| MIT-Haiti (CreoleVal MT dir)   | ✅       | ❌ quarantine   | ❌          | ❌           | Mixed per-file licenses — unresolved; also a CreoleVal *eval* set |
| CreolePro dictionary (private) | ✅       | ❌ quarantine   | ❌          | ❌           | Kreyòl-MT-derived, license "other" — unresolved               |
| FLORES+                        | eval only| ❌              | ❌ (terms)  | ❌           | Eval-only by its own terms; pin the dataset version           |

**`splits.yaml`** — the registry, fixed before ingestion: `train` / `tokenizer_eval` / `model_selection_dev` / `final_devtest` / `exhibit_examples`. Rules: all CreoleVal/FLORES content excluded from training; authored material split by document/collection, never randomly by sentence; a benchmark-contamination check (n-gram overlap vs FLORES+/CreoleVal) runs against the final training corpus. **Proverbs get a deliberate two-way split**: a *teachable* set (allowed into training — Station 2's "who knows the real proverb?" requires models to have seen proverbs) and a *held-out probe* set that never appears anywhere in training.

---

## Workstream A — Corpus v0 (Phase 0a)

### A1. Sources — rights-clear only (quarantined sources sit in `data/quarantine/` untouched)

| Source                  | Est. size            | origin / genre               | How                                                                                                                                                  |
| ----------------------- | -------------------- | ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| MADLAD-400 ht **clean** | 84.3M "tokens" (their count — whitespace-ish, *not* our BPE; treat all totals as estimates) | machine-mixed web / crawl | HF `allenai/MADLAD-400`, language config `ht` (verify exact config/split names at impl time) |
| ht Wikipedia            | ~10M words           | authored (encyclopedic; **bot-stubs flagged separately**, not blanket-labeled "authored") | Fresh dump from [dumps.wikimedia.org/htwiki](https://dumps.wikimedia.org/htwiki/) + extractor |
| Proverbs (teachable split only) | tiny         | oral tradition / proverb     | Already ours; probe split withheld per `splits.yaml`                                                                                                  |

Noisy MADLAD is deferred to a Phase 1 experiment. MIT-Haiti and the dictionary are quarantined until rights resolve (Workstream 0).

### A2. Pipeline (one script per stage, JSONL in/out, metadata per plan.md §5.2 schema)

1. **Ingest** → `data/raw/`, one JSONL per source; every doc gets the full §5.2 metadata: origin, genre, source + URL + download timestamp, stable doc ID, raw content hash, license ID, split assignment.
2. **Normalize** — NFC Unicode (è/ò consistency), strip control chars, normalize spaces *within* lines but **preserve paragraph boundaries** (structure is signal). _No_ lowercasing, _no_ accent stripping. Record cleaned-content hash.
3. **Filter** — **source-specific** thresholds (a 20-char floor would delete proverbs and conversational text; crawl text can afford stricter floors), max symbol/digit ratio, boilerplate lines. Flag Wikipedia bot-stubs via simple heuristics (template phrasing, length, edit history if cheap).
4. **Dedup** — exact (hash) then near-dup (MinHash/LSH, 5-gram shingles, ~0.8), at **both document and paragraph level**, **across** sources. Deterministic survivor priority: owned/authored > Wikipedia > crawl. Keep a **duplicate map** (cluster ID + why the survivor won) rather than silently discarding.
5. **Quality audit** — a deterministic, stratified sample (~200 docs across sources/length-bands), manually reviewed; report estimated rates of wrong-language, boilerplate, unreadable, and translation-shaped text. An hour of eyeballing, and the findings are exhibit content.
6. **Report** — `reports/corpus_v0.md`: docs and size by source/origin/genre reported in **bytes, chars, whitespace words, and tokens under a named reference tokenizer**; dedup removal rates; audit results; top crawl domains; sample docs. **This report is the raw material for Station 5's nutrition label.**

Output: `data/clean/corpus_v0-*.jsonl` + the report. The ~90–100M figure is an **estimate, not a completion criterion** — report what's actually there.

**Gotchas:** MADLAD loading may need `trust_remote_code`/specific data-file paths — check the card. Keep raw downloads immutable; all cleaning is re-runnable downstream.

---

## Workstream B — Tokenizer v0 (Phase 0a)

**Tool: nanochat's rustbpe, pinned to a specific commit.** Two byte-level BPE implementations are *not* interchangeable (pre-tokenization regex, prefix-space handling, merge ordering, serialization all differ) — using the real Phase 1 tokenizer now avoids a false dress rehearsal. Fallback if rustbpe integration fights back: HF `tokenizers`, explicitly demoted to analysis-only, with a parity test against rustbpe required before Phase 1.

1. **Training data**: a **deterministic sample with explicit source weighting** (seeded; weights recorded in the report) — otherwise a "Kreyòl tokenizer" is mostly a MADLAD-crawl tokenizer. Train split only.
2. **Train the sweep**: vocab ∈ {8k, 16k, 24k, 32k}; NFC-normalized input; special tokens reserved to match nanochat's expectations.
3. **Evaluate each size** on `tokenizer_eval`:
   - **Decision metrics**: held-out bytes-per-token (compression), out-of-domain Kreyòl compression, English/French/code-switching regression (don't cripple the languages Kreyòl coexists with), embedding-table parameter cost, round-trip correctness, robustness (accents, apostrophes, spacing, capitalization, names, numbers, spelling variation).
   - **Interpretability/exhibit metric** (secondary for the decision, primary for Station 1): whole-word survival on the core list — TMA markers (`te, ap, pral, ta, a va` combos), pronouns (`mwen/m, ou/w, li/l, nou/n, yo/y`), determiners (`la, a, an, lan, nan, yo`), negation (`pa`), top-500 corpus words, proverb vocabulary — each checked word-initial and with leading space.
4. **Pick** the smallest vocab where compression gains flatten (expect 16–24k; plan.md §3.2's parameter-budget math is why we don't just take the biggest).
5. **Export**: `tokenizer.json` + a plain JSON `{vocab, merges}` for the browser (Station 1).

Cheap ablation worth 30 minutes (exhibit content): train one 16k tokenizer on _English_ text and tokenize the proverbs with both — the side-by-side of which words shatter is Station 1's story in one image.

---

## Workstream C — Fertility measurement (Phase 0a; the novel numbers)

Protocol summary (full rationale in plan.md §3.3):

1. **Data**: FLORES+ `hat_Latn / eng_Latn / fra_Latn` devtest, **version pinned, joined by (split, id)** (~1,012 aligned sentences) + the authored set (probe-split proverbs; MIT-Haiti only if rights resolve) as a translated-vs-authored check. FLORES+ terms: eval-only, don't re-host.
2. **Tokenizers**: cl100k + o200k via `tiktoken` (labeled as tokenizers — o200k is "the tokenizer GPT-4o-era models use," not "GPT-4o"); Llama-3, Gemma-3, Qwen3, NLLB, SmolLM3 via HF `AutoTokenizer` (record revisions); tokenizer v0 (ours).
3. **Claude — separately labeled API measurement**: `client.messages.count_tokens` returns an *estimate* that may include request scaffolding — measure empty-request overhead and subtract, pin the exact model ID, preserve the request format, and label the result **"Claude API input parity for model X"**, not a raw tokenizer count. Batch large (~200 sentences/message), backoff on 429.
4. **Metrics per tokenizer**: primary parity = **sum(ht) ÷ sum(eng)** (and ÷ fra) over the corpus, with paired-bootstrap CIs and per-sentence quantiles; tokens/word under a defined segmentation rule (apostrophe-clitic handling like `m'ap` documented up front); whole-word survival (core list from B3); sentences-per-8k-budget; $ premium at **date-stamped** prices, only for tokenizers attached to actually-priced APIs.
5. **Pipeline validation**: first **replicate Petrov's 1.74× on his released data** ([repo CSV](https://github.com/aleksandarpetrov/tokenization-fairness), same tokenizer version) — that validates our code. Then run FLORES+ as our own measurement; expected close to his, but not gated to an arbitrary tolerance.
6. **Outputs**: `fertility/results.csv`, parity bar chart, and `reports/fertility.md` with methodology, dataset/tool/price snapshot dates, and claims phrased as "first *published* numbers we could find, as of <date>."

Shareable milestones hiding in here: first published Claude/o200k/Llama-3/Gemma parity numbers for Haitian Creole; the vs-French (lexifier) comparison; our tokenizer flipping the tax (ht/en parity likely below 1).

---

## Workstream D — Base-model probe (Phase 0b)

**Question:** which **base checkpoint** starts from the strongest Kreyòl position for Model B's CPT? Reputation doesn't settle this (none publish HT evals) — ~$5–10 of GPU time does.

- **Candidates (pretrained bases, not instruct variants)**: `Qwen/Qwen3-1.7B-Base`, `Qwen/Qwen3-4B-Base`, `HuggingFaceTB/SmolLM3-3B-Base`, `google/gemma-3-4b-pt`, `meta-llama/Llama-3.2-3B` (control/floor). Record exact HF revisions.
- **Harness**: small Modal app, unquantized bf16 `transformers` on one L40S/A100-40. Weights cached in a Modal volume. Record prompts, decoding settings, stop conditions, and the sacreBLEU signature.
- **Selection data: FLORES+ `dev` — `final_devtest` is reserved** for final reporting on the eventual chosen model(s).
- **Staged funnel** (cost + attention control):
  1. Smoke test: 20 examples per model — catch formatting/decoding breakage.
  2. Fixed 200–300-item dev subset, every model.
  3. Full dev on the top two.
- **Measures** (base models don't reliably follow zero-shot instructions — these are completion-style):
  1. **Bits-per-byte** on the authored-Kreyòl held-out slice — the cross-tokenizer-comparable LM measure and the primary signal of Kreyòl exposure.
  2. **Standardized few-shot MT completion** (5-shot, fixed template, greedy), eng→hat and hat→eng, scored spBLEU + chrF2++. Contextualize against the Robinson 2023 numbers but **don't claim direct comparability** (different prompts/pipeline/dataset version).
  3. **Few-shot proverb completion** (probe split, ~15 items) — doubles as Station 2 preview material.
  4. **Blinded naturalness mini-review**: 10 fixed prompts, outputs shuffled and de-identified, scored on a 3-point rubric (unusable / degraded-but-Kreyòl / plausible Kreyòl) by a fluent speaker.
- **Decision scorecard**: BPB (primary), few-shot chrF2++ eng→hat, naturalness rubric — recorded per model in `reports/base_model_probe.md` with the pick and its numbers. (A separate, optional instruct-variant probe can preview end-user behavior, but it doesn't drive the CPT base decision.) If everything is dismal, that's a finding — it strengthens the CPT story and the exhibit narrative.

---

## Checklist (order matters)

**Phase 0a:**

- Day 1: clear HF gates + keys *(approvals lag — everything below proceeds while waiting)*
- 0: write `rights.yaml` + `splits.yaml`; resolve or explicitly defer the MADLAD license question
- A (1% first): run the pipeline end-to-end on a 1% sample; fix schema/filters; then full build → quality audit → corpus report
- B: tokenizer sweep (rustbpe, pinned) → compression + survival eval → pick vocab → export browser JSON
- C: Petrov replication on his data → FLORES+ measurement → Claude API measurement (separately labeled) → CSV/chart/report

**Phase 0b:**

- D: Modal probe harness → smoke test → staged funnel → scorecard → decision report

**Wrap:** commit reports; update plan.md §3.1 (chosen base) + §5.1 (measured corpus counts); queue Twitter material.

**Definition of done:** rights matrix + split registry committed; corpus v0 + audited report exists; tokenizer v0 chosen on compression evidence with survival documented; fertility CSV replicates Petrov on his data and adds ≥4 new tokenizer rows with CIs and snapshot dates; Model B base picked via the scorecard with recorded revisions.

## Risks / open questions

- **MADLAD license discrepancy** (CC-BY-4.0 card prose vs ODC-BY repo metadata) must be resolved before publishing any artifact built on it — tracked in `rights.yaml`.
- Quarantine discipline: MIT-Haiti and the dictionary stay out of *all* artifacts (tokenizer included) until rights resolve; analysis/counting on them is fine.
- FLORES+ hat is translated text — the authored set exists to catch fertility/probe skew from translationese.
- rustbpe integration effort is unproven — the HF-tokenizers fallback exists, but only with an explicit parity test before Phase 1.
- Exact HF config names (MADLAD splits, FLORES+ subsets) verified at implementation time — card layouts drift.
