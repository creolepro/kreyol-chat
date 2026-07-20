# Phase 0 — Corpus v0, Kreyòl Tokenizer, Fertility Numbers, Base-Model Probe

**Parent plan:** [plan.md](./plan.md) (§3.3 exp 1, §5, §11)
**Budget:** ≈ $0–10 compute (Claude API counting is free; probe runs ≈ $5–10 on Modal)
**Hardware:** laptop CPU for everything except Workstream D (one small Modal GPU)
**Timebox:** 1–2 weekends

## Goals & deliverables

| # | Deliverable | Feeds |
|---|---|---|
| 1 | **Corpus v0** — deduplicated, provenance-tagged JSONL (~90–100M tokens) + a stats report | Model C pretraining (Phase 1), tokenizer training, Station 5 nutrition label |
| 2 | **Tokenizer v0** — byte-level BPE, vocab size chosen from a sweep | Model C (Phase 1), Station 1 |
| 3 | **Fertility report** — tokens-per-content ratios for Kreyòl vs English vs French across ~8 tokenizers, incl. the first measured Claude/o200k/Llama-3/Gemma numbers for HT | Publishable writeup, Station 1, Twitter series |
| 4 | **Base-model probe** — zero-shot Kreyòl scores for Qwen3/SmolLM3/Gemma-3(/Llama-3.2) | The Model B base-model decision (Phase 2) |

Success = we can answer three questions with our own numbers: *how much usable Kreyòl text exists (really)*, *what does the token tax measure on modern tokenizers*, and *which small model starts from the strongest Kreyòl baseline*.

---

## Setup

**Where the code lives:** `ml/` at the repo root — the Python workspace for all model/corpus/tokenizer work. Docs, code, and provenance live together in one repo; the data dir is git-ignored.

```
ml/
├── pyproject.toml          # uv-managed; python ≥3.12
├── data/                   # git-ignored
│   ├── raw/                # downloaded sources, untouched
│   ├── clean/              # corpus v0 JSONL shards
│   └── eval/               # flores_plus, proverbs, authored set
├── corpus/                 # Workstream A scripts
├── tokenizer/              # Workstream B scripts + trained tokenizer.json
├── fertility/              # Workstream C script + results CSV
├── probe/                  # Workstream D (Modal app)
└── reports/                # generated stats/markdown
```

**Deps:** `datasets`, `tokenizers`, `tiktoken`, `transformers`, `sacrebleu`, `anthropic`, `modal`, `datasketch` (near-dup), `matplotlib`.

**Accounts/gates to clear on day 1 (approval can lag — do these first):**
- HF account + accept terms for: `openlanguagedata/flores_plus`, `meta-llama/Llama-3.1-8B` (tokenizer only), `google/gemma-3-4b-it`. (Qwen3, SmolLM3, NLLB, MADLAD are ungated.)
- `ANTHROPIC_API_KEY` for `count_tokens`; Modal account (free $30/mo credits cover all of Phase 0).

---

## Workstream A — Corpus v0

### A1. Sources (clean build only — noisy MADLAD deferred to a Phase 1 experiment)

| Source | Est. tokens | Provenance tag | How |
|---|---|---|---|
| MADLAD-400 ht **clean** | 84.3M | `web_crawl` | HF `allenai/MADLAD-400`, language config `ht` (verify exact config/split names at impl time) |
| ht Wikipedia | ~10M | `authored_kreyol` (encyclopedic) | Fresh dump from [dumps.wikimedia.org/htwiki](https://dumps.wikimedia.org/htwiki/) + extractor; fallback: HF `wikimedia/wikipedia` dated `…ht` config |
| CreolePro dictionary | small | `dictionary` | Optional, private source — not in this repo (KV store export) |
| MIT-Haiti sentences (CreoleVal MT dir) | ~120k words | `authored_kreyol` (education) | [github.com/hclent/CreoleVal](https://github.com/hclent/CreoleVal) — **check per-file license before use** |
| Proverbs list (Jaden Lakou set + additions) | tiny | `oral_tradition` | Already ours |

Held out, never trained on: FLORES+ (eval only), a 1–2k-sentence authored-Kreyòl held-out slice for perplexity/bits-per-byte.

### A2. Pipeline (one script per stage, JSONL in/out)

1. **Ingest** → `data/raw/`, one JSONL per source, every doc wrapped with metadata: `{text, source, url?, provenance, genre, license, date_collected}` (schema = main doc §5.2).
2. **Normalize** — NFC Unicode (è/ò consistency), strip control chars, collapse whitespace. *No* lowercasing, *no* accent stripping.
3. **Filter** — min length (≥ ~20 chars), max symbol/digit ratio, boilerplate lines (nav/cookie strings), optional langid spot-check on a sample (MADLAD-clean is already filtered; don't over-engineer).
4. **Dedup** — exact (hash normalized text) then near-dup (MinHash/LSH over 5-gram shingles, threshold ~0.8). Dedup **across** sources — Wikipedia text appears inside MADLAD.
5. **Report** — `reports/corpus_v0.md`: docs/tokens by source, by provenance tag, dedup removal rates, top domains in the crawl slice, sample docs per source. **This report is the raw material for Station 5's nutrition label — write it once, use it twice.**

Output: `data/clean/corpus_v0-*.jsonl` + the report. Expect ~90–100M tokens post-dedup.

**Gotchas:** MADLAD loading may need `trust_remote_code`/specific data-file paths — check the card. Wikipedia's 72k articles are heavily bot-stub — that's fine (stubs are still valid Kreyòl), but note the genre skew in the report. Keep raw downloads immutable; all cleaning is re-runnable downstream.

---

## Workstream B — Tokenizer v0

**Tool for Phase 0: HF `tokenizers`** (ByteLevel BPE — same family as GPT-2/4 tokenizers, robust to accents/code-switching). Model C will retrain with nanochat's rustbpe at Phase 1 for zero integration friction; identical algorithm/vocab-size settings makes v0 the dress rehearsal, and v0 remains the analysis/Station-1 artifact.

1. **Train the sweep**: vocab ∈ {8k, 16k, 24k, 32k} on corpus v0 (train split only). Byte-level, NFC pre-normalized input, special tokens reserved (`<|bos|>` etc. — match nanochat's expectations).
2. **Evaluate each size** on the held-out slice:
   - fertility (tokens/word, tokens/byte);
   - **whole-word survival** on the core list — TMA markers (`te, ap, pral, ta, a va` combos), pronouns (`mwen/m, ou/w, li/l, nou/n, yo/y`), determiners (`la, a, an, lan, nan, yo`), negation (`pa`), the top-500 corpus words, proverb vocabulary — each checked word-initial and mid-sentence (leading-space variant);
   - % of proverbs that tokenize with every word intact.
3. **Pick by elbow**: smallest vocab where survival/fertility gains flatten (expect 16–24k; §3.2's parameter-budget math is why we don't just take the biggest).
4. **Export**: `tokenizer.json` + a plain JSON `{vocab, merges}` for the browser (Station 1).

Cheap ablation worth 30 minutes (exhibit content): train one 16k tokenizer on *English* text and tokenize the proverbs with both — the side-by-side of which words shatter is Station 1's story in one image.

---

## Workstream C — Fertility measurement (the novel numbers)

Protocol summary (full rationale in main doc §3.3):

1. **Data**: FLORES+ `hat_Latn / eng_Latn / fra_Latn` devtest (1,012 aligned sentences) + the authored set (proverbs + MIT-Haiti sample) as a translated-vs-authored check.
2. **Tokenizers**: cl100k + o200k (`tiktoken`); Llama-3, Gemma-3, Qwen3, NLLB, SmolLM3 (HF `AutoTokenizer`); tokenizer v0 (ours); **Claude via `client.messages.count_tokens`** — free endpoint; send each language's corpus concatenated in a handful of large messages so wrapper overhead is negligible; record the model ID counted against.
3. **Metrics per tokenizer**: parity ratio vs English *and* vs French; tokens/word; whole-word survival (same core list as B2); sentences-per-8k-budget; $ premium at current API prices.
4. **Sanity gate**: cl100k parity must reproduce Petrov's **1.74×** (±0.02 — their FLORES-200 slice differs slightly from FLORES+; if outside, reconcile before trusting anything else).
5. **Outputs**: `fertility/results.csv`, a bar chart (parity ratio per tokenizer), and a 1-page `reports/fertility.md`. Publish script+CSV alongside any public writeup.

Shareable milestones hiding in here: first-ever Claude/GPT-4o/Llama-3/Gemma fertility numbers for Haitian Creole; the vs-French (lexifier) comparison; our tokenizer at ~1.0× by construction.

---

## Workstream D — Base-model zero-shot probe

**Question:** which base starts from the strongest Kreyòl position for Model B? Reputation doesn't settle this (none publish HT evals) — ~$5–10 of GPU time does.

- **Candidates**: Qwen3-1.7B, Qwen3-4B, SmolLM3-3B, Gemma-3-4B, Llama-3.2-3B (control/floor).
- **Harness**: small Modal app, bf16 `transformers` generation on one L40S/A100-40 (eval in full precision, not quantized — we're comparing models, not deployment). Batch all candidates in one script; cache model weights in a Modal volume.
- **Tasks**:
  1. **MT both directions** on FLORES+ devtest (or a 300-sentence slice to keep runtime down): eng→hat and hat→eng, 0-shot and 5-shot, greedy. Score spBLEU + chrF2++ (`sacrebleu`) — same metrics as the Robinson table so our numbers slot next to ChatGPT 47.0 / Google 53.4.
  2. **Bits-per-byte** on the authored-Kreyòl held-out slice — the cross-tokenizer-comparable version of perplexity (per-token perplexity is *not* comparable across different vocabularies).
  3. **Proverb completion**, ~15 items, 0-shot ("Piti piti, …") — scored manually; doubles as Station 2 preview material.
  4. **Qualitative sniff test**: 10 fixed prompts (greeting, simple question, short translation, register shift), outputs into the report for native-speaker eyeballing.
- **Decision rule**: rank by chrF2++ (eng→hat) primary, bits-per-byte secondary, qualitative veto. Record the pick + numbers in `reports/base_model_probe.md`. If everything is dismal (possible!), that's a finding too — it strengthens the CPT story and the exhibit narrative.

---

## Checklist (suggested order)

- [ ] Day 1: clear HF gates, keys, `ml/` scaffold, download MADLAD ht clean + Wikipedia dump *(gates approve while you build)*
- [ ] A: ingest → normalize → filter → dedup → corpus report
- [ ] B: tokenizer sweep → survival eval → pick vocab → export browser JSON
- [ ] C: fertility script (cl100k sanity gate first) → full tokenizer table → CSV/chart/report
- [ ] D: Modal probe harness → run candidates → decision report
- [ ] Wrap: commit reports; update main doc §3.1 (chosen base) + §5.1 (measured corpus counts); queue Twitter material

**Definition of done:** corpus v0 + report exists; tokenizer v0 chosen with survival evidence; fertility CSV reproduces Petrov and adds ≥4 novel tokenizer rows; Model B base picked with recorded numbers.

## Risks / open questions

- **License check before training on** MIT-Haiti/CreoleVal files (mixed per-file licenses) and the dictionary's Kreyòl-MT-derived entries (dataset license "other") — for *Phase 0 analysis* (tokenizing/counting) this is fine; flag for the Phase 1 training-data decision.
- FLORES+ hat is translated text — the authored set exists to catch fertility/probe skew from translationese.
- Claude `count_tokens` is rate-limited — batch large, backoff on 429.
- Exact HF config names (MADLAD splits, FLORES+ subsets) verified at implementation time — the card layouts drift.
