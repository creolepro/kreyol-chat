# Phase 1 — Training Model C (Kreyòl-first, from scratch)

**Parent plan:** [plan.md](./plan.md) §3.2 (Model C), §3.3 (experiments), §5.3 (synthetic policy), §11
**Inherits from Phase 0 (frozen):** corpus v0 + splits/rights registries (Workstreams 0/A) · kreyol-bpe tokenizer, 24,576 vocab, `kreyol_aware` pre-tokenization, pinned nanochat `92d63d4e8bb4` (Workstream B) · fertility baselines (C) · base-model scorecard as the comparison anchor (D).
**Budget:** ≈ **$15–40 GPU total** on Modal (within the $30/mo free credits). Compute is trivial; the scarce resources are the fixed-prompt list (needs the human) and native review for SFT data.
**Does not gate / not gated by:** Workstream D closing (D gates Phase 2/Model B, not this).

## Goals & deliverables

| # | Deliverable | Feeds |
|---|---|---|
| E | **Corpus v0.1 + labeled eval slices** — audit-driven junk filters; authored vs translation-shaped holdout slices; the frozen checkpoint prompt list | G, H, Station 5 nutrition label |
| F | **Training infra + conversion proof** — nanochat-on-Modal smoke run; measured tok/s; HF→GGUF→llama.cpp and browser export validated end-to-end | G, H, §7.3 event serving |
| G | **Model C v0/v1 (d12, 203M)** — the flagship run: multi-epoch, Pythia-style checkpoints, per-checkpoint generations, BPB learning curves, model card + nutrition label v1 | Stations 2/5/6 + slider, kreyol.chat, Phase 3 eval |
| H | **Micro-model fleet results** — the controlled experiments (Q1–Q5 below; Q6 deferred) | G's final config, Stations 1/5, publishable writeups |
| I | **Model C chat** — midtraining + SFT; the first conversational Kreyòl-first model | Station 2, kreyol.chat |

Success = a from-scratch model whose every behavior traces to data we can point at; checkpoint assets that show Kreyòl being learned; and causal answers (not vibes) for the tokenizer and corpus-mixture questions.

---

## Workstream E — Corpus v0.1 + eval slices (CPU, ≈$0)

> **Status 2026-07-21: done** ([report](../ml/reports/corpus_v0_1.md)). **Corpus v0.1** is a drop-only stage on the frozen v0 shard (`corpus/junk.py`): 5 deterministic MT/CMS junk fingerprints remove **3,508 docs = 2.44% of docs / 6.30% of o200k tokens** (9.55M tokens) → **140,432 docs / 142.1M o200k tokens**; every survivor byte-identical to v0. Filters (`price_listing` 2,160 · `commercial_spam` 702 · `mt_placeholder` 624 · `html_entity` 19 · `foreign_script` 3) were **calibrated against the 200 human-verified audit labels → 0/42 natural false-positives**. **Honest finding:** the audit's ~17% crawl junk is mostly *semantic* (fluent machine-translated commercial text, no mechanical tell), so deterministic fingerprints are high-precision / low-recall by design; the residual translationese stays in and is *measured* (below), with harder filtering left to fleet Q3 rather than assumed. **Eval slices** (`corpus/evalslices.py`, in `splits.yaml`, never trained): **`authored_eval`** 357 docs / 278.7k tok (57 human-verified natural + 300 seeded non-stub Wikipedia) and **`translation_shaped_eval`** 400 docs / 780.6k tok (100 human-verified translated + 300 seeded residual-fingerprint crawl) — tiers labelled in the manifests. **Checkpoint prompt list drafted** (`corpus/checkpoint_prompts_DRAFT.json`, 10 prompts) — **pending human sign-off, not frozen**; the proverb slot is a placeholder (no probe-split proverb committed to any prompt set). Bot-stubs stay **in**, flagged.

The audit ([summary](../ml/reports/audit_model_summary.md)) pays for itself here.

1. **Junk filters** from the audit's MT/CMS fingerprints: `&#39;`-style HTML entities, `XNUMX`/`XNMX` placeholders, untranslated-English-fragment density, plus the SEO/gambling patterns it identified. Deterministic, documented, applied as a *new pipeline stage* (raw data untouched; v0 remains reproducible). Expect to remove much of the crawl's ~17% junk. Report the delta in all four units.
2. **Labeled eval slices** (the audit's second dividend): from the 200 human-verified labels + cheap heuristics, build two additional held-out slices — `authored_eval` (native-voice docs: audit-natural crawl, non-stub Wikipedia, owned) and `translation_shaped_eval`. These become the standing quality axes every model (fleet, d12, Model B later) is measured on. Registered in `splits.yaml`; never trained on.
3. **Freeze the checkpoint prompt list** (exhibit dependency — needs the human's sign-off): ~10 fixed prompts run at every checkpoint forever — greeting, proverb completion (probe split), short eng→hat translation, a question, a continuation, register shift. **This list defines what the "watch it learn" slider shows; choose it for the exhibit, not for the metrics.**
4. **Stub policy:** bot-stubs stay in v0.1 (flagged) — their fate is decided by fleet Q4, not assumed.

## Workstream F — Training infra + conversion proof (smoke, ≈$1–3)

One-time friction, eaten deliberately on a throwaway run:

1. nanochat (pinned commit) wired into a Modal app: image build, corpus shards → Volume, kreyol-bpe swapped in (24,576 vocab — verify nanochat's vocab-size plumbing and kernel padding behave; the tokenizer's `meta.json` pattern must be the one used end-to-end).
2. Train a **short throwaway run** (micro depth or a few hundred d12 steps). Verify: checkpoint save/resume across Modal function calls, the log-then-linear checkpoint schedule config, loss logging.
3. **Conversion proof, same session:** checkpoint → HF format → `convert_hf_to_gguf.py` → **runs in llama.cpp** (greedy-generate a Kreyòl sentence) → browser-format export loads. If any link breaks with our custom vocab, we learn it on a $2 run. This de-risks §7.3's entire local-first event architecture.
4. **Measured tok/s** on the target GPU → replace the estimated cost table below (plan §7.2's costs-as-hypothesis rule).

## Workstream G — Model C flagship (d12, ≈$4–10/run)

May run **twice**: `v0` (natural-mix defaults — validates end-to-end and produces first slider assets) and `v1` (fleet-informed mixture/epochs/curriculum). Budget allows both; don't block v0 on the fleet.

- **Config:** d12 (203M), single H100 (multi-GPU complexity not earned at ~1h wall-clock), multi-epoch to ~0.6–1.2B effective tokens (epoch count from fleet Q5 for v1; default ~4–6 for v0), two-stage curriculum for v1 (broad mix → authored-upweighted tail, per plan §3.1's Sailor2 logic + fleet Q2).
- **Checkpoints:** log-then-linear token schedule (≈0 / 10M / 25M / 50M / 100M / 250M / 500M / 1B effective), weights kept (small — ~10 checkpoints ≈ a few GB in the Volume).
- **At every checkpoint:** (a) the frozen prompt list generated and archived as JSON — the slider asset; (b) BPB on all holdout slices (general / authored / translation-shaped / FLORES hat) — the learning curves.
- **At the end:** BPB compared against the Workstream D base-model scorecard **on the same slices** — the 203M Kreyòl-first model vs the 3–4B multilingual bases is the David-vs-Goliath chart, whichever way it comes out.
- **Model card + nutrition label v1** generated from provenance: exact composition by origin/genre/source, epochs, tokenizer, known gaps (the Station 5 artifact, produced by the training run itself).

## Workstream H — Micro-model fleet (≈$10–30 total)

Identical-twin experiments at ~20–30M params (smallest viable nanochat depth), ~100–300M tokens each (near-Chinchilla at this size — results not confounded by undertraining). All measured on the same E-slices; key comparisons run **2 seeds**. Each question has a decision attached:

| Q | Question | Setup | Decides |
|---|---|---|---|
| 1 | Does the Kreyòl vocabulary improve **learning**, not just cost? | kreyol-bpe vs the English-24k ablation tokenizer; identical data/order/steps; BPB (2 seeds) | Station 1's claim strength; the tokenizer thesis, causally |
| 2 | Does authored-upweighting shift the model's **voice**? | natural vs authored-upweighted mix; BPB on authored vs translation-shaped slices + fixed prompts | G-v1's late-curriculum stage; Station 5 demo |
| 3 | Quality vs quantity: does junk-filtering win at fixed compute? | v0 vs v0.1 corpus | How aggressive E's filters should get |
| 4 | Bot-stubs: food or filler? | stubs-in vs stubs-out | Corpus policy for v0.2+ |
| 5 | How far does repetition stretch **our** corpus? | 4 vs 8 vs 12 epochs, fixed unique data | G's token schedule (currently an extrapolated assumption) |
| 6 | Synthetic: none vs unreviewed vs reviewed | *(deferred to the synthetic-data phase — spec'd here so the eval slices anticipate it)* | §5.3 ceilings from our own data; the scaling-ladder policy |

## Workstream I — Midtraining + SFT (≈$2–5)

Turns the text-continuer into something a visitor can talk to.

1. **Midtraining** (format mechanics, bulk-but-structured data): nanochat chat template/special tokens; translation-task turns. ⚠️ **Rights constraint:** Kreyòl-MT parallel data is license-"other" (quarantined for model training per `rights.yaml`); FLORES is eval-only. Translation turns must come from rights-clear pairs — the practical path is **synthetic pairs via the CreoleMT API over openly-licensed English source text**, provenance-tagged `synthetic_unreviewed`/`_reviewed` per §5.3 (real-data-majority anchor still holds: midtraining is small next to pretraining). Resolve-or-route-around before building.
2. **SFT** (~1–10k excellent examples, loss masked to responses): reviewed Aya-haitian sample + self-instruct seeded on native corpus text (MURI-style) + proverb/dictionary QA. **Native-speaker review is mandatory** (plan §6.2) — this is a human-time dependency, not a compute one.
3. **Eval:** the frozen prompt list (now answering, not continuing — the demoable transition), BPB regression check on the slices (chat tuning shouldn't damage the language model), blinded naturalness sheet for the human.

---

## Checklist (order matters)

- E: filters → v0.1 build + report → eval slices registered → **prompt list frozen (human sign-off)**
- F: Modal wiring → smoke run → **full conversion chain proven** → measured tok/s recorded
- G-v0: d12 on defaults → checkpoints + slider assets + learning curves → vs-scorecard comparison
- H: Q3/Q4/Q5 (corpus/schedule decisions) → Q1/Q2 (thesis experiments, 2 seeds)
- G-v1: fleet-informed rerun → final Model C base + nutrition label v1
- I: midtrain (rights-clear pairs) → SFT (native-reviewed) → chat-capable Model C
- Wrap: update plan.md §3.2/§11 with actuals; fertility + BPB tables gain Model C rows; queue series material

**Definition of done:** conversion chain proven; d12 trained with archived checkpoint generations + learning curves on the standing slices; Q1–Q5 answered with recorded configs/seeds; a chat-capable Model C that answers in Kreyòl; nutrition label v1 generated from provenance.

## Risks / open questions

- **nanochat + custom 24k vocab**: config plumbing and kernel padding verified only at toy scale (B0) — F's smoke run is the real test. Same for GGUF export of nanochat's arch.
- **Rights for translation-turn data** (Workstream I) — routed around via own-API synthetic over open sources, or Kreyòl-MT license resolution; do not let this drift into the training set unresolved.
- **Checkpoint prompt list is forever** — regenerating slider assets for new prompts requires kept weights (we keep them, but treat the list as an exhibit-design decision, made once, with care).
- **Fleet variance at 20M params** — 2 seeds on decision-critical comparisons; don't publish single-seed deltas.
- **SFT native review is the long pole** — schedule reviewer time early; compute waits on humans in this phase, not the reverse.
