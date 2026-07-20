# kreyol-chat

**Kreyòl Depi Nan Rasin / Building a language model from Kreyòl roots**

No dedicated Haitian Creole generative LLM exists today — the open ecosystem for Kreyòl is
encoder-decoder machine translation (Kreyòl-MT, OPUS-MT, NLLB) plus large multilingual models
that carry only trace amounts of it. This repository builds three small models to study what
changes when Haitian Creole gets its *own* tokenizer, its *own* data priorities, and its *own*
evaluation — and it will grow into [kreyol.chat](https://kreyol.chat), a site where you can talk
to those models and compare them side by side. The goal is not to replace speakers or define
"correct" Kreyòl; it is to understand how AI represents the language, and to let Haitian speakers
decide what good language technology should sound like.

This is a research-and-learning project, built in the open. It is honest about what it does and
doesn't do: the models here are small, the corpus is small, and many of the interesting numbers
have simply never been measured for Haitian Creole before. Measuring them is part of the point.

## Why this project

A few verified facts frame the work (full sourcing in [docs/plan.md](docs/plan.md)):

- **The token tax is real.** The same Kreyòl sentence costs about **1.74× more tokens** than its
  English equivalent on `cl100k_base` (the GPT-4 / ChatGPT tokenizer), versus only ~1.11× on
  NLLB's tokenizer, which was designed with low-resource languages in mind
  ([Petrov et al. 2023](https://arxiv.org/abs/2305.15425)). More tokens means higher cost, slower
  responses, less usable context, and weaker learning — for one language, from one design choice.
- **Frontier LLMs measurably lag on Kreyòl.** On FLORES-200 English→Haitian Creole, ChatGPT
  (gpt-3.5-turbo) scores 24.5 spBLEU / 47.0 chrF2++ — while dedicated MT systems score
  substantially higher ([Robinson et al. 2023](https://aclanthology.org/2023.wmt-1.40/)):

  | System | spBLEU | chrF2++ |
  |---|---|---|
  | ChatGPT (gpt-3.5-turbo), 0-shot | 24.5 | 47.0 |
  | NLLB-MoE (54.5B, dedicated MT) | 30.5 | 51.9 |
  | Google Translate | **31.8** | **53.4** |

  The same ChatGPT matches or beats NLLB on French — but trails dedicated MT by 5–7 spBLEU points
  on Haitian Creole. That is the textbook low-resource gap, in one table.
- **The project is data-bound, not compute-bound.** All deduplicated open Haitian Creole text adds
  up to roughly **100–200M tokens** — dominated by a single Common Crawl-derived source. Training
  small models at this scale costs tens of dollars; assembling and curating the corpus is the
  actual work.
- **Three models, one variable each.** See below.

## The three models

Each model isolates a single variable so that comparing them answers the research question:
*how do tokenizer design, corpus composition, and language adaptation affect a small model's
ability to understand and generate natural Haitian Creole?*

| | Model A — Baseline | Model B — Kreyòl-adapted | Model C — Kreyòl-first |
|---|---|---|---|
| What | Frontier API + dedicated MT, as-is | Small open model + continued pretraining on Kreyòl | From-scratch small model, Kreyòl tokenizer, Kreyòl-curated corpus |
| Represents | What the world gets today | What adaptation buys | What starting from Kreyòl looks like |
| Tokenizer | Their own (~1.7× tax) | Base model's (kept) | Trained on Kreyòl, 16–24k vocab (~1.0× by construction) |

Design principle: **one variable per model.** Model B isolates data adaptation (same tokenizer,
same architecture, new data). Model C isolates starting from Kreyòl (own tokenizer, own data
priorities). More detail — architecture lineage, training recipes, sizing math — is in
[docs/plan.md](docs/plan.md).

## Status

**Phase 0 — corpus, tokenizer, fertility measurement, base-model probe.** See the runbook:
[docs/phase-0.md](docs/phase-0.md). No training code is in the repo yet; this is the initial
scaffold and the public project docs.

| Phase | Focus | Status |
|---|---|---|
| **0** | Corpus v0, Kreyòl tokenizer, fertility numbers vs ~8 tokenizers, base-model probe | In progress — [runbook](docs/phase-0.md) |
| 1 | Model C v0 (from-scratch, nanochat-style) + micro-model mixture ablations | Planned |
| 2 | Model B (Kreyòl-adapted continued pretraining + instruction stage) | Planned |
| 3 | Evaluation — CreoleVal, FLORES+, a Kreyòl-construction test suite, human eval | Planned |
| 4 | The [kreyol.chat](https://kreyol.chat) site / interactive exhibit | Planned |
| 5 | Post-event & ongoing — community eval set, voice, license consultation | Planned |

## Data policy

This repository ships **reproduction scripts, not redistributed corpora.** Source licenses vary and
several forbid or complicate re-hosting: Kreyòl-MT's dataset is tagged license "other," and
CreoleVal's sub-datasets carry mixed per-file licenses. The Phase 0 pipeline downloads sources into
a git-ignored `ml/data/` directory on your own machine; the repo describes *how* to assemble the
corpus, and never contains the corpus itself. A license check is required before training on any
source.

**Model-weight release decisions are deferred to community consultation** (see below) — whether and
how to publish the trained models and tokenizer is a decision to make *with* Haitian speakers, not
before.

## Community

Built by **[CreolePro](https://www.creolemt.com)** as part of ongoing Haitian community
language-and-culture work. Contributions from Kreyòl speakers are especially welcome — proverbs,
evaluation sentences, corrections, and judgments of what sounds natural. Human evaluation is
central to this project, not decorative.

Two efforts are the project's north stars:

- **Michel DeGraff's "Creole Exceptionalism" critique** — Kreyòl is a full, rule-governed language,
  not "simplified French"
  ([Language in Society 34(4), 2005](https://www.cambridge.org/core/journals/language-in-society/article/abs/linguists-most-dangerous-myth-the-fallacy-of-creole-exceptionalism/25195A225AF5A6976E3EFEC8D53C42A8)).
  DeGraff (MIT Linguistics; MIT-Haiti Initiative) is a co-author of the CreoleVal benchmark this
  project evaluates on.
- **Te Hiku Media's data-sovereignty precedent** — their
  [Kaitiakitanga License](https://github.com/TeHikuMedia/Kaitiakitanga-License) treats
  community-donated language data as a treasure held under guardianship, where the community decides
  how its data is used. The license itself is specific to Māori tikanga and not directly reusable,
  but the *pattern* guides this project's consent and governance design.

No community data-sovereignty license exists yet for Haitian Creole or any Caribbean creole. This
project does not claim to be community governance — it aims to practice its principles honestly:
plain-language consent, provenance-tagged data, a review queue rather than silent training, and a
published account of what goes into each model.

## Repository layout

```
ml/     # Python workspace for all model / corpus / tokenizer work (uv-managed)
docs/   # public project docs — plan.md (the full plan) and phase-0.md (the runbook)
web/    # reserved for the kreyol.chat Next.js app (stub for now)
```

## License

Code and scripts in this repository are licensed under the
[Apache License 2.0](LICENSE), copyright 2026 CreolePro. Third-party datasets, models, and corpora
referenced by the scripts keep their own licenses — check each before use.
