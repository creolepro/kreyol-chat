# Kreyòl Depi Nan Rasin — Building a Language Model from Kreyòl Roots

**Status:** Research & planning (verified July 18, 2026)
**Context:** A CreolePro project ([creolemt.com](https://www.creolemt.com)), following [Jaden Lakou Kreyòl](https://www.creolemt.com/jaden-lakou), as part of ongoing Haitian community language-and-culture work. This doc captures the full research-verified plan for training small Haitian Creole language models and turning the process into an interactive exhibit.

**Research question:**
> How do tokenizer design, corpus composition, and language adaptation affect a small model's ability to understand and generate natural Haitian Creole?

**The honest narrative (what we claim and don't claim):**
> We built several small models to study what changes when Kreyòl gets its own vocabulary, its own data priorities, and its own evaluation criteria. The goal is not to replace speakers or define "correct" Kreyòl — it's to understand how AI represents the language, and to let Haitian speakers decide what good language technology should sound like.

---

## 1. What the research verified (and corrected)

Every claim below was checked against primary sources in July 2026. Corrections to our initial assumptions are called out — several of them make the story *better*.

### 1.1 The token tax is real and now has exact numbers

Petrov et al. 2023, *"Language Model Tokenizers Introduce Unfairness Between Languages"* (NeurIPS 2023, [arXiv:2305.15425](https://arxiv.org/abs/2305.15425)) measured Haitian Creole directly on parallel FLORES sentences ([raw data](https://github.com/aleksandarpetrov/tokenization-fairness)). Tokens needed for identical content, Haitian Creole ÷ English:

| Tokenizer | ht/en premium |
|---|---|
| GPT-2 / GPT-3 (r50k/p50k) | **1.90×** |
| **cl100k_base (GPT-4 / ChatGPT)** | **1.74×** |
| Qwen | 1.72× |
| Llama 1/2 (32k SentencePiece) | 1.58× |
| BLOOM | 1.56× |
| mBART-50 / XLM-R | 1.39× |
| mT5 | 1.22× |
| M2M-100 | 1.16× |
| **NLLB (dedicated MT)** | **1.11×** |

**The exhibit-ready fact:** the same Kreyòl sentence costs ~74% more tokens than its English equivalent on the GPT-4-era tokenizer — but only 11% more on NLLB's tokenizer, which was *designed* with low-resource languages in mind. Tokenizer design is a choice, and the choice has consequences.

**Correction to our framing:** Haitian Creole sits at the *milder* end of the disparity (some languages hit 15×) because it's Latin-script and French-lexified. The story isn't "Kreyòl is catastrophically fragmented" — it's "even a well-behaved Latin-script language pays a real tax when nobody put it in the room."

**The vs-French angle (same paper):** Petrov et al. also single out Haitian and Mauritian Creole vs *French itself*: on CamemBERT (a French model), the Creoles show "large differences in tokenization lengths" compared to French, while English sits closer to parity. Kreyòl is penalized even relative to its own lexifier — a pointed exhibit fact given DeGraff's "not simplified French" framing.

**Why the premium matters (beyond "more tokens"):** tokens are the currency for everything in an LLM, so one multiplier taxes four things at once:

- **Cost & quotas** — per-token billing means the same conversation costs a Kreyòl speaker ~74% more (Ahia et al.'s fairness point: the tax lands on the speaker, not the provider), and token-denominated rate limits/quotas exhaust ~1.7× faster.
- **Latency** — autoregressive decoding is one step per token, so identical replies generate ~1.74× slower in Kreyòl.
- **Effective context** — a context window holds ~40% less Kreyòl than English: fewer few-shot examples fit, RAG retrieves less, chat history truncates sooner. The model is effectively *smaller* for the language.
- **Learning quality** — fragmented words misalign the modeling unit with the semantic unit, and inconsistent splits (leading space, punctuation, capitalization) spread one word's statistics across several surface variants → weaker representations. This is the mechanism Model C tests. It also compounds during pretraining: token-denominated data mixes deliver *less content* per Kreyòl token, and per-token metrics (perplexity, safety thresholds) calibrated on English are silently miscalibrated for Kreyòl (adjacent finding: low-resource languages bypass safety training more easily — Yong et al. 2023, [arXiv:2310.02446](https://arxiv.org/abs/2310.02446)). Flip side for us: a 16–24k Kreyòl tokenizer stretches the ~100–200M-token corpus further.

Station 1 punchline: *same sentence — less memory, slower answer, higher price, weaker learning.*

**Update (2026-07-19): we measured the post-Petrov generation ourselves** — first published HT parity numbers we could find for these tokenizers, on 1,012 FLORES+ devtest sentences with bootstrap CIs (full report: [ml/reports/fertility.md](../ml/reports/fertility.md)):

| Tokenizer (our measurement) | ht/en parity |
|---|---|
| **kreyol-bpe (ours, 24k — Workstream B)** | **0.67** |
| NLLB (reproduces Petrov) | 1.10 |
| **o200k (GPT-4o-era)** | **1.41** |
| Claude API input parity (claude-opus-4-8) | 1.51 |
| Gemma-3 (256k vocab) | 1.53 |
| SmolLM3 (= Llama-3-family tokenizer, 128k) | 1.70 |
| Qwen3 | 1.72 |
| cl100k (reproduces Petrov's 1.74) | 1.74 |

Three findings: **o200k roughly halves the tax vs cl100k** — tokenizer progress *can* reach Kreyòl — but **newer ≠ fairer**: Qwen3 and the Llama-3-family tokenizer are still at cl100k-era levels (the tax is a design choice, not an era). And **our Kreyòl-first tokenizer flips the tax entirely — ht/en 0.67× (ht/fr 0.57×)**: the same content costs *fewer* tokens in Kreyòl than in English once the vocabulary is trained on Kreyòl. Station 1 computes exactly this live.

### 1.2 The quality gap is real: frontier LLMs measurably lag on Kreyòl

Robinson et al. 2023, *"ChatGPT MT: Competitive for High- (but not Low-) Resource Languages"* (WMT 2023, [aclanthology.org/2023.wmt-1.40](https://aclanthology.org/2023.wmt-1.40/)), English→Haitian Creole on FLORES-200:

| System | spBLEU | chrF2++ |
|---|---|---|
| ChatGPT (gpt-3.5-turbo), 0-shot | 24.5 | 47.0 |
| ChatGPT, 5-shot | 24.8 | 47.2 |
| NLLB-MoE (54.5B, dedicated MT) | 30.5 | 51.9 |
| **Google Translate** | **31.8** | **53.4** |

The same ChatGPT **matches or beats** NLLB on French (56.4 vs 56.2 spBLEU) and German — but trails dedicated MT by **5–7 points** on Haitian Creole. Globally, ChatGPT underperformed traditional MT for 84.1% of 203 languages, with resource level the dominant predictor. This is the textbook low-resource gap, in one clean table.

*(Direction + numbers re-verified against the paper PDF directly: Table 11 is ENG→X only — "the FLORES-200 English data was taken from Wikipedia… making fair X→ENG evaluation infeasible" — and the `hat_Latn` row and GPT-4's absence for it are confirmed verbatim.)*

**Gaps:** no published FLORES numbers for GPT-4/GPT-4o/Claude/Gemini on Haitian Creole specifically (GPT-4 was only run on higher-resource languages in that paper). The WMT-2025 Creole shared task ([statmt.org/wmt25/creole-mt.html](https://www2.statmt.org/wmt25/creole-mt.html)) includes Haitian and uses Kreyòl-MT's public model as baseline — worth watching, and running our own Claude-vs-baselines eval fills a real hole.

### 1.3 Kreyòl-MT verified — and Haitian is the giant of the Creole data world

*Kreyòl-MT* (Robinson et al., NAACL 2024, [arXiv:2405.05376](https://arxiv.org/abs/2405.05376), [github.com/JHU-CLSP/Kreyol-MT](https://github.com/JHU-CLSP/Kreyol-MT)):

- 41 Creole languages, 172 translation directions, **14.5M unique Creole sentences gathered, 11.6M released publicly** (+ ~3.4M monolingual across 40 languages).
- **Haitian Creole: 5,715,227 public parallel sentences (~6.02M total)** — about **49% of the entire public corpus**, mostly paired with English (~667k direct hat-eng rows in the HF viewer) and French. By far the best-resourced Creole.
- Models (all MIT-licensed): `jhu-clsp/kreyol-mt` and `-pubtrain` (fine-tuned mBART-50, ~611M params) plus **`-scratch` and `-scratch-pubtrain` — 77M-param Transformers trained from scratch**. The from-scratch models are a direct precedent for our Model C at almost the same scale.
- Dataset: [`jhu-clsp/kreyol-mt`](https://huggingface.co/datasets/jhu-clsp/kreyol-mt), license tagged "other." ⚠️ The live HF snapshot is currently **partial (~1.9M sentences)** — the LDS-sourced portion (~360k sentences) is held back pending an LDC release. Full data via contacting the authors. **License check required before any commercial use.**

**Correction to our framing:** Haitian's slice is **not primarily religious** — it's dominated by NLLB *web-mined* text ("Other/Mix" genre). Religious skew is the story for the *smaller* Creoles. For Haitian the sharper concern is **translation-shaped, machine-aligned web text** — which actually strengthens the "translation-shaped Kreyòl vs. authored Kreyòl" thesis.

### 1.4 Genre skew now has a number

*"Limitations of Religious Data… MT for Guinea-Bissau Creole"* ([arXiv:2504.02674](https://arxiv.org/html/2504.02674)): models trained on Bible + Jehovah's Witnesses data alone averaged **4.23 BLEU** on a general-domain test set — fluent-looking, unusable. Adding just **300 everyday-domain sentences** gained **+4.0–6.7 BLEU**; 600 (oversampled) reached 11.9 BLEU (~3×).

**Event lesson, now citable:** *more data is not always more language* — and even tiny amounts of the right data move the needle. This directly justifies community collection of small, high-quality conversational Kreyòl (Station 4).

### 1.5 The data ceiling: ~100–200M open tokens (the project is data-bound, not compute-bound)

Full inventory in §5. Bottom line: all deduplicated open Haitian Creole text ≈ **1–2 × 10⁸ tokens**, dominated by one source (MADLAD-400's Common Crawl harvest). Compute for models at this scale costs tens of dollars (§7). **Corpus assembly and curation is the actual project** — which is exactly the story the exhibit wants to tell.

### 1.6 Nobody has built this yet (as far as we can find)

Gaps verified as of July 2026 — phrased carefully: "we found no publicly released X," not "X cannot exist":

- **No dedicated Haitian Creole generative LLM found** on Hugging Face or in the literature — the text ecosystem is encoder-decoder MT (Kreyòl-MT, OPUS-MT, NLLB) plus multilingual models with trace HT (mT5 ~0.33%). Masakhane doesn't cover Haitian (African languages only). The earliest academic study of the question is Lent et al. 2021, *On Language Models for Creoles* ([arXiv:2109.06074](https://arxiv.org/abs/2109.06074)) — analysis, not a released HT model.
- **No commercial TTS API supports Haitian Creole at all** (verified against Google, Azure, Polly, ElevenLabs docs — §9).
- **Common Voice has 2.7 minutes of Haitian Creole audio** (30 clips, 3 speakers).
- **No community data-sovereignty license exists for any Caribbean creole** (§10) — Te Hiku Media's Kaitiakitanga license (Māori) is the precedent, with no Haitian equivalent.

---

## 2. Architectural lineage: one paper connects every model in the exhibit

*Attention Is All You Need* (Vaswani et al. 2017, [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)) is the common ancestor of everything on the table, and the exhibit should say so explicitly:

- **The original Transformer** was an encoder-decoder built *for machine translation*: multi-head scaled dot-product attention, sinusoidal positional encodings, ~65M params (base) / ~213M (big), trained on WMT14 English–German — **4.5M sentence pairs with a ~37k shared BPE vocabulary**.
- **Kreyòl-MT's models** (mBART-50 fine-tunes and the 77M from-scratch models) are encoder-decoder Transformers — the *same shape* as the 2017 original.
- **Our Model C** (nanochat/nanoGPT-style) is the decoder-only branch of the family tree (GPT lineage), with modern refinements (rotary positional embeddings instead of sinusoids, RMSNorm, etc.) — but the attention mechanism is the 2017 mechanism.
- **Claude and ChatGPT** are the same decoder-only lineage at massive scale.

**The exhibit-ready fact:** the paper that launched all of modern AI trained on **4.5M English–German sentence pairs. Haitian Creole today has ~6M parallel pairs — more raw pairs than the dataset that started the Transformer revolution.** (Kept honest: quality, domain, alignment, and institutional attention differ — raw pair count was never the whole gap.) And Vaswani et al. baked subword tokenization (BPE) into the recipe from day one — the tokenizer question Station 1 explores is literally as old as the architecture.

**Exhibit tie-in:** an attention-map visualization station ("Anndan tèt modèl la" — inside the model's head) rendering Model C's attention weights on a proverb like *Dèyè mòn gen mòn* — which words attend to which, straight from the mechanism in Figure 2 of the paper.

---

## 3. The three-model experiment

| | Model A — Baseline | Model B — Kreyòl-adapted | Model C — Kreyòl-first |
|---|---|---|---|
| What | Frontier API (Claude) + dedicated MT (NLLB-600M / Google Translate) as-is | Small open model + continued pretraining on Kreyòl | From-scratch small model, Kreyòl tokenizer, Kreyòl-curated corpus |
| Represents | What the world gets today | What adaptation buys | What starting from Kreyòl looks like |
| Size | — | 1.7B–4B | ~200M (nanochat d12) + micro-models for ablations |
| Tokenizer | Their own (1.7× tax) | Base model's (kept — see below) | **Trained on Kreyòl, 16–24k vocab — low fertility on Kreyòl; ht/en parity likely <1 (it flips the tax)** |
| Compute cost | API pennies | ~$25–75/run | ~$5–100/run |

Framing: **three development paths, not a controlled experiment.** A, B, and C differ in size, data, initialization, and training history all at once — comparing them answers "what does each *path* deliver?" (the exhibit's question), but it cannot attribute differences to any single cause. The controlled science lives in three **paired comparisons**, each changing exactly one thing:

- **Adaptation**: Model B's exact base checkpoint *before* vs. *after* CPT — same everything else. (The Phase 0 probe banks the "before" numbers.)
- **Tokenizer**: two micro-models with identical architecture, data order, and recipe — one on the Kreyòl tokenizer, one on a control tokenizer. ~$10; this is the *actual* tokenizer experiment.
- **Corpus mixture**: micro-models identical except the data mix (§3.3 experiment 2).

**Training-objective primer** — three pretraining lineages show up around this project; knowing which is which prevents confusion:

| Objective | What the model learns | Who uses it |
|---|---|---|
| **Causal LM** (next-token prediction) | Predict token *t+1* from tokens *1..t*; generation falls out directly | GPT lineage: Qwen3, SmolLM3, nanochat — **our Models B & C** |
| **Denoising autoencoder** (corrupt text → reconstruct it) | Encoder reads noised input, decoder rebuilds the original; strong starting point for translation fine-tuning | BART/mBART ([Lewis et al. 2019](https://arxiv.org/abs/1910.13461); [Liu et al. 2020](https://arxiv.org/abs/2001.08210)) → **Kreyòl-MT's models** |
| **Masked LM** (fill in blanks) | Bidirectional understanding; no generation | BERT/XLM-R — CreoleVal's baseline models |

What we borrow from Kreyòl-MT is the *principle*, not the objective: **pretrain on abundant data, adapt with scarce data**. Their base (mBART) arrived already knowing ~50 languages from denoising pretraining, so Creole parallel data only had to teach the mapping. Our Model B does the same move on the decoder-only branch: a causally-pretrained base arrives knowing English/French/code, and our Kreyòl corpus only has to teach the delta.

### 3.1 Model B — Kreyòl-adapted (continued pretraining)

**Method: full-parameter continued pretraining (CPT), not LoRA.** The evidence is unambiguous for language acquisition:

- *LoRA Learns Less and Forgets Less* (Biderman et al., TMLR 2024, [arXiv:2405.09673](https://arxiv.org/abs/2405.09673)): LoRA substantially underperforms full fine-tuning in the CPT regime; full training learns perturbations of 10–100× higher rank than typical LoRA — a new language is exactly the large distribution shift LoRA can't absorb. LoRA remains fine for the later instruction-tuning stage.
- Proven CPT recipes: **Sailor2** ([arXiv:2502.12982](https://arxiv.org/abs/2502.12982)) used 80% target-language / 20% replay; **Swallow** (Japanese, [arXiv:2404.17790](https://arxiv.org/abs/2404.17790)) landed at ~90/10 and found performance rose monotonically with target-language data.
- **Our mixture: ~85–90% Haitian Creole / 10–15% replay** (English + French + a little code/math to preserve the base model's abilities).

**What CPT actually is, and why it works:**

- **Mechanically, CPT is just resumed pretraining**: the same next-token cross-entropy objective the base was trained with, pointed at a new data mix. No labels, no new task, loss on every token. The canonical reference is *Don't Stop Pretraining* (Gururangan et al. 2020, [arXiv:2004.10964](https://arxiv.org/abs/2004.10964)) — domain/task-adaptive pretraining reliably beats using the base as-is.
- **Why 200M tokens can teach a language**: cross-lingual transfer. The base's subword vocabulary and internal representations are shared across languages ([XLM-R, Conneau et al. 2020](https://arxiv.org/abs/1911.02116)), so Kreyòl's French-derived lexicon and English contact vocabulary land on circuits that already exist — the model doesn't learn language from zero, it learns *Kreyòl's deltas*: the grammar statistics (TMA markers, postposed determiners, `yo`), orthography, and idiom that its pretraining never saw.
- **Recipe mechanics** (why each knob exists): peak learning rate ~10× *below* the base's pretraining peak, with warmup — we're nudging a converged model, not shattering it. The 10–15% replay of English/French/code anchors the original distribution against catastrophic forgetting (Swallow found even ~1% helps; Sailor2 used 20%). Two-stage curriculum per Sailor2: broad web-crawled Kreyòl first, then upsample high-quality/native-authored text late in training so the final distribution tilts toward the voice we want.
- **The SFT stage is the same objective, masked**: loss computed only on assistant-response tokens, with a chat template — this teaches turn-taking and instruction-following, not language ([InstructGPT](https://arxiv.org/abs/2203.02155)). It's a small distribution shift, which is why LoRA suffices here and why ~1–10k *excellent* examples beat large noisy sets ([LIMA](https://arxiv.org/abs/2305.11206)).

**Tokenizer: keep the base model's.** Kreyòl is Latin-script with regular orthography; modern tokenizers handle it passably, full swaps buy efficiency but not measured quality ([arXiv:2408.15793](https://arxiv.org/abs/2408.15793)), and keeping it makes the A-vs-B comparison clean. Optional: light vocab *extension* (a few hundred high-frequency Kreyòl subwords, FOCUS-style init) only if measured fertility is poor.

**Base model candidates** (must be Apache-2.0-clean for a community project, locally runnable at the event). CPT starts from the **pretrained base checkpoints** — probe and adapt those, not the instruction-tuned variants:

1. **[Qwen3-1.7B-Base](https://huggingface.co/Qwen/Qwen3-1.7B-Base) / [Qwen3-4B-Base](https://huggingface.co/Qwen/Qwen3-4B-Base)** — Apache 2.0, 36T-token pretraining across 119 languages, best small-model quality reputation. *(Whether HT is in the 119 is unverified — the probe settles it.)*
2. **[SmolLM3-3B-Base](https://huggingface.co/HuggingFaceTB/SmolLM3-3B-Base)** — Apache 2.0 and *fully open* (data recipe, code, public intermediate checkpoints), strong French coverage (helpful for a French-lexifier creole). Best transparency story for an exhibit about openness.
3. [gemma-3-4b-pt](https://huggingface.co/google/gemma-3-4b-pt) (140+ languages) only if its custom license is acceptable; Llama 3.2's HT quality is likely weak (8 official languages, HT not among them).

**Pre-commit probe** (staged, on FLORES+ **dev** — devtest is reserved for final reporting): base models don't reliably follow zero-shot instructions, so the primary base-model measures are **bits-per-byte on authored Kreyòl + standardized few-shot completion**, scored on a small scorecard (BPB, few-shot chrF, blinded naturalness) rather than a single metric. Full protocol: [phase-0.md](./phase-0.md) Workstream D. The deltas become exhibit content.

**Tooling: Axolotl on Modal** (YAML-config CPT + SFT, documented Modal path: [modal.com/blog/fine-tuning-llms](https://modal.com/blog/fine-tuning-llms), [modal-labs/llm-finetuning](https://github.com/modal-labs/llm-finetuning)). Unsloth as single-GPU alternative (has an explicit continued-pretraining mode that also trains embeddings/lm_head).

**Instruction stage (small, high-quality, ~1–10k examples):**
- Seed from the Aya Collection's `haitian` config ([CohereLabs/aya_collection_language_split](https://huggingface.co/datasets/CohereLabs/aya_collection)) — verified to exist, but it's machine-translated (HT is *not* in the 65-language human-annotated Aya set), so it needs a native-speaker quality pass.
- Generate self-instruct data seeded on *native* Kreyòl text (MURI-style reverse instructions, [arXiv:2409.12958](https://arxiv.org/abs/2409.12958)) rather than translating English instruction sets — avoids the worst translationese.
- Kreyòl-MT pairs for translation-task instructions.
- **Native-speaker review of a large sample is mandatory** (§6).

### 3.2 Model C — Kreyòl-first (from scratch)

**Codebase: [karpathy/nanochat](https://github.com/karpathy/nanochat)** (MIT). It is exactly the needed pipeline in one repo: tokenizer training (rustbpe, byte-level BPE) → pretraining → midtraining → SFT → eval → CLI + web chat UI. One `--depth` dial sizes the model (d12 = 203M params; d20 = 561M / 11.2B tokens / the "$100 speedrun" on 8×H100). Runs on a single GPU (auto gradient accumulation) or 8×A100/H100. A [nanochat-on-Modal tutorial](https://aiengineering.academy) exists (~$112 for the full d20 pipeline on 8×A100).

- nanoGPT remains the reference for maximal hackability but lacks tokenizer training, SFT, and serving; FareedKhan-dev/train-llm-from-scratch is tutorial-grade (good for *learning*, not for the deployable artifact). modded-nanogpt is a source of speed tricks (Muon), not a base.
- **Architecture constraint (decided now, saves pain later):** stay on nanochat's Llama-compatible (or a GPT-2-compatible) architecture so conversion to GGUF (`convert_hf_to_gguf.py`) and browser formats is turnkey. A custom architecture means hand-writing conversion code ([llama.cpp discussion](https://github.com/ggml-org/llama.cpp/discussions/19815)).

**What from-scratch training actually means:**

- Same causal next-token objective as Model B — but the weights start **random**. Everything must be learned from our corpus alone, and it's learned in a visible order: character/orthography statistics first, then word forms, then grammar, then meaning and facts. That acquisition sequence is exactly what the Pythia-style checkpoints capture, which is why the "watch it learn Kreyòl" station works — it's not a gimmick, it's the actual training dynamics ([GPT-2, Radford et al. 2019](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) is the lineage anchor).
- **Why decoder-only rather than Kreyòl-MT's encoder-decoder**: their models are translation specialists — the encoder-decoder shape (and denoising pretraining) is optimized for mapping one sequence to another. Our target is a *generative* Kreyòl model that chats, completes proverbs, and explains — plus turnkey GGUF/browser conversion, which the GPT/Llama lineage gets for free. Translation becomes just another instruction ("Tradui sa a…") rather than a dedicated architecture.
- **nanochat's pipeline stages, and why each exists**: *tokenizer training* defines the atoms everything else is built from (our one big deviation: 16–24k Kreyòl vocab, below). *Base pretraining* on the raw corpus produces a model that can only continue text. *Midtraining* teaches conversation structure — chat-turn formatting, multiple-choice mechanics — cheap format knowledge that doesn't need much data; ours would use Kreyòl chat templates and can fold in parallel data as translation turns. *SFT* on a small curated instruction set (native-reviewed, per §3.1's LIMA logic) makes it an assistant. The stages are separated because each needs progressively less data and more curation.
- **Hyperparameters mostly come for free**: nanochat's `--depth` dial auto-derives width, heads, learning rate, and training horizon to be compute-optimal at each size. Our deliberate deviations from its defaults: the smaller Kreyòl vocabulary, the multi-epoch data-constrained schedule (Muennighoff — §3.2 sizing), and the provenance-tagged curriculum ordering with synthetic data confined to late stages (§5.3).

**Tokenizer: byte-level BPE, 16–24k vocab, trained on the Kreyòl corpus.** Not nanochat's default 65,536 — at 200M params the embedding tables would swallow the model. Research nuance worth exhibiting: Unigram-over-BPE advantages ([arXiv:2508.08424](https://arxiv.org/abs/2508.08424)) apply to *morphologically rich* languages; Kreyòl is analytic/isolating with phonemic orthography, so BPE is sound. Verify the tokenizer keeps intact: `ap`, `pral`, `te`, `yo`, postposed articles (`la/a/an/nan`), accented `è/ò`, digraphs (`ou/an/en/on`), and proverb vocabulary.

**Data-bound sizing (the honest math):** ~100–200M unique open tokens vs. Chinchilla-optimal ~2.5B for even a 125M model. Plan:
- Multi-epoch training — data-constrained scaling research (Muennighoff et al., [arXiv:2305.16264](https://arxiv.org/abs/2305.16264)) shows ~4 epochs of repeated data ≈ fresh data, with value decaying by ~16 epochs. 150M unique × 4–8 epochs ≈ 0.6–1.2B effective tokens — a respectable run for d12.
- Optionally add the Kreyòl side of the 6M parallel pairs and reviewed synthetic data *as tagged curriculum stages* (§5.3) — each addition is itself an experiment.
- **d12 (203M) is the primary artifact; ~10–30M-param micro-models are the ablation fleet** (mixture experiments at ~$1–5/run).

**Checkpoint curriculum (the "watch it learn Kreyòl" exhibit):** copy Pythia's log-then-linear schedule ([arXiv:2304.01373](https://arxiv.org/abs/2304.01373) — 154 public checkpoints; also OLMo, SmolLM3). Save at ~0 / 10M / 25M / 50M / 100M / 250M / 500M / 1B effective tokens; at every checkpoint run the *same* fixed prompt set (a greeting, a proverb completion, a short translation, a question). Visitors scrub a slider and watch fluency emerge.

### 3.3 The four experiments (mapped to stations)

1. **Tokenizer experiment** → Station 1. Train the Kreyòl tokenizer; measure fertility (tokens/sentence) for Kreyòl vs English vs French across: our tokenizer, cl100k, o200k, Llama 3, Gemma, NLLB. *(✅ done — external tokenizers 2026-07-19 and **our tokenizer 2026-07-20 (ht/en 0.67×, flips the tax)**; see §1.1 table + [ml/reports/fertility.md](../ml/reports/fertility.md) and [ml/reports/tokenizer_v0.md](../ml/reports/tokenizer_v0.md).)* Show which words survive intact.

   **Measurement protocol** (~150 lines of Python, no GPU, an afternoon):
   - *Data:* FLORES+ `hat/eng/fra` devtest — 1,012 parallel sentences, same content in all three languages (gated HF repo, CC-BY-SA) — plus a small authored-Kreyòl set (proverbs, MIT-Haiti sentences), since FLORES's Haitian side is itself translated; a translated-vs-authored fertility gap would be a finding on its own.
   - *Tokenizers:* `tiktoken` for cl100k + o200k (labeled as tokenizers, not products); HF `AutoTokenizer` for Llama 3 / Gemma 3 (gated repos — accept terms) and Qwen3 / NLLB / SmolLM3 (open); **Claude via the free `count_tokens` API endpoint** — Anthropic documents the result as an *estimate* that may include request overhead, so measure empty-request overhead, pin the exact model ID, and label the result **"Claude API input parity for model X"** rather than a raw tokenizer count. Add our rustbpe tokenizer once trained.
   - *Metrics per tokenizer:* primary parity = **sum(ht tokens) ÷ sum(eng tokens)** over the whole corpus (and vs French — the lexifier angle), with paired-bootstrap confidence intervals and per-sentence quantiles ("mean of per-sentence ratios" is a different statistic — never silently substitute it); tokens/word under a *defined* segmentation rule (apostrophe clitics like `m'ap` decided up front); whole-word survival of a core vocab list (with/without leading space); sentences-per-fixed-token-budget + $ premium at **date-stamped** API prices, shown only for tokenizers tied to actually-priced APIs.
   - *Rigor:* NFC-normalize everywhere (è/ò); first **replicate Petrov's 1.74× on his released data** (repo CSV, same tokenizer version) to validate the pipeline, then report FLORES+ as our own measurement (expected close — not gated to match); publish script + output CSV; claim "first *published*, as of <date>" rather than "first-ever."
   - *Reuse:* the same code runs **live in-browser** at Station 1 — js-tiktoken (cl100k/o200k) + transformers.js-loaded HF tokenizers + our BPE as JSON, fully offline; only the Claude numbers are precomputed.
2. **Corpus experiment** → Station 5. Micro-models on contrasting mixtures: formal/web-dominant vs everyday/conversational-weighted vs balanced; and no-synthetic vs unreviewed-synthetic vs reviewed-synthetic. Same prompts to each → "data selection shapes a model's voice."
3. **Linguistic construction eval** → §6.1. A Kreyòl-phenomena test suite (TMA markers, postposed determiners, `yo` pluralization, serial verbs, negation, false friends vs French, proverbs) — not translated English benchmarks.
4. **Human evaluation** → Station 4 + §6.2. Fluent speakers rate naturalness/appropriateness dimensions; consented contributions become a community eval set.

---

## 4. Baselines to beat (verified numbers to put on the wall)

| Eval | System | Score |
|---|---|---|
| FLORES-200 eng→hat | ChatGPT (3.5) 0-shot | 24.5 spBLEU / 47.0 chrF |
| FLORES-200 eng→hat | Google Translate | 31.8 spBLEU / 53.4 chrF |
| FLORES-200 eng→hat | NLLB-MoE | 30.5 spBLEU / 51.9 chrF |
| CreoleVal MIT-Haiti eng→hat | CreoleM2M | 22.0 BLEU / 43.9 chrF |
| CreoleVal MIT-Haiti eng→hat | opus-mt-en-ht | 14.7 BLEU / 35.8 chrF |
| CreoleVal Tatoeba eng→hat | opus-mt-en-ht | 45.2 BLEU (⚠️ authors call Tatoeba "overly optimistic") |
| CreoleVal NER (hat) | XLM-R | 0.84 span-F1 |

Kreyòl-MT's own models beat CreoleVal's by +6.4 chrF (X→eng) / +14.1 chrF (eng→X) on average, and their diverse-genre model beat a genre-specific one on 26 of 34 directions — the diversity argument, quantified. (Their per-direction Haitian chrF lives in Figures 3/5 of the PDF — read directly when we need exact targets.)

---

## 5. Data plan

### 5.1 Corpus inventory (verified sizes & licenses)

**Monolingual Haitian Creole text:**

| Source | Size | License | Notes |
|---|---|---|---|
| [MADLAD-400](https://huggingface.co/datasets/allenai/MADLAD-400) ht clean | **84.3M tokens** / 110k docs | CC-BY-4.0 per card (ODC-BY cited elsewhere — confirm) | The backbone. Common Crawl-derived |
| MADLAD-400 ht noisy | 163M tokens / 426k docs | 〃 | Superset; known contamination for low-resource langs |
| [ht.wikipedia.org](https://ht.wikipedia.org/wiki/Espesyal:Estatistik) | 72,023 articles / ~10.5M words | CC-BY-SA 4.0 | Bot-stub-inflated; real prose smaller |
| CC-100 ht | 9.1MB .xz (~5–8M tokens, est.) | research | ⊂ Common Crawl, overlaps MADLAD |
| OSCAR 23.01 ht | 2 documents (!) | — | Effectively empty; LID artifact |
| Bible(s) | ~850k words each | varies ([ebible.org](https://ebible.org/bible//country.php?c=HT)) | JW300 withdrawn from open distribution — don't rely on it |
| Mission 4636 / WMT11 pack | ~17k SMS pairs + med/news/dicts | research-use | 2010 earthquake corpus; historically resonant (also [rmunro/disaster_response_messages](https://github.com/rmunro/disaster_response_messages)) |
| Leipzig Corpora `hat_community_2017` | size unverified (bot-protected) | research | [corpora.uni-leipzig.de](https://corpora.uni-leipzig.de) — check manually |

**Parallel:** ~**6.0M unique hat–eng pairs** is the realistic deduplicated ceiling (Kreyòl-MT's own dedup; raw OPUS sums to ~15.8M but NLLB-mined/CCAligned overlap enormously — [OPUS API](https://opus.nlpl.eu/opusapi/?source=en&target=ht&preprocessing=moses)). CCMatrix never covered ht.

**Eval-only:** FLORES+ `hat_Latn`, 997 dev + 1,012 devtest, CC-BY-SA ([openlanguagedata/flores_plus](https://huggingface.co/datasets/openlanguagedata/flores_plus)); CreoleVal tasks (§6).

**Instruction:** Aya Collection `haitian` config (machine-translated — review before use); xP3/BLOOMZ has no HT.

**Audio:** Common Voice ht = 0.04 hours total. Radio Haiti-Inter annotated spoken corpus announced (LREC 2026) — speech, watch for release.

**Deduplicated realistic total: ~90–100M tokens (clean build) to ~185–210M (everything incl. noisy web).**

**Measured (2026-07-20, corpus v0 clean build — [report](../ml/reports/corpus_v0.md)):** **143,940 docs / 151.6M `o200k` tokens / 91.1M whitespace words / 484M chars**, from MADLAD-400 ht clean + ht Wikipedia (dump 20260701) + 35 teachable proverbs, after ~20% dedup removal. Note the unit: 151.6M is *o200k* tokens (ht ≈ 1.66 o200k tokens/whitespace word), **not** MADLAD's ~84.3M whitespace-token estimate — the two aren't comparable, and the whitespace-word count (91.1M) is what lines up with the ~90–100M estimate above. ~57% of surviving Wikipedia is bot-stub (flagged).

**Additional sources (private or partnership-dependent, not redistributed in this repo):** a dictionary KV store built on Kreyòl-MT data; translation-platform glossaries; any consented text from events; API-generated synthetic translations (§5.3). Plus partnership targets: Platfòm MIT-Ayiti materials, Haitian publishers/newsrooms, diaspora orgs (licensing conversations required).

### 5.2 Provenance, rights, and splits (non-negotiable — established *before* any ingestion)

One overloaded "provenance" string can't reproduce or legally audit a corpus build. Every document carries structured metadata — a lightweight Pydantic schema, not a database — with separate fields:

- **origin** — `authored_kreyol` · `human_translation` · `machine_translation` · `machine_translation_reviewed` · `oral_transcription` · `synthetic_reviewed` · `synthetic_unreviewed`
- **genre** — encyclopedic · conversational · educational · religious · news · dictionary · historical · proverb
- **source/acquisition** — MADLAD, Wikipedia (dump date), CreolePro, etc., plus URL, download timestamp, stable doc ID, and content hashes (raw + cleaned)
- **rights** — license ID + URL, plus a per-source **rights matrix**: analysis / tokenizer-training / model-training / redistribution, each explicitly allowed, denied, or **unresolved → quarantined**. Unresolved sources (MIT-Haiti, the Kreyòl-MT-derived dictionary) never enter public artifacts — *including the tokenizer*, which is itself a derived artifact of its training text.
- **split** — assigned from a registry created before ingestion: `train` / `tokenizer_eval` / `model_selection_dev` / `final_devtest` / `exhibit_examples`. All CreoleVal/FLORES content stays out of training; authored material splits by document or collection, never randomly by sentence; benchmark-contamination checks run against the final training corpus. **Proverbs split deliberately**: a teachable set (models may learn them — Station 2's "who knows the real proverb?" depends on it) and a held-out probe set that never appears in training.

This one schema powers the corpus experiments, the legal audit trail, and Station 5's nutrition label.

### 5.3 Synthetic data policy (research-backed ceilings)

Our API can generate synthetic Kreyòl at scale; the model-collapse literature gives concrete guardrails:

- **Real/native data stays the majority anchor** (≥50% overall); never below **~5% real in any training round** (Seddik et al. 2024 — the mathematical floor below which collapse sets in).
- **Accumulate, never replace**: keep all real data in every round (Gerstgrasser et al. 2024 — accumulation avoids collapse; replacement causes it; original framing: Shumailov et al., Nature 2024).
- Synthetic data enters **only in later curriculum stages**, after the model has seen native Kreyòl.
- Every synthetic batch gets a **human-review sample**; reviewed and unreviewed synthetic are separate tags — and the no-synth / unreviewed-synth / reviewed-synth micro-model comparison is Experiment 2's best demo.

---

## 6. Evaluation

### 6.1 Automated

- **CreoleVal** ([github.com/hclent/CreoleVal](https://github.com/hclent/CreoleVal) — GitHub is authoritative; licenses vary per sub-dataset, check before *training* on any of it): Haitian appears in 4 of 8 tasks — machine comprehension (incl. the culturally-*localized* `hat-loc` variant — Station 3's theme as a benchmark), NER, Tatoeba sentence matching, MT (Bible + MIT-Haiti).
- **FLORES+ hat_Latn** for translation, spBLEU + chrF2++, matching Robinson et al.'s setup so our numbers slot into their table.
- **Our linguistic construction suite** (new, native-speaker-written): TMA markers (`te/ap/pral/ta` + combinations), postposed definite articles (`la/a/an/lan/nan`), plural `yo`, possessives, serial verbs, negation, question formation, gender-neutral pronoun handling, French false friends, spelling variation, code-switching, proverb completion/interpretation. Also: perplexity per checkpoint on a held-out authored-Kreyòl set (never web-crawled).
- **Tokenizer metrics**: fertility on parallel FLORES text (our tokenizer vs cl100k/o200k/Llama-3/Gemma/NLLB), % whole-word survival for a core vocabulary list.

### 6.2 Human (central, not decorative)

Fluent speakers rate model output on separate axes: grammatical validity · naturalness · cultural appropriateness · fidelity to intended meaning · register appropriateness · "sounds translated vs. originally composed" · dialect acceptability · "would a Haitian speaker actually say this?"

Sourcing: community reviewers from the event network (compensated where possible), with consented Station 4 contributions accumulating into a community eval set over time. CreoleVal's own ethics framing (DeGraff co-authored) demands exactly this: engage the community, benefit the community.

**Protocol:** model outputs are **blinded and randomized** — raters never see which model (or what size/brand) produced a sample, or expectation bias swamps the signal. Model *selection* happens on dev splits only; `final_devtest` is touched once, for final reporting.

---

## 7. Infrastructure & budget

### 7.1 Modal (primary — confirmed good fit)

Verified pricing ([modal.com/pricing](https://modal.com/pricing), per-second billing, scale-to-zero): T4 $0.59/h · L4 $0.80 · A10 $1.10 · L40S $1.95 · A100-40GB $2.10 · A100-80GB $2.50 · **H100 $3.95** · H200 $4.54 · B200 $6.25. Starter plan: **$30/month free credits**. Functions cap at 24h (checkpoint-resume pattern documented: [long-training example](https://modal.com/docs/examples/long-training)); up to 8 GPUs/node; Volumes for checkpoints; nanochat/nanoGPT Modal tutorials exist.

Alternatives if a big run needs cheaper raw hours: RunPod (~$2–2.7/h H100), Vast.ai (~$1.5–1.9, variable reliability), Lambda (~$3), SF Compute (~$1.5–1.75 blocks). For our bursty, small-scale pattern, Modal's zero-idle model likely wins on total cost, and the $30/month credits cover the micro-model fleet entirely.

### 7.2 Cost estimates (from measured throughputs — llm.c/nanochat anchors)

| Run | Compute | Est. cost |
|---|---|---|
| Micro-model (10–30M) mixture ablation | <1 GPU-h | **$1–5** |
| Model C d12 (203M), 0.6–1.2B effective tokens | ~1–3 h on 1–8 GPUs | **$5–30/run** |
| Model C d20 (561M), full nanochat pipeline | ~4h on 8×A100/H100 | **~$100–125** |
| Model B CPT (1.7B, ~200M–1B tokens, full-param) | ~1.5–6 H100-h | **$6–25/run** |
| Model B CPT (4B) | ~3–11 H100-h | **$12–45/run** |
| SFT/LoRA instruction stage | ~1–2 GPU-h on L40S/A100 | **$2–5** |
| Whole experimental program (incl. iteration ×3–5) | | **≈ $300–800 total** |

Compute is a rounding error next to the human work (corpus curation, native-speaker review, exhibit build). Budget accordingly. Treat every figure above as a **hypothesis**: run a 1%-scale throughput benchmark before quoting costs publicly — the Model B ranges in particular may be optimistic.

### 7.3 Event serving (local-first — venue wifi is the #1 failure mode)

- **Model B (1.7–4B)** → native on an M-series MacBook via Ollama/MLX, GGUF Q4_K_M (~40–80+ tok/s even on M1; MLX decode wants 32GB unified memory). Zero network.
- **Model C (203M)** → **fully in-browser** in a Next.js kiosk page via WebLLM or transformers.js v4 (WebGPU): 100M–1.5B at Q4 is comfortable (~1–2GB), offline once cached, and it lives naturally in the kreyol.chat site. Pre-cache before doors open. (This is why Model C stays Llama/GPT-2-arch-compatible.)
- **Model A (Claude API)** → the only internet-dependent station. Mitigations: dedicated 5G/LTE hotspot as primary uplink; graceful degradation to cached example responses; pre-rendered clips.
- If anything must be hosted: Modal with one keep-warm replica during event hours (eat the idle cost), or HF Inference Endpoints for custom weights. Fly GPUs are deprecated — do not build on them.

---

## 8. The exhibit — "Anndan Lespri Modèl la" (Step Inside a Kreyòl Language Model)

Same arc as Jaden Lakou: interaction → explanation → cultural meaning → participation.

| Station | Experience | Tech |
|---|---|---|
| **1. Kijan AI wè fraz sa a?** | Type *Dèyè mòn gen mòn* → see it split by cl100k/o200k vs our Kreyòl tokenizer; token counts, cost framing, which words survive intact | Client-side tokenizer JS (tiktoken wasm + our BPE) — fully offline |
| **2. Fini pwovèb la** | Visitor predicts the ending of *Piti piti…*; then A, B, C each complete it; who *knows* the proverb vs. who generated a plausible sentence? | B local (Ollama), C in-browser, A via API w/ fallback |
| **3. Literal oswa sans viv la?** | A proverb → literal translation vs. lived meaning vs. usage situation, across models; grounded by dictionary lookups | Models + **the fast dictionary as retrieval layer** (facts vs. interpretation — labeled) |
| **4. Montre modèl la** | *Kilès ki sonnen pi natirèl?* — fluent visitors rank outputs; consented, recorded, reviewed later into the community eval set | Simple form + consent flow (§10) |
| **5. Kisa ki fòme modèl sa a?** | The nutrition label: token counts, sources, %authored vs translated vs synthetic, vocab size, params, known weaknesses, underrepresented dialects, who reviewed it | Static from provenance DB |
| **6. Anndan tèt modèl la** | Attention-map visualization of Model C on a proverb — the Vaswani et al. mechanism, on Kreyòl. Presented honestly: one internal routing signal, *not* "what the model thought" | Precomputed or in-browser attention weights |
| **(Bonus) Gade l ap aprann** | Checkpoint slider: the same prompt at 10M → 1B tokens of training; watch Kreyòl emerge | Pre-generated outputs per Pythia-style checkpoint |

**Dictionary integration** (the grounding layer): word-tap anywhere → definition, pronunciation, POS, example, related words, proverb appearances (KV store, ~100ms) + clearly-labeled model-generated contextual explanation. Teaches the retrieval-vs-generation distinction honestly.

---

## 9. Voice (realistic path, licensing-aware)

**State of the world (all verified):** no commercial TTS API supports Haitian Creole (Google/Azure/Polly/ElevenLabs checked). Open options only:

- **facebook/mms-tts-hat** (VITS, 16kHz, single-speaker): works, intelligible-but-dated, **CC-BY-NC 4.0** — a company-run event plausibly reads as commercial use under a conservative reading. Don't build on it without counsel or Meta permission.
- **SpeechT5 route** (MIT base) / Sesame CSM (Apache): architecture-clean; existing community HT fine-tunes have license gaps (one no-license, one gated) — treat as inspiration, not dependencies.
- **The good news:** VITS/SpeechT5 fine-tuning is low-resource-friendly — quick voice adaptation from **~80–150 clean clips**, production-ish single-speaker quality from **~2–10 hours** ([ylacombe/finetune-hf-vits](https://github.com/ylacombe/finetune-hf-vits)). Recording 1–3 hours of one consented community speaker is feasible and becomes its own story: *"nou antrene vwa pa nou"* — with multiple speakers over time, avoiding one false "universal Haitian voice."

**ASR** (listen-and-repeat): off-the-shelf Whisper is weak on HT; use MMS-ASR's `hat` adapter ([facebook/mms-1b-all](https://huggingface.co/facebook/mms-1b-all)) or a fine-tuned Whisper; near-field mic, short scripted phrases; expect moderate accuracy in a noisy venue. SOTA context: monolingual wav2vec2 on ~1.4k hrs reached ~WER 33.7 (Interspeech 2025) — HT ASR is genuinely hard; set expectations.

**Progression:** (1) event: pre-rendered/human audio + optionally a consented fine-tuned voice; (2) community pronunciation collection with consent (the Common Voice gap is total — 2.7 minutes exist); (3) longer-term: a multi-speaker community voice model.

---

## 10. Data governance & consent (Station 4 is a governance artifact, not just a demo)

**Precedent:** Te Hiku Media (Māori) — community-donated speech under the [Kaitiakitanga License](https://github.com/TeHikuMedia/Kaitiakitanga-License): data as *taonga* (treasure) under guardianship, not ownership; use requires permission; commercial use requires explicit consent; derivatives inherit the license; benefits flow back to the community. Their ASR (~92% accuracy) proves community-owned pipelines work. Their ["Whisper is another case study in Colonisation"](https://blog.papareo.nz/whisper-is-another-case-study-in-colonisation/) post is required reading: *"The communities from where the data was collected should decide whether their data should be used and for what."* (Note: the Kaitiakitanga license itself is not directly reusable — it's tied to Māori tikanga and requires their permission — but the *pattern* is.)

**Frameworks:** CARE Principles (Collective benefit, Authority to control, Responsibility, Ethics) and OCAP (Ownership, Control, Access, Possession).

**Verified gap:** no Kaitiakitanga/OCAP-equivalent exists for Haitian Creole or any Caribbean creole. Our consent design can be an early, honest step — not claiming to *be* community governance, but practicing its principles:

1. Explicit, plain-language (Kreyòl + English) consent at Station 4: what's collected, what it's for, that it's reviewed before any use.
2. Contributions go to a *review queue*, never silently into training.
3. Provenance-tagged as community-contributed; contributors can withdraw — stated honestly: withdrawal removes a contribution from all *future* datasets and training runs, but removing it from already-trained weights would require retraining. This is why review happens *before* anything is trained on.
4. Publish the nutrition label (Station 5) as the standing transparency commitment.
5. Longer-term: convene Haitian community stakeholders (Akademi Kreyòl Ayisyen, MIT-Haiti, diaspora orgs) on whether a Kreyòl community-data license should exist — CreolePro as facilitator, not owner.

---

## 11. Phased plan

**Phase 0a — Rights, corpus, tokenizer, fertility (≈$0, CPU only)**
Rights matrix + split registry *first*, then corpus v0 (rights-clear sources only: MADLAD-clean + Wikipedia + owned material; unresolved sources quarantined), provenance-tagged, with a stratified quality audit. Train the 16–24k Kreyòl tokenizer (nanochat rustbpe, pinned commit). Measure fertility vs cl100k/o200k/Llama-3/Gemma/NLLB + Claude via `count_tokens` (novel numbers — protocol in §3.3). *Station 1 is already demoable after this phase.*

**Phase 0b — Base-model probe (≈$5–10)**
Staged probe of the *base* checkpoints (Qwen3-Base / SmolLM3-Base / gemma-3-4b-pt) on FLORES+ **dev**: bits-per-byte + few-shot scorecard → pick Model B's base. Full runbook: [phase-0.md](./phase-0.md).

**Phase 1 — Model C v0 (≈$30–100)**
nanochat d12 on corpus v0, multi-epoch, Pythia-style checkpoints. Micro-model mixture ablations. GGUF + browser conversion validated *early*. *Stations 2/6/bonus become demoable.*

**Phase 2 — Model B (≈$25–75)**
Full-param CPT (85–90/10–15 mix) via Axolotl on Modal; instruction stage from reviewed Aya-haitian + native-seeded self-instruct; LoRA acceptable here.

**Phase 3 — Evaluation (human time, not compute)**
CreoleVal + FLORES+ + construction suite on A/B/C; recruit native-speaker reviewers; build the human-eval rubric; assemble nutrition labels.

**Phase 4 — Exhibit build**
Next.js kiosk pages (in-browser Model C, tokenizer viz, checkpoint slider), Ollama setup on event MacBooks, dictionary integration, consent flow, **a public-interaction safety pass** (toxic-output probing, memorized-PII checks, prompt-abuse handling, kiosk auto-reset between visitors), 5G hotspot + fallbacks, dry run.

**Phase 5 — Post-event / ongoing**
Fold reviewed Station 4 data into eval set v2. Synthetic-data experiment (no/unreviewed/reviewed). Voice recording with consented speakers. Community-license conversation. Consider publishing: the tokenizer-fertility numbers, the construction suite, and the models (license TBD with community input) — each is a genuine contribution.

---

## 12. Open questions

1. **Event date & venue** → sets the Phase 4 deadline and hardware plan.
2. **Legal review**: Kreyòl-MT dataset ("other" license) and MADLAD terms vs. our commercial status; MMS CC-BY-NC; per-task CreoleVal licenses if any of it is used for *training*.
3. **Reviewer recruitment & compensation**: how many fluent speakers, from where (event partners? Akademi contacts?), paid how.
4. **Publish or not**: do we release Model C + tokenizer openly, and under what license — decided *with* community input (§10), not before.
5. **MIT-Haiti / DeGraff outreach**: worth a direct conversation about Platfòm materials and endorsement? (He co-authored our eval suite; the alignment is real.)
6. **Claude/GPT-4o FLORES numbers for HT don't exist publicly** — running and publishing them would be a genuine contribution and hard evidence for the quality-gap story.

---

## 13. Key sources

**Anchors:** Vaswani et al. 2017, *Attention Is All You Need* — [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) · Robinson et al. 2024, *Kreyòl-MT* — [arXiv:2405.05376](https://arxiv.org/abs/2405.05376) · Lent et al. 2024, *CreoleVal* — [arXiv:2310.19567](https://arxiv.org/abs/2310.19567) / [TACL](https://aclanthology.org/2024.tacl-1.53/)

**Tokenization & quality gap:** Petrov et al. 2023 — [arXiv:2305.15425](https://arxiv.org/abs/2305.15425) · Robinson et al. 2023, *ChatGPT MT* — [WMT 2023](https://aclanthology.org/2023.wmt-1.40/) · Ahia et al. 2023 (no HT coverage) — [EMNLP](https://aclanthology.org/2023.emnlp-main.614/) · Guinea-Bissau religious-data — [arXiv:2504.02674](https://arxiv.org/html/2504.02674)

**Training:** nanochat — [github.com/karpathy/nanochat](https://github.com/karpathy/nanochat) · Biderman et al., *LoRA Learns Less…* — [arXiv:2405.09673](https://arxiv.org/abs/2405.09673) · Sailor2 — [arXiv:2502.12982](https://arxiv.org/abs/2502.12982) · Swallow — [arXiv:2404.17790](https://arxiv.org/abs/2404.17790) · Muennighoff et al., data-constrained scaling — [arXiv:2305.16264](https://arxiv.org/abs/2305.16264) · Pythia — [arXiv:2304.01373](https://arxiv.org/abs/2304.01373) · Gururangan et al., *Don't Stop Pretraining* — [arXiv:2004.10964](https://arxiv.org/abs/2004.10964) · mBART — [arXiv:2001.08210](https://arxiv.org/abs/2001.08210) · XLM-R — [arXiv:1911.02116](https://arxiv.org/abs/1911.02116) · InstructGPT — [arXiv:2203.02155](https://arxiv.org/abs/2203.02155) · LIMA — [arXiv:2305.11206](https://arxiv.org/abs/2305.11206) · Modal — [pricing](https://modal.com/pricing), [long-training](https://modal.com/docs/examples/long-training)

**Data:** MADLAD-400 — [HF](https://huggingface.co/datasets/allenai/MADLAD-400) · Kreyòl-MT data — [HF](https://huggingface.co/datasets/jhu-clsp/kreyol-mt) · FLORES+ — [HF](https://huggingface.co/datasets/openlanguagedata/flores_plus) · Aya — [HF](https://huggingface.co/datasets/CohereLabs/aya_collection) · OPUS en–ht — [API](https://opus.nlpl.eu/opusapi/?source=en&target=ht&preprocessing=moses) · WMT11 Haiti pack — [statmt.org](https://www.statmt.org/wmt11/featured-translation-task.html)

**Linguistics & governance:** DeGraff, *Creole Exceptionalism* — [Language in Society 2005](https://www.cambridge.org/core/journals/language-in-society/article/abs/linguists-most-dangerous-myth-the-fallacy-of-creole-exceptionalism/25195A225AF5A6976E3EFEC8D53C42A8), [Language 80(4)](http://lingphil.mit.edu/papers/degraff/degraff-lang-80-04.pdf) · MIT-Haiti — [haiti.mit.edu](https://haiti.mit.edu/resources/) · Kaitiakitanga License — [github.com/TeHikuMedia](https://github.com/TeHikuMedia/Kaitiakitanga-License) · Papa Reo on Whisper — [blog.papareo.nz](https://blog.papareo.nz/whisper-is-another-case-study-in-colonisation/) · CARE — [GIDA](https://datascience.codata.org/articles/10.5334/dsj-2020-043)

**Speech & serving:** MMS-TTS hat — [HF](https://huggingface.co/facebook/mms-tts-hat) · VITS fine-tuning — [ylacombe/finetune-hf-vits](https://github.com/ylacombe/finetune-hf-vits) · MMS-ASR — [HF](https://huggingface.co/facebook/mms-1b-all) · WebLLM — [github.com/mlc-ai/web-llm](https://github.com/mlc-ai/web-llm) · transformers.js — [github.com/huggingface/transformers.js](https://github.com/huggingface/transformers.js/) · GGUF conversion — [llama.cpp discussion](https://github.com/ggml-org/llama.cpp/discussions/19815)

**Known-unverified items** (flagged inline throughout): exact HT fertility on o200k/Llama-3/Gemma (unpublished — we'll measure); whether HT is in Qwen3's 119 / Gemma 3's 140+ language lists; Kreyòl-MT per-direction Haitian chrF (in PDF figures only); nanochat d20 CORE score (conflicting sources); Modal region multipliers; jsbeaudry TTS model licenses (gated repos).
