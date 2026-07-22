# Base-model probe — Workstream D (Phase 0b)

_Snapshot 2026-07-21. Precision: **unquantized bf16** (transformers, `torch_dtype=bfloat16`). Selection corpus: FLORES+ **dev** only (`openlanguagedata/flores_plus` @ `b3a5298db572`); `final_devtest` untouched. GPU: Modal L40S._

**Question.** Which pretrained base starts from the strongest Kreyòl position for Model B's continued-pretraining? Reputation doesn't settle it (none publish HT evals); a few dollars of GPU does.

## Final pick — **`google/gemma-3-4b-pt`** *(naturalness folded in 2026-07-22)*

**`google/gemma-3-4b-pt`** wins on every axis: authored-Kreyòl BPB **1.170** (primary), eng→hat chrF2++ **43.1** (~1.7× the runner-up; 43.2 on full-dev), the only probe-proverb recall, and the blinded naturalness rubric (**avg 2.7** vs 1.2–1.9 for the field — the only model scoring 3s on translation, register, and continuation). Naturalness provenance: scored blind on the 10-prompt sheet by a **working-level (non-native) reader**; only the extremes are treated as robust (Gemma clearly top, Qwen3-1.7B clearly floor — the 1.6–1.9 middle cluster is within noise); native-speaker review is deferred to Phase 3 per plan §6.2. **License caveat stands** (Gemma Terms vs the community-project Apache bar — a Phase 2 decision; best Apache-clean alternative: Qwen3-4B).


## Scorecard

BPB = bits-per-byte (primary, ↓). chrF2++ / spBLEU on the 250-item dev subset (↑). Proverbs = exact-continuation hits / near-misses out of 15. Naturalness = 1–3 blinded rubric avg (working-level reader; extremes robust, middle cluster within noise).

| Model | Params | BPB authored ↓ | BPB full ↓ | chrF2++ e→h | spBLEU e→h | chrF2++ h→e | spBLEU h→e | Proverbs hit/near | Naturalness |
|---|---|---|---|---|---|---|---|---|---|
| `Qwen/Qwen3-1.7B-Base` | 1.7B | **1.731** | 2.080 | 16.5 | 2.3 | 33.5 | 14.5 | 0/0 | 1.2 |
| `Qwen/Qwen3-4B-Base` | 4B | **1.534** | 1.815 | 25.0 | 5.3 | 47.3 | 26.6 | 0/0 | 1.9 |
| `HuggingFaceTB/SmolLM3-3B-Base` | 3B | **2.485** | 2.869 | 17.6 | 3.0 | 37.7 | 17.4 | 0/0 | 1.6 |
| `google/gemma-3-4b-pt` ⭐ | 4B | **1.170** | 1.355 | 43.1 | 20.6 | 58.0 | 37.1 | 1/0 | **2.7** |
| `meta-llama/Llama-3.2-3B` _(control)_ ⭐ | 3B | **1.291** | 1.714 | 25.6 | 6.2 | 45.7 | 25.1 | 0/1 | 1.7 |

⭐ = promoted to the full-dev MT stage.

## Full-dev refinement (top 2, 500 sentences)
  
_Bounded to a seeded 500-sentence sample (not the full 992) for cost: Gemma-3's multimodal `generate` path is ~20× slower than the plain causal LMs. Both finalists are scored on the identical sample._

| Model | chrF2++ e→h | spBLEU e→h | chrF2++ h→e | spBLEU h→e |
|---|---|---|---|---|
| `google/gemma-3-4b-pt` | 43.2 | 20.5 | 58.1 | 37.3 |
| `meta-llama/Llama-3.2-3B` | 25.9 | 6.2 | 44.8 | 23.6 |

## Findings

- **google/gemma-3-4b-pt starts from the strongest Kreyòl position** on both automated axes — lowest authored BPB (1.170) and highest eng→hat chrF2++ (43.1, ~1.7× the runner-up). It is also the only model with an exact proverb hit.
- **SmolLM3-3B is the *worst* on BPB** despite its strong-French reputation (the reason it was hypothesized to transfer well to a French-lexifier creole). French coverage did **not** translate into a Kreyòl exposure advantage here — a direct, if negative, answer to that open question.
- **The control (meta-llama/Llama-3.2-3B) is not the floor** — it ranks #2 of 5 on BPB. Llama 3.2 lists 8 official languages (HT not among them), yet it beats the Qwen3 and SmolLM3 bases on authored-Kreyòl BPB.
- Absolute quality is modest across the board — these are base checkpoints doing completion-style few-shot, and greedy decoding makes them loop (visible in the naturalness sheet). The point is **relative** starting position for CPT, not usable output.
- These are the **"before" numbers** banked for the later before/after adaptation comparison (Model B CPT).
- **License caveat (decisive for the final pick).** The leader `google/gemma-3-4b-pt` is under the **Gemma Terms of Use**, not Apache-2.0. plan.md §3.1 requires an Apache-2.0-clean base for this community project (Gemma listed *only if its custom license is acceptable*). So the quality ranking and the license gate are **separate decisions**: if the Gemma terms are ruled out, the best Apache-2.0-clean base is **`Qwen/Qwen3-4B-Base`** (next Apache-2.0 base by BPB). Llama-3.2 is also non-Apache (Llama license) and is the control, not a candidate.

## Proverb completion — raw outputs (Station 2 preview)

Each probe proverb is cut at its midpoint; the model completes the line (5-shot format, greedy). ✓ = exact continuation, ≈ = near-miss (chrF ≥ 50). The 15 probe proverbs are the held-out probe split — in no training or eval set.

_Note: the two format exemplars are real proverbs (`Piti piti zwazo fè nich li.` / `Men anpil, chay pa lou.`); models with no recall for the target proverb often just echo the exemplar tail (`…chay pa lou`), so a low chrF reflects echo, not only ignorance. A genuine hit (e.g. Gemma on `Dèyè mòn gen mòn.`) is therefore the signal that matters._

**1. Dèyè mòn gen mòn.**  
_“Behind mountains, there are more mountains.”_  
prompt → `Dèyè mòn` · gold → `gen mòn.`

| Model | Completion | chrF | |
|---|---|---|---|
| `qwen3-1.7b` | , chay pa lou. | 5.3 |  |
| `qwen3-4b` | , mwen pa lou. | 11.9 |  |
| `smollm3-3b` | , mòn. | 43.5 |  |
| `gemma3-4b` | gen mòn. | 100.0 | ✓ |
| `llama3.2-3b` | , mòn pa lou. | 20.0 |  |

**10. Sak vid pa kanpe.**  
_“An empty sack can't stand up.”_  
prompt → `Sak vid` · gold → `pa kanpe.`

| Model | Completion | chrF | |
|---|---|---|---|
| `qwen3-1.7b` | èt, chay pa lou. | 14.1 |  |
| `qwen3-4b` | è, men anpil, chay pa lou. | 19.6 |  |
| `smollm3-3b` | an, chay pa lou. | 17.0 |  |
| `gemma3-4b` | , pa genyen kòd. | 17.2 |  |
| `llama3.2-3b` | an, men pa lou. | 20.3 |  |

**11. Zanmi nan nesesite se zanmi tout bon vre.**  
_“A friend in need is a true friend indeed.”_  
prompt → `Zanmi nan nesesite se` · gold → `zanmi tout bon vre.`

| Model | Completion | chrF | |
|---|---|---|---|
| `qwen3-1.7b` | fè li. | 5.2 |  |
| `qwen3-4b` | kòmèt. | 5.5 |  |
| `smollm3-3b` | zanmi. | 21.6 |  |
| `gemma3-4b` | zanmi nan lavi. | 21.1 |  |
| `llama3.2-3b` | yon pwovèb. | 8.7 |  |

**12. Bouch granmoun santi, men pawòl li pa santi.**  
_“An elder's mouth may smell, but their words don't.”_  
prompt → `Bouch granmoun santi, men` · gold → `pawòl li pa santi.`

| Model | Completion | chrF | |
|---|---|---|---|
| `qwen3-1.7b` | anpil, chay pa lou. | 14.8 |  |
| `qwen3-4b` | anpil, chay pa lou. | 14.8 |  |
| `smollm3-3b` | chay pa lou. | 10.6 |  |
| `gemma3-4b` | anpil, chay pa lou. | 14.8 |  |
| `llama3.2-3b` | anpil, chay pa lou. | 14.8 |  |

**19. Bay kou bliye, pote mak sonje.**  
_“The one who strikes forgets, the one who bears the scar remembers.”_  
prompt → `Bay kou bliye,` · gold → `pote mak sonje.`

| Model | Completion | chrF | |
|---|---|---|---|
| `qwen3-1.7b` | chay pa lou. | 7.2 |  |
| `qwen3-4b` | men anpil, chay pa lou. | 8.9 |  |
| `smollm3-3b` | chay pa lou. | 7.2 |  |
| `gemma3-4b` | bay kou bliye. | 9.1 |  |
| `llama3.2-3b` | men kèk. | 8.6 |  |

**27. Twòp prese pa fè jou louvri.**  
_“Being in too much of a hurry doesn't make the day dawn faster.”_  
prompt → `Twòp prese pa` · gold → `fè jou louvri.`

| Model | Completion | chrF | |
|---|---|---|---|
| `qwen3-1.7b` | lou. | 16.7 |  |
| `qwen3-4b` | lou. | 16.7 |  |
| `smollm3-3b` | fè piti. | 13.7 |  |
| `gemma3-4b` | fè bon. | 12.6 |  |
| `llama3.2-3b` | fè li. | 17.4 |  |

**28. Chen gen fòs nan je mèt li.**  
_“A dog is brave in its master's yard.”_  
prompt → `Chen gen fòs nan` · gold → `je mèt li.`

| Model | Completion | chrF | |
|---|---|---|---|
| `qwen3-1.7b` | , men anpil, chay pa lou. | 8.6 |  |
| `qwen3-4b` | chèk. | 7.8 |  |
| `smollm3-3b` | . | 22.4 |  |
| `gemma3-4b` | bouch li. | 21.5 |  |
| `llama3.2-3b` | . | 22.4 |  |

**29. Malfini kouri di li wè, poul kouri di li pa vle.**  
_“The hawk runs and says it saw something; the chicken runs and says it doesn't want trouble.”_  
prompt → `Malfini kouri di li wè, poul` · gold → `kouri di li pa vle.`

| Model | Completion | chrF | |
|---|---|---|---|
| `qwen3-1.7b` | èt pa lou. | 12.0 |  |
| `qwen3-4b` | èt. | 5.5 |  |
| `smollm3-3b` | èt. | 5.5 |  |
| `gemma3-4b` | di li wè. | 20.6 |  |
| `llama3.2-3b` | èt kouri di li wè. | 53.1 | ≈ |

**30. Kabrit ki gen twòp mèt mouri nan solèy.**  
_“A goat with too many owners dies in the sun.”_  
prompt → `Kabrit ki gen twòp` · gold → `mèt mouri nan solèy.`

| Model | Completion | chrF | |
|---|---|---|---|
| `qwen3-1.7b` | ètètètètètètètètètètètètètètètèt | 2.9 |  |
| `qwen3-4b` | . | 15.7 |  |
| `smollm3-3b` | , fèt twòp. | 5.9 |  |
| `gemma3-4b` | kòd, pa ka fè kòd. | 4.7 |  |
| `llama3.2-3b` | , gen twòp. | 5.1 |  |

**31. Lè chat pa la, rat pran kay.**  
_“When the cat's away, the rats take over the house.”_  
prompt → `Lè chat pa la,` · gold → `rat pran kay.`

| Model | Completion | chrF | |
|---|---|---|---|
| `qwen3-1.7b` | chay pa lou. | 10.2 |  |
| `qwen3-4b` | men anpil, chay pa lou. | 11.8 |  |
| `smollm3-3b` | chay pa lou. | 10.2 |  |
| `gemma3-4b` | mouch la fè kòrs. | 7.3 |  |
| `llama3.2-3b` | li fè tèt. | 5.5 |  |

**36. Pitit se richès malere.**  
_“Children are the wealth of the poor.”_  
prompt → `Pitit se` · gold → `richès malere.`

| Model | Completion | chrF | |
|---|---|---|---|
| `qwen3-1.7b` | kreyòl. | 9.9 |  |
| `qwen3-4b` | yèt, men anpil, chay pa lou. | 12.4 |  |
| `smollm3-3b` | piti, chay se chay. | 11.0 |  |
| `gemma3-4b` | bon bagay, men li pa bon bagay. | 7.9 |  |
| `llama3.2-3b` | yon moun ki pa gen yon pèsonalite. | 12.7 |  |

**38. Lajan nan men, zanmi toupatou.**  
_“Money in hand, friends everywhere.”_  
prompt → `Lajan nan men,` · gold → `zanmi toupatou.`

| Model | Completion | chrF | |
|---|---|---|---|
| `qwen3-1.7b` | chay pa lou. | 13.9 |  |
| `qwen3-4b` | men anpil. | 11.0 |  |
| `smollm3-3b` | chay pa lou. | 13.9 |  |
| `gemma3-4b` | lajan nan men. | 11.6 |  |
| `llama3.2-3b` | men nan lajan. | 9.6 |  |

**42. Moun ki renmen ou di ou verite.**  
_“The person who loves you tells you the truth.”_  
prompt → `Moun ki renmen ou` · gold → `di ou verite.`

| Model | Completion | chrF | |
|---|---|---|---|
| `qwen3-1.7b` | , pa lou. | 8.2 |  |
| `qwen3-4b` | , ou pa lou. | 10.9 |  |
| `smollm3-3b` | , ou renmen li. | 15.0 |  |
| `gemma3-4b` | , pa gen okenn bagay ki ka fè li kite ou. | 14.8 |  |
| `llama3.2-3b` | , ou pa renmen li. | 14.4 |  |

**45. Dlo ki koule pa janm tounen nan sous.**  
_“Water that flows never returns to the spring.”_  
prompt → `Dlo ki koule pa` · gold → `janm tounen nan sous.`

| Model | Completion | chrF | |
|---|---|---|---|
| `qwen3-1.7b` | lou. | 8.3 |  |
| `qwen3-4b` | lou. | 8.3 |  |
| `smollm3-3b` | fè chay lou. | 6.5 |  |
| `gemma3-4b` | fè kòd. | 3.5 |  |
| `llama3.2-3b` | gen kè. | 6.0 |  |

**47. Wè pa di konnen.**  
_“Seeing doesn't mean knowing.”_  
prompt → `Wè pa` · gold → `di konnen.`

| Model | Completion | chrF | |
|---|---|---|---|
| `qwen3-1.7b` | lou. | 10.2 |  |
| `qwen3-4b` | lou, men anpil, chay pa lou. | 10.5 |  |
| `smollm3-3b` | fè zanmi. | 8.4 |  |
| `gemma3-4b` | fè, fè pa wè. | 4.9 |  |
| `llama3.2-3b` | wè, men wè pa wè. | 8.6 |  |

## Method & provenance

**BPB (bits-per-byte, primary).** `total_nll_bits / total_utf8_bytes` over the corpus **tokenizer_eval holdout** (never-trained docs, Workstream B definition). Each doc scored independently (no cross-doc context); the tokenizer's BOS id — or EOS id where no BOS exists (GPT-2 document-start convention) — is prepended as context and never scored; every real token is scored; docs longer than 2048 tokens are split into windows each re-seeded with the start token. Denominator is UTF-8 bytes, so BPB is **cross-tokenizer comparable**. Two slices:

- **authored-only** (primary signal): Wikipedia non-stub + owned docs — 225 docs, 429,711 B (scored in full).
- **full holdout**: 215 of 2109 holdout docs, seed-subsampled to a 700,000-byte budget (703,693 B).

**MT few-shot completion.** 5-shot, fixed template, greedy (`do_sample=False, num_beams=1`), stop at newline. Template per line: `English: …` / `Haitian Creole: …`. sacreBLEU signatures:
- spBLEU: `nrefs:1|case:mixed|eff:no|tok:flores200|smooth:exp|version:2.6.0`
- chrF2++: `nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no|version:2.6.0`

**External context (NOT comparable).** Robinson et al. 2023 (arXiv:2309.07423) report eng->hat MT quality for several systems on FLORES-style data; their numbers use a different prompt, pipeline, and dataset version and are NOT directly comparable to these few-shot base-completion scores — cited only for rough orientation.

## Runs, revisions, cost

| Model | Resolved revision | transformers | load s | total s |
|---|---|---|---|---|
| `Qwen/Qwen3-1.7B-Base` | `ea980cb0a6c2ae4b936e82123acc929f1cec04c1` | 4.55.0 | 6.3 | 159.1 |
| `Qwen/Qwen3-4B-Base` | `906bfd4b4dc7f14ee4320094d8b41684abff8539` | 4.55.0 | 8.5 | 203.7 |
| `HuggingFaceTB/SmolLM3-3B-Base` | `d78a42f79198603e614095753484a04c10c2b940` | 4.55.0 | 32.4 | 179.3 |
| `google/gemma-3-4b-pt` | `cc012e0a6d0787b4adcc0fa2c4da74402494554d` | 4.55.0 | 6.1 | 1878.8 |
| `meta-llama/Llama-3.2-3B` | `13afe5124825b4f3751f836b40dafda64c1ed062` | 4.55.0 | 8.3 | 139.6 |

**GPU cost (estimate).** ~69.7 GPU-min on Modal L40S ≈ **$2.26** at $1.95/GPU-h (list price; Modal's dashboard is authoritative). BPB adds no generation cost (forward pass only).

**Reproduce.** `uv run python -m probe.run --stage all` (smoke → main → fulldev → report). Modal auth required; weights cache in a Modal volume so reruns skip re-download. Raw results (with FLORES prompts/refs) stay under the git-ignored `ml/data/probe/`.
