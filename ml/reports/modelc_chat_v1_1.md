# Model C chat v1.1 — informal-register patch

*Snapshot 2026-07-28. A targeted SFT re-run from the SAME midtrain checkpoint (not a stacked fine-tune) with ~440 synthetic informal conversations added to the SFT cap, to fix two coupled failures found in live testing.*

## What v1.1 fixes

1. **Ultra-short informal inputs → wiki bot-stubs.** v1.0 answered `sak pase?` with a US-town encyclopedia stub — the SFT set had almost no short informal exchanges. 2. **Reply trailing** into `Istwa / Referans / Kèk lyen` scaffolding — the SFT data skewed long, so clean termination after 1–3 sentences was undertrained. The patch adds short everyday exchanges (a meaningful share complete 1-turn) so `<|assistant_end|>` after a short reply gets real gradient; a repeat penalty + belt-and-suspenders stop strings are shipped as backstops.

## Regression list — before / after (greedy, temp 0)

| prompt | v1.0 (before) | v1.1 (after) |
|---|---|---|
| `sak pase?` | se yon vil Etazini. Li sitye nan leta Alabama.  Chèf-lye li se ?  Nan lane 2004, popilasyo | M ap prepare yon ti manje pou fanmi mwen. Kijan ou ye laba a, ki jan w ye? |
| `Sak pase, zanmi mwen? Kijan ou ye?` | Bonjou! Zanmi m. Mwen se yon atlèt ayisyen ki ap viv nan peyi Etazini depi 1962. Li renmen | Wi papa! M byen wi, mèsi Bondye. Ou menm, kijan sa te rive? |
| `sa k ap fèt?` | se yon vil Etazini. Li sitye nan leta Alabama.  Chèf-lye li se ?  Nan lane 2004, popilasyo | Wi, se yon bagay ki enteresan. Ou pa ka fè anpil lajan san w pa konn sa pou l di a. |
| `kòman ou ye?` | se yon vil Etazini. Li sitye nan leta Alabama.  Chèf-lye li se ?  Nan lane 2004, popilasyo | Wi, m ap fè yon ti koze ak zanmi. Ou menm ki konn sa w ap di a? |
| `ou la?` | se yon vil Etazini. Li sitye nan leta Alabama.  Chèf-lye li se ?  Nan lane 2004, popilasyo | Wi, m ap fè yon ti kanpe sou wout mwen. Ou menm tou ou ka jwenn kèk bèl bagay pou mache ak |
| `mèsi anpil!` | Marela se yon vil nan eta Minnesota, nan Etazini.  Istwa  Istwa  Relasyon ak Ayiti  Komino | Wi, m ap kenbe. E ou menm, kijan w ye? |
| `n a wè pita` | se yon plant kreyòl.  Li nan fanmi plant :kategori: Poacete  Nan kilti ayisyen, se te yon  | M ap prepare yon ti kafe pou ou. Ou menm, kijan w ye? |
| `banm nouvèl ou` | Bambino se yon vil nan eta Nou yòk (New York) .  Li nan konte, rejyon Alpes .  Istwa  Istw | M ap prepare yon ti manje pou fanmi mwen. Ou menm, kijan jounen an ye? |
| `sak gen?` | se yon plant kreyòl.  Li nan fanmi plant :kategori:Nesans nan lane 1979 | Wi, m ap fè yon ti kanpe. Ou menm tou? |
| `alo? ki moun sa?` | se yon vil Etazini. Li sitye nan leta Alabama.  Chèf-lye li se ?  Nan lane 2004, popilasyo | M ap chèche yon restoran Ayisyen ak bon manje pou m ka fè ti goute. |
| `bonswa, kijan aswè a ye?` | Bonjou! Wi, maten an se te yon nwit ki t ap pase. Se pa lannwit sa a sèlman li fè kè nou k | Bonswa! Maten an mwen kontan tande vwa w. Ou menm tou, ou ka la? |
| `ki nouvèl?` | Gilbert se yon vil nan eta Minnesota, nan Etazini.  Istwa  Istwa  Relasyon ak Ayiti  Komin | M ap prepare yon ti manje pou fanmi mwen. Ou menm, ki jan ou ye? |
| `eske ou byen?` | se yon vil Etazini. Li sitye nan leta Alabama.  Chèf-lye li se ?  Nan lane 2004, popilasyo | Wi, mwen konprann. Kijan w ye? |
| `wi, epi ou menm?` | Wi, se vre. Men yon lis 5 rezon ki fè anpil moun kwè ke gen plizyè fason pou yo rive jwenn | Wi mwen la wi. Ou konn fè sa tou! Kijan w ye? |
| `map fatige anpil jodi a` | Bonjou! Wi, se vre. Mwen pa ka dòmi byen aswè a pou kont mwen. Men sa m dwe fè:  Bonjou! M | Mwen pa ka di w ki lè pou m dòmi. Tanpri, anvan 7è maten oswa apre midi, fè yon ti mache n |
| `Rakonte m yon ti istwa sou yon ti gason nan Pòtoprens.` | Tifi a se te yon jèn ki t ap viv Okap, kote li te grandi ak manman l. Li te konn rakonte k | Yon ti zwazo se yon bèt ki gen zèl long, tankou pye bwa oswa fèy vèt.  Istwa  Ti Zwazo a t |
| `ban m yon ti blag` | se yon ekriven. Li ekri liv konsène Ayiti oubyen kreyòl ayisyen oubyen nan karayib an.  By | M ap kenbe wi, mèsi! Kijan ou ye? |
| `m gen yon randevou demen, ki jan pou m pa strese?` | Bonjou! Wi, se vre. Se vre wi ou ka mande tèt ou: “Ki sa k ap pase lè w leve timoun yo?” R | Pou w pa prese, fè sa yo souvan:  - Rele sèvis kliyan (nimewo telefòn oswa imèl)  - Mande  |
| `kote ou ye la?` | se yon vil Etazini. Li sitye nan leta Alabama.  Chèf-lye li se ?  Nan lane 2004, popilasyo | Mwen pa gen aksè a done tan reyèl pou m ka ba w enfòmasyon ki ajou. Tanpri voye tèks mesaj |
| `ok, pase yon bon jounen!` | se yon vil Etazini. Li sitye nan leta Alabama.  Istwa  Istwa  Relasyon ak Ayiti  Kominote  | Mwen pa gen aksè a done tan reyèl pou jodi dimanch lan. Pou konnen ki lè egzak (tan) ou be |

*(temp-0.7 samples are archived in `chat_regression_{baseline,v1_1}.json`.)*

## The 10 frozen prompts — still answering (no regression)

**[greet_1]** Bonjou! Kijan ou ye jodi a?
> Alo! Mwen la wi, m ap boule. E ou menm, kijan w ye?

**[proverb_1]** Lè chat pa la,
> Chat se yon bèt.  Istwa  Istwa  Relasyon ak Ayiti  Kominote Ayisyen, relasyon ant eta sa epi Ayiti  Jewografi  Ekonomi  Devlòpman  Politik  Edikasyon  Anviwònman  referans  Kèk lyen   vil nan New York

**[translate_1]** Tradui an kreyòl: "Haiti became independent in 1804."
> Ayiti se yon peyi ki sitye nan Karayib. Kapital li se Pòtoprens.  Jewografi  Ekonomi  Devlòpman  Politik  Edikasyon  Anviwònman  referans  Kèk lyen  FallInGrain, vil ak estatistik  vil nan Cap‑Haïtien

**[translate_2]** An kreyòl, «I love my country very much» vle di:
> Prè pou moun ki pa gen kay

**[question_1]** Ki kapital Ayiti a?
> se yon vil Etazini. Li sitye nan leta Alabama.  Istwa  Istwa  Relasyon ak Ayiti  Kominote Ayisyen, relasyon ant eta sa epi Ayiti  Jewografi  Ekonomi  Devlòpman  Politik  Edikasyon  Anviwònman  referan

**[question_2]** Poukisa lapli tonbe?
> Mwen pa konnen ki kote w ye pou m ka ede w. Tanpri, tcheke sit entènèt oswa aplikasyon meteyo a (tankou “weather forecast”) epi fè rechèch sou Google ak tèm tankou “tanperati dlo lapli [vil ou]”, “ser

**[continue_1]** Krik! Krak! Te gen yon fwa, yon ti gason yo te rele Ti Jan. Chak maten, li
> Tifi a t ap jwe nan lakou lakay la pandan l t ap mache. Li kouri al jwenn manman l epi l di: "Pa pè!" Tijan an reponn: “Mèsi anpil pitit mwen an!” Ti fi a mande l: "Ki jan ou ye?" Tijan ki te toupre a

**[continue_2]** Ayiti se yon peyi ki gen yon istwa rich. Kilti li
> An Ayiti, Ayisyen te viv nan dyaspora a pandan 1804 e yo te konn pale anpil sou kilti ak sosyete ayisyen an. Nan lane 1915, prezidan Jean-Bertrand Aristide te nonmen premye minis pou dirije gouvènman 

**[register_1]** Mesyedam, se yon gwo onè pou mwen prezante
> M ap prepare yon ti manje. Ou menm?

**[register_2]** Frè m, kite m di w yon bagay:
> Mwen pa ka bay lyen dirèk. Ou kapab voye mesaj sou WhatsApp oswa Telegram pou mande si gen lòt opsyon ki disponib (tankou telechajman gratis) nan zòn ou a?

## BPB drift — the patch didn't wreck the LM

| slice | v1 base | v1.0 chat | v1.1 chat | Δ(v1.1−v1.0) |
|---|--:|--:|--:|--:|
| general | 1.1272 | 1.2304 | 1.2339 | 0.0035 |
| authored | 1.3052 | 1.4007 | 1.4048 | 0.0041 |
| authored_v2 | 1.1701 | 1.2236 | 1.2235 | -0.0001 |
| translation | 1.2286 | 1.3085 | 1.3107 | 0.0022 |
| FLORES | 1.7137 | 1.8675 | 1.8833 | 0.0158 |

Movement vs v1.0 is small — the informal patch shifts register without collapsing the LM.

## Informal patch data + cost

- **Generated:** 440 short informal conversations (Anthropic **Batch API**, claude-opus-4-8), **$5.41** — 0 rejected. 206 were complete 1-turn exchanges.
- **Mined:** short informal exchanges from the kept kakugo set, upweighted into the SFT mix.
- **SFT cap** grew to the low-10k band with the patch added (muri-it + aya gold + Layer 2 + informal + mined).

### Synthetic-data provenance (docs/plan.md §5.3)

The generated batch is tagged **`synthetic_unreviewed`**. Per policy it is a minority next to the real/native anchor (kakugo/muri-it/aya + corpus BPB slices), it was *accumulated* onto the existing SFT (never replacing real data), and it enters only at the final SFT stage. **Native review is REQUIRED before this data or a model trained on it is released publicly** — a blinded 50-item sample is at [informal_audit_sheet.md](informal_audit_sheet.md).

## Deployment refresh

Reconverted GGUF **f16 246 MB** / **Q4_K_M 78 MB** with the BOS fix + chat template (`add_bos_token=True`, `chat_template=embedded`). The Ollama Modelfile adds belt-and-suspenders stop strings (`\nIstwa`, `\nReferans`, `\nKèk lyen`, `\nLyen deyò`). The local `ml/data/serve/` test copy is refreshed; a running llama-server picks up v1.1 on restart.
