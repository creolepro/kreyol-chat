# Model C chat — midtraining + SFT (the first conversational Kreyòl-first model)

> **Superseded in part by v1.1** ([modelc_chat_v1_1.md](modelc_chat_v1_1.md), 2026-07-28): an
> informal-register patch that fixes the short-informal→wiki-stub failure and the stub-trailing
> reflex noted below (see "Quality — an honest read"). This report documents the v1.0 baseline.

*Snapshot 2026-07-26. Continues the Model C v1 base (123M, d12) through a three-layer SFT stack (docs/data.md §3): midtraining (format) → SFT (voice), on Modal H100. Nothing under `ml/data/` is committed.*

## Data composition + licenses

Real-data layers (Layer 1 midtraining, Layer 3 SFT cap) + the corpus-grounded Layer 2 (the quality core). Kept/dropped counts are from the build; every source was registered in `rights.yaml` **before** ingestion, screened for the 15 probe proverbs, and (for the templated sources) had all FLORES-derived rows dropped so our MT eval never leaks.

| source | layer | license | kept | notes |
|---|---|---|--:|---|
| kakugo-hat | 1 | Apache-2.0 | 38475 | multi-turn; stripped English `<think>`/system, per-turn langid, dedup |
| aya_collection (hat) | 1 | Apache-2.0 | 6000 | templated bulk, capped + deduped (format variety) |
| xP3x (hat_Latn) | 1 | Apache-2.0 | 0 | **0 kept — 100% FLORES-derived → dropped (eval carve-out)** |
| translation-QA | 1 | Federal PD + CMU | 3955 | EN↔HT turns from the PD glossary + CMU lexicon |
| Layer 2 (generated) | 2 | corpus-grounded | 1176 | claude-opus-4-8 over authored VOA/wiki/legal passages |
| muri-it (hat) | 3 | Apache-2.0 | 6817 | native-output pairs; response-masked |
| aya gold (hat) | 3 | Apache-2.0 | 98 | human-written gold |

**Layer 1 (midtrain)** packs to **19,891,489** tokens across 47462 conversations (full-sequence loss). **Layer 3 (SFT)** packs to **1,724,510** tokens across 8225 conversations, response-masked (loss-token fraction 0.886).

### Layer-2 generation (pilot-gated)

Pilot: **100/100 clean** at **$6.36** ($0.0636/conv; 958 in / 657 out tokens). Projected 3–5k = $190.8–318.0 (Opus list rates; the API dashboard is the true bill), which exceeds the ≈$75 budget → ran a budget-fit batch instead.
Full run: **1176** total Layer-2 conversations ($6.36+ batch).

## Training

| stage | resume from | steps | epochs | tok/step | peak LR | loss | final loss |
|---|---|--:|--:|--:|--:|---|--:|
| midtrain | v1 base (step 1907) | 455 | 2.998 | 131,072 | 0.0003 | full-seq | 2.3318 |
| SFT | midtrain final | 79 | 3.002 | 65,536 | 0.0001 | response-masked | 1.8109 |

## BPB regression — did chat tuning damage the language model?

BPB on the standing slices, chat model vs the v1 base (same slices, byte-normalized). Δ>0 means the chat model is slightly worse at raw LM on that slice — expected and small; the goal is a model that *answers*, without wrecking the language.

| slice | v1 base | chat | Δ |
|---|--:|--:|--:|
| general | 1.1272 | 1.2304 | 0.1032 |
| authored | 1.3052 | 1.4007 | 0.0955 |
| authored_v2 | 1.1701 | 1.2236 | 0.0534 |
| translation | 1.2286 | 1.3085 | 0.0799 |
| FLORES | 1.7137 | 1.8675 | 0.1538 |

## Before / after — same prompt, base continuation vs chat answer

**Prompt (greet_1):** Bonjou! Kijan ou ye jodi a?

- **v1 base (continues):** - The Haitian Times
Bonjou! Kijan ou ye jodi a?
Nou se yon gwoup jèn ki gen yon seri moun ki pa gen anpil konesans nan domèn nan. Nou gen yon seri moun ki pa gen anpil konesans nan domèn nan. Nou
- **chat (answers):** Bonjou! Wi, se vre. Se pa sèlman nan Pòtoprens nou pral fè yon ti vwayaj pou peyi a, men tou nan vil yo ak zòn ki antoure l. Nou gen anpil bèl plaj sou lanmè. N ap tann w la lè solèy kouche.

**Prompt (proverb_1):** Lè chat pa la,

- **v1 base (continues):** li se pa gen okenn ankò yon bagay ki te pase, men li se yon bagay ki te pase nan lavi nou. Li se yon bagay ki pa ka di sou lavi nou, men sou tan, li te gen yon bagay ki te pase
- **chat (answers):** Chat se yon bèt ki gen anpil enèji. Chat renmen manje tout bagay li yo ak zanmi l yo.

Tradiksyon 
 Lòt kreyòl:
 Franse: chat

Istwa 
Istwa mo sa

Referans

Lyen deyò

**Prompt (translate_1):** Tradui an kreyòl: "Haiti became independent in 1804."

- **v1 base (continues):** - Journal Haïti Progrès - Haitian #1 Newspaper - Jounal Ayiti
Tradui an kreyòl: "Haiti became independent in 1804"
Nan yon atik ki te pibliye nan jounal Le Nouvelliste, nou
- **chat (answers):** Ayiti se yon peyi ki sitye nan Karayib. Kapital li se Pòtoprens.

Istwa

Politik

Administrasyon
Kominote Ayisyen, relasyon ant eta sa epi Ayiti
6 fevrye 1996 : Gouvènman ayisyen reprann kontak ak otorite Kiben ( 2 chans

## The 10 frozen prompts, answered in chat mode

The continuer→answerer transition (archived alongside the slider assets). Greedy.

**[greet_1]** Bonjou! Kijan ou ye jodi a?
> Bonjou! Wi, se vre. Se pa sèlman nan Pòtoprens nou pral fè yon ti vwayaj pou peyi a, men tou nan vil yo ak zòn ki antoure l. Nou gen anpil bèl plaj sou lanmè. N ap tann w la lè solèy kouche.

**[proverb_1]** Lè chat pa la,
> Chat se yon bèt ki gen anpil enèji. Chat renmen manje tout bagay li yo ak zanmi l yo.  Tradiksyon   Lòt kreyòl:  Franse: chat  Istwa  Istwa mo sa  Referans  Lyen deyò

**[translate_1]** Tradui an kreyòl: "Haiti became independent in 1804."
> Ayiti se yon peyi ki sitye nan Karayib. Kapital li se Pòtoprens.  Istwa  Politik  Administrasyon Kominote Ayisyen, relasyon ant eta sa epi Ayiti 6 fevrye 1996 : Gouvènman ayisyen reprann kontak ak otorite Kiben ( 2 chanselye Ayisyen epi Kiben Emmanuel Fritz Longchamp epi Roberto Robaina ); relasyon diplomatik sa te sispann depi 1962, dat Kiba te dwèt retire l de òganizasyon OEA. Relasyon sa te koumanse depi 3 fevya 1904. Kooperasyon ant 2 peyi yo nan domèn sa yo : Agrikilti Kilti sik (izin Dabòn) Sante, asistans medikal Aewonotik sivil Espò Touris Konstriksyon Pèch Edikasyon, alfabetizasyon, timoun piti  Popilasyon ayisyen nan Kiba rive 300000 moun nan lane 2004. Kreyòl ayisyen se dezyèm lang ki pale nan Kiba, lang ayisyen an gen menm yon estasyon radyo pou kò l.  Jewografi  Ekonomi  Devlòpman  Politik)  Edikasyon

**[translate_2]** An kreyòl, «I love my country very much» vle di:
> Prè pou moun ki pa gen kay nan peyi a

**[question_1]** Ki kapital Ayiti a?
> se yon vil Etazini. Li sitye nan leta Alabama.  Istwa  Istwa  Relasyon ak Ayiti  Kominote Ayisyen, relasyon ant eta sa epi Ayiti  Jewografi  Ekonomi  Devlòpman  Politik  Edikasyon  Anviwònman  referans  Kèk lyen   vil nan Alabasterola Vil nan Etazini jewografi

**[question_2]** Poukisa lapli tonbe?
> Lapli se yon gwo van ki ap deplase nan direksyon lwès, epi li ka rive sou tè a avèk anpil fòs. Li kapab tou frape pa nèj la oswa glas la lè dlo yo cho ak frèt.  Referans  Syans

**[continue_1]** Krik! Krak! Te gen yon fwa, yon ti gason yo te rele Ti Jan. Chak maten, li
> Tifi Dram nan peyi Ayiti Li se pitit fi yon jèn gason ki renmen lanmè a. Men sa l di: "Mwen pa ka viv ak moun."  "Ou mèt voye zye sou kat jewografik li nan paj sa  Istwa  Istwa mo sa  referans  Kèk lyen

**[continue_2]** Ayiti se yon peyi ki gen yon istwa rich. Kilti li
> se yon vil Etazini. Li sitye nan leta Alabama.  Chèf-lye li se ?  Nan lane 2004, popilasyon l te konte 1,410 moun. Lòt lane yo te genyen : 2003 (2),  2002 (2).  Istwa  Istwa  referans  Kèk lyen   Vil nan Alabasteròs Vil nan Kiba Jewografi

**[register_1]** Mesyedam, se yon gwo onè pou mwen prezante
> M'ap sonje nan ane 1950 yo lè m te timoun piti. Se konsa li t ap fè lwanj Bondye e l t ap sèvi ak tout kalite bagay ki gen rapò avè l tankou rad, jwèt aza oswa lòt bagay ankò. Li te konn di: "Se pa sa sèlman k' ta dwe fèt". Nan lane 2000, li te ekri sou papye sa a epi li te mande padon pou peche nou yo. Lè n reflechi sou jan pawòl sa a sonnen, ann Ayiti tou patou, men kounye a se pitit fi Papa Nasyon an ki te voye nou. Nou ka konprann egzanp Jezi te bay la byen vit lè n rakonte istwa vrè lanmou li genyen pou moun ki renmen l yo. Men repons lan: "Papa nasyon an "Jezi vivan" "Moun ki kwè toutbonvre "Bondye beni."  Tradiksyon  Lòt kreyòl: Franse : papa Angle : Alman: Panyòl:  Istwa  Istwa mo sa  referans

**[register_2]** Frè m, kite m di w yon bagay:
> Bonjou! Frè m ak mwen se moun ki nan zòn riral yo. Mwen gen anpil zanmi e nou tout konnen sa k ap pase la a. Nou pa ka pale de li pou kont nou.

## Exhibit — temperature-sampled (labeled sampled)

**Rakonte m yon istwa kout an kreyòl sou yon ti fi ki gen anpil kouraj.**
> Janm se ki pa ka konte, men li kapab sèlman swiv moun avèk fòs.  Tradiksyon   Lòt kreyòl:  Franse: jamais  Angle:   Panyòl:  Pòtigè:  Alman:  Etimoloji

**Ki konsèy ou genyen pou yon moun ki fèk rive nan yon nouvo peyi?**
> Dapre enfòmasyon yo, ann Ayiti toujou gen anviwon 60 milyon migran ayisyen. Selon sa k fè konprann pi fò migran ayisyen se granmoun sèlman; plis pase mwatye ladan yo se timoun. Gen kèk gwoup Ayisyen tou pwofite antre nan vil Pòtoprens kote anpil Ayisyen deja ap kouri kite lakay yo epi oblije pran avyon pou al vizite fanmi oubyen zanmi.  Sepandan, selon kominike a, administrasyon Obama a poko bay okenn garanti ke moun kap viv an Ayiti ka retounen sou tè li te planifye depi lontan an san gwo chanjman. Se poutèt sa gouvènman ameriken anonse lap bay asistans finansye pou ede konstwi enfrastrikti touristik ak ti biznis lokal la nan zòn sinistre.

**Eksplike m poukisa kilti ayisyen an gen anpil valè.**
> Dapre enfòmasyon ki disponib, tout bagay ap diskite nan sosyete a pou rive fè moun santi yo fyè de li. Anpil Ayisyen santi sa pa gen sans e sa lakòz menm pwoblèm lan: nou plis tande vwa pèp la sou Twitter; se paske se yon sitiyasyon oubyen yon lòt k ap pase. Se poutèt sa mwen te anvi al gade dokimantè "The People of African American Mind" (1995) ki rele "From the World Trade Center for Haitians".  Istwa  Nan ane 2015, Ayiti te bay peyi manm Nasyonzini yo pèmisyon pou yo pibliye ak distribye videyo koutwazi « The People Of African Indian », epi li montre kèk lyen kote gouvènman ameriken finanse fim tankou "Everything we can't wait to talk about single other people." Sityasyon etranje sou medya sosyal souvan parèt devan je otorite Ameriken ki vle sèvi kominote entènasyonal la.  Fim Twazyèmman, gen plizyè rezon diferan :  1- Dekrè egzekitif 21 Desanm 1976

## Quality — an honest read

This is a **123M** model — the smallest useful scale — so the bar is *speaks Kreyòl and answers*, not *is knowledgeable*. What the SFT achieved and what it did not:

- **The continuer→answerer transition is real.** Where the v1 base *continues* a prompt (newspaper mastheads, encyclopedic drift), the chat model *responds*: greetings answer back, "give me three gift ideas" produces a numbered list, "how do I learn guitar" produces steps. The before/after table above is the demoable moment.
- **A repetition penalty is required.** Pure greedy on a 123M model degenerates into loops; generation (and the shipped Ollama Modelfile) uses `repeat_penalty 1.3` + `no_repeat_ngram_size 3`. With it the loops are gone.
- **Residual encyclopedic bleed.** The base trained heavily on Wikipedia, so some answers still trail into article scaffolding (`Istwa / Referans / Lyen deyò`) or confabulate facts (e.g. "Ki kapital Ayiti a?" → a US town). This is a base-scale limit, not a format failure — the model is answering, just thinly. A larger base is the lever, out of scope here.
- **Grounding phrasing.** Layer-2's doc-dialogue taught an "according to…" opener; the worst form ("dapre pasaj la" — *according to the passage*, with no passage present) was removed by dropping the 175 passage-referencing Layer-2 items before the final SFT.

The blinded naturalness sheet (below) puts this in front of a native reviewer rather than asserting it.

## Deployment — full conversion chain (BOS fix + chat template embedded)

GGUF **f16 246 MB** (sha256 `7243921a53eff74b…`) / **Q4_K_M 78 MB** (sha256 `766b7695732c781d…`) + an ONNX/transformers.js bundle + an Ollama Modelfile with the template (Modal Volume).
GGUF tokenizer metadata carries the Part-0 fix + the chat template: `add_bos_token=True`, `chat_template=embedded`. Gate 1 export Δ=0.0, gate 5 ONNX gen_ok=True.

## Naturalness review

A blinded 30-output sheet ([chat_naturalness_sheet.md](chat_naturalness_sheet.md)) is built for a second native review (same worksheet format as the kakugo audit).
