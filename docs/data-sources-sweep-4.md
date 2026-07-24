# Haitian Creole data-source research — sweep 4

*Researched 2026-07-24. This sweep deliberately excludes the sources already resolved
by the first three surveys. Every positive license claim below is supported by a
quoted source or terms page; ambiguous, noncommercial, and no-derivatives material
is not treated as trainable. Status vocabulary matches `ml/corpus/rights.yaml`:
**TRAIN-OK**, **PERMISSION-ROUTE**, **QUARANTINE**, and **EVAL-ONLY**.*

## Summary table

| Source | Register | Authored vs. translated | Grounded volume estimate | Rights verdict | Acquisition path |
|---|---|---|---|---|---|
| **Woy Magazine** | Essays, interviews, memory, culture, politics, diaspora, personal narrative | Mixed; many Haitian contributors, with some articles marked as originally written in Kreyòl and others translated | **224 Kreyòl article URLs**. English mirror contains ~259k words; estimated **200k–300k Kreyòl words**, medium confidence | **PERMISSION-ROUTE** | Request an export plus original-language metadata from `Info@woymagazine.com` |
| **EspasKreyòl** | Literature, history, social science, culture, religion, journalism, poetry | Predominantly authored Kreyòl, including archived Kreyòl newspapers; some republished material | **247 posts / 197,680 raw words** through WordPress API; approximately **150k–200k usable words**, high confidence | **PERMISSION-ROUTE / ITEM VERIFY** | WordPress REST API; request per-item license/provenance manifest from `espaskreyol@gmail.com` |
| **Reading Room Haiti** | Children's books, stories, early literacy | Primarily translated/adapted | Site claims **700 Kreyòl books**; likely **0.3M–1.5M words**, low confidence until catalog and duplicates are inspected | **PERMISSION-ROUTE / ITEM VERIFY** | Offline-capable PWA; request catalog/API export and license manifest from `readingroomhaiti@gmail.com` |
| **Corpus of Northern Haitian Creole** | Natural interviews, rural and urban speech, personal narrative, Capois dialect | Authored/spoken | **10 interviews, 20 speakers, about 9–10 hours**; approximately **65k–90k transcribed words**, medium confidence | **PERMISSION-ROUTE** | Contact Indiana University CeLT/Creole Institute, investigators, and rights/consent holders |
| **OtakuTòk podcast** | Informal youth conversation, humor, anime/pop culture, code-switching | Authored/spoken | **36 episodes**; estimated **18–36 hours / 130k–320k words**, low–medium confidence | **PERMISSION-ROUTE** | Creator agreement followed by transcription; contact `otakutokpodcast@gmail.com` |
| **Learn Haitian Creole / Aprann Kreyòl Ayisyen** | Pedagogical dialogue, daily life, health, cultural explanation | Creator-authored, often bilingual/pedagogical | **76 episodes**, 2019–2025; likely **100k–250k spoken words**, low confidence | **PERMISSION-ROUTE** | Ask Mimi/Jeanne Fortune for transcript/audio corpus license |
| **Cric? Crac! — 1901 original** | Literary verse, humor, rural storytelling, fables | Adaptation/transcreation rather than modern translationese | **246 pages / 31 fables**; estimated **25k–50k Kreyòl words**, medium-low confidence | **TRAIN-OK: public domain in US**, original edition only | OCR the 1901 scan; do not use the copyrighted modern-orthography edition |
| **Lessons in Haitian Creole — 1921** | Dialogues, phrases, proverbs, grammar, historical material | Pedagogical/constructed | **93 printed pages**; approximately **10k–20k Kreyòl words**, low confidence | **TRAIN-OK: public domain in US** | Internet Archive full-text/OCR/PDF downloads |
| **Anthologie d'un siècle de poésie haïtienne — 1925** | Poetry and literary history | Authored literary Kreyòl mixed with French | **408 pages / ~72k total extracted words**, but probably **under 10k Kreyòl words**, medium confidence | **TRAIN-OK: CC0** | Download Manioc PDF; manually isolate Kreyòl passages |
| **CMU Haitian Corpus** | News, medical text, dictionary entries, read speech | Mostly translated/read; low incremental authored value | About **443k Haitian tokens**: ~336k news, ~10k medical, ~97k dictionary | **TRAIN-OK** | Direct CMU download; perform strict deduplication against existing corpus |
| **Atlas Linguistique d'Haïti** | Nationwide dialect fieldwork, elicited speech, phrases | Authored speech, but elicited rather than conversation | **499 recordings / 356.3 hours / 200+ speakers**; theoretical **2.6M–3.2M words** if fully transcribed, low–medium confidence | **PERMISSION-ROUTE / CONSENT VERIFY** | COCOON/BNF plus collaboration with FLA–UEH and project investigators |
| **CreoleCentric dictionary** | Lexicon, usage examples, definitions, pronunciation | Provenance unclear; platform also produces AI translation | **28,000+ entries**; possibly **250k–500k word-like units**, low confidence | **PERMISSION-ROUTE** | Request a licensed dictionary export and provenance breakdown |
| **USAID/RTI ToTAL — M ap li nèt ale** | Scripted classroom dialogue, readers, teacher guides, literacy instruction | Authored/adapted pedagogical Kreyòl | At least a **400+ page Grade 1 workbook**, reader, and scripted guides; likely **100k–300k words**, low confidence | **PERMISSION-ROUTE** | Ask RTI, USAID, and MENFP for an explicit model-training license and complete source files |
| **Zilora Haitian Creole Natural Speech Corpus** | Spontaneous speech, dialects, code-switching | Claimed natural/native speech | **30 rows / 2.57 MB**; negligible | **QUARANTINE pending provenance and consent** | Ask publisher for releases, recording provenance, and speaker consent |

Volume calculations for speech use a transparent 120–150 words/minute range. They
indicate potential transcription yield, not existing text volume.

## Per-source evidence and notes

### 1. Woy Magazine

This is the strongest newly identified editorial corpus. Woy describes Haitians and
diaspora Haitians carrying out discourse about Haiti "from our own perspective," and
says the publication is "entirely bilingual (English and Kreyòl)." Its current
register includes essays, conversations, cultural criticism, solidarity writing, and
memory work—not NGO prose.

- Source: [Woy — About us](https://www.woymagazine.com/about-us/about-us)
- Catalog: [Kreyòl sitemap](https://www.woymagazine.com/sitemap_ht.xml)

The Kreyòl sitemap exposes 224 article URLs. A count of the English mirror produced
258,550 words, making 200k–300k Kreyòl words a reasonable range.

No text-reuse license or corpus grant appears on the site. The default rule therefore
applies: the Copyright Office says, "Once you create an original work and fix it …
you are the author and the owner."

- Rights reference: [U.S. Copyright Office — What is Copyright?](https://copyright.gov/what-is-copyright/)

**Action:** ask for a one-time corpus export under CC BY or CC BY-SA, retaining
author, article URL, publication date, and original-language fields. Train only on
articles whose authors or Woy can license them.

### 2. EspasKreyòl

EspasKreyòl is a real missed corpus, not a directory. Its public WordPress API
returned 247 posts containing 197,680 raw words. Categories include literature,
history, language, music, poetry, science, and archives of *Sèl*, *Libète*,
*Bon Nouvèl*, and *Popouri*. Sample pages contain long-form Kreyòl cultural writing
rather than web boilerplate.

The site says it seeks resources "under a free license or in the public domain," but
does not identify a license on individual articles. Its footer separately states,
"© Se Espas Kreyòl ki eyandwa sit la."

- Source and rights statements: [EspasKreyòl](https://espaskreyol.org/)
- Acquisition endpoint: `https://espaskreyol.org/wp-json/wp/v2/posts`

That is insufficient for blanket ingestion, especially for republished newspaper
archives. "Free license" might mean CC BY, but it might also include NC/ND variants
or an unsupported rights assumption.

**Action:** request a machine-readable rights manifest containing `post_id`, source
publication, author, original date, copyright holder, license, and license URL. If
EspasKreyòl can explicitly license its original posts, that subset could move
immediately to TRAIN-OK.

### 3. Reading Room Haiti

The site currently claims 700 Kreyòl children's books, not roughly 1,000. It says its
app uses books available "for anyone to use for free under creative commons
licenses."

- Source and rights statement: [Reading Room Haiti](https://readingroomhaiti.org/)

That statement does not identify the license of each book. CC BY-NC and CC BY-ND
would be unacceptable, and part of the catalog probably overlaps Bloom Library and
Storybooks Haiti.

**Action:** request the catalog database or API response with title, source
collection, language, download URL, and exact license. Deduplicate by normalized text
and source ID before estimating value. Do not ingest the entire app based on the
blanket statement.

### 4. Corpus of Northern Haitian Creole

The corpus contains approximately ten hours of recorded and transcribed interviews
with 20 people from northern Haiti. It includes natural Capois speech and
sociolinguistic variation rather than read prompts. A later survey identifies it
explicitly as "Not open source."

- Rights/status evidence: [Creole NLP resource registry](https://creole-nlp.github.io/)
- Corpus description: [The Gulf of Guinea Creole Corpora](https://www.researchgate.net/publication/263177586_The_Gulf_of_Guinea_Creole_Corpora)

At 120–150 words/minute, its transcript should contain roughly 65k–90k words after
removing interviewer speech and metadata.

**Action:** contact Indiana CeLT, the original investigators, and any current rights
holder. Permission must cover transcripts, model training, redistribution of
processed text, and participant consent. Because the speakers include rural
participants recorded for fieldwork, copyright permission alone is not enough.

### 5. OtakuTòk

The program describes itself as a Haitian Kreyòl podcast where anime fans in Haiti
"pale e diskite." It is unusually valuable because it supplies contemporary youth
vocabulary, humor, turn-taking, and pop-culture discussion.

- Source: [OtakuTòk listing](https://rephonic.com/podcasts/otakutok)

No open license was located. Public availability is not permission to transcribe and
redistribute.

**Action:** propose a creator partnership: transcript production at project expense,
creator review, explicit contributor license, episode-level provenance, and an
opt-out/removal process.

### 6. Learn Haitian Creole / Aprann Kreyòl Ayisyen

Apple lists 76 episodes and labels the copyright "© Mimi." Episodes include
daily-life vocabulary, mental-health discussions, and a car-repair-shop dialogue.

- Source and rights statement: [Apple Podcasts](https://podcasts.apple.com/us/podcast/learn-haitian-creole-aprann-krey%C3%B2l-ayisyen/id1492768191)

The program is less spontaneous than OtakuTòk, but its clean pedagogical dialogue
could be useful for continued pretraining or instruction shaping.

**Action:** ask the creator to license existing scripts and transcripts directly.
Because the creator appears to control the show, this is likely a simpler permission
route than an institutional archive.

### 7. Cric? Crac!

The original 1901 edition contains 246 pages of La Fontaine fables recast by Georges
Sylvain as verse narrated by a Haitian mountain speaker.

- Catalog and page count: [Open Library](https://openlibrary.org/books/OL17946863M/Cric_crac%21)
- Scan: [Creighton Digital Repository](https://cdr.creighton.edu/items/76eb5d40-0ed0-4408-80f3-e569f5c9f6ea)

The Copyright Office states that works published before January 1, 1931 have entered
the US public domain.

- Rights evidence: [U.S. Copyright Office — What is Copyright?](https://copyright.gov/what-is-copyright/)

**Important distinction:** use only the 1901 scan. The 1999 edition's conversion
into modern Kreyòl orthography and its annotations are later creative work and
should not be copied.

This is small but high-value literary/humorous Kreyòl. Preserve the original
orthography as a provenance field; optionally create a separately reviewed
normalization without replacing the source.

### 8. Lessons in Haitian Creole

The Internet Archive record identifies the 1921 publication, 93 printed pages, and
reports "no visible notice of copyright; stated date is 1921," with status
`NOT_IN_COPYRIGHT`.

- Source, rights evidence, and downloads:
  [Internet Archive](https://archive.org/details/lessonsinhaitian00haitrich)

The text includes vocabulary, dialogues, readings, and proverbs. It uses historical
spelling and reflects the US occupation period, so it should receive a
`historical_pedagogy` genre tag rather than being mixed invisibly into modern prose.

### 9. Anthologie d'un siècle de poésie haïtienne

The 1925 anthology is predominantly French but contains several Kreyòl poems and
linguistic passages. Its Manioc record gives the license as "Licence Creative
Commons CC0."

- Source and rights statement: [Manioc](https://www.manioc.org/en/patrimon/PAP11095)

A PDF extraction produced roughly 72k total words, but Kreyòl likely accounts for
fewer than 10k. This should be manually segmented; automatic language identification
will probably fail on the historical orthography.

### 10. CMU Haitian Corpus

CMU's license grants permission "to use and distribute this data and its
documentation without restriction," including modification, publication, and
sublicensing.

- Source and license: [CMU Haitian Corpus](https://www.speech.cs.cmu.edu/haitian/)
- Text downloads: [CMU Haitian text](https://www.speech.cs.cmu.edu/haitian/text/)

The text side contains roughly 443k Haitian tokens across translated newswire,
medical text, and dictionary entries. This is legally clean but not a solution to
the authored-register problem. Use it only after exact and MinHash deduplication
against existing crawl and translation sources.

### 11. Atlas Linguistique d'Haïti

The 2025 paper reports digitized field recordings from the late 1970s and 1980s,
covering more than 200 speakers. The archive includes 499 recordings and 356.3
hours, but much of it is elicited words and phrases rather than free conversation.

- Research paper:
  [Speech Technologies with Fieldwork Recordings: the Case of Haitian Creole](https://aclanthology.org/2025.computel-main.5/)
- Archive:
  [COCOON record](https://cocoon.huma-num.fr/exist/crdo/meta/cocoon-8ea988d2-bf16-303d-81a0-0c55cc0)

No explicit reusable license for the underlying recordings was verified. The derived
LLL-CREAM model is marked "CC BY-NC-SA 4.0," but that does not license the source
audio or future transcripts.

- Derived-model terms:
  [LLL-CREAM model card](https://huggingface.co/LLL-CREAM/wav2vec2-HAT-1.4K-large)

**Action:** treat this as a transcription partnership with FLA–UEH, investigators,
and archive custodians. Establish participant-consent and culturally
sensitive-material rules before processing.

### 12. CreoleCentric

CreoleCentric advertises "28,000+ Haitian Creole words with audio pronunciations,
usage examples, and English translations."

- Source: [CreoleCentric](https://creolecentric.com/)

Its terms say, "All content, features, functionality, technology, and intellectual
property … are owned by CreoleCentric Labs."

- Rights statement:
  [CreoleCentric Terms of Service](https://creolecentric.com/legal/terms/)

The dictionary may contain a substantial clean lexicon, but provenance is unknown
and the platform also uses AI translation. It should not be treated automatically
as human-authored.

**Action:** request a licensed static export plus fields identifying source,
human-authored versus generated examples, reviewer, and dialect.

### 13. ToTAL / M ap li nèt ale

The USAID final report confirms that the project created student books and teacher
guides which were validated by MENFP and approved by USAID. Funding does not make
contractor-created material public domain.

- Project report:
  [USAID ToTAL final report](https://pdf.usaid.gov/pdf_docs/PA00K911.pdf)

The SharEd terms state that the site and content are "copyright of Research Triangle
Institute. All Rights Reserved," and prohibit redistribution except with permission.

- Rights statement: [RTI SharEd Terms of Use](https://shared.rti.org/terms-use)

**Action:** ask RTI/USAID/MENFP for an explicit open-model license covering the
original Kreyòl source files. The strongest portion would be scripted
teacher–student dialogue and readers, not administrative reports.

### 14. Zilora natural-speech dataset

The dataset card claims "Natural conversations, not read speech" and "Manual,
native-verified" transcripts. It contains only 30 rows and is tagged MIT.

- Dataset card:
  [Zilora Haitian Creole Natural Speech Corpus](https://huggingface.co/datasets/ZiloraSystems/zilora-haitian-creole-speech)

The card gives no recording dates, speaker releases, recruitment procedure, or
consent language. An MIT tag supplied by the uploader does not establish the
speakers' authorization.

**Verdict:** quarantine until the publisher provides releases and provenance.

## Top five by authored value per effort

1. **Woy Magazine** — Best combination of contemporary authored Kreyòl, register
   breadth, structured acquisition, and a small identifiable editorial organization.
   Preserve original-language metadata so translated articles do not contaminate the
   authored slice.
2. **EspasKreyòl** — Nearly 200k raw words already exposed through a clean API,
   including historical journalism and cultural writing. The entire decision turns
   on obtaining a real per-item license manifest.
3. **Cric? Crac! (1901)** — Immediately usable, literary, humorous, and unlike
   anything in the current corpus. Small, but the OCR/normalization effort is bounded
   and scientifically useful.
4. **Corpus of Northern Haitian Creole** — Only about 10 hours, but actual
   conversation and personal narrative are far more valuable than another million
   translated web tokens. Existing transcripts make this materially easier than
   transcribing a new archive.
5. **OtakuTòk** — Potentially 100k–300k words of modern informal dialogue, youth
   vocabulary, and humor from one reachable creator. A creator-first agreement could
   make it both legally clean and community governed.

**Next tier:** Reading Room Haiti after a license-manifest audit; Learn Haitian
Creole for pedagogical dialogue; Atlas Linguistique for a longer institutional
collaboration.

## Dead ends and low-value checks

| Lead checked | Result |
|---|---|
| **Defense Language Institute Haitian Creole Basic Course** | Not automatically federal PD. The 203-page Volume III says, "Further reproduction outside the ERIC system requires permission of the copyright owner." [ERIC PDF](https://files.eric.ed.gov/fulltext/ED058795.pdf) |
| **Peace Corps "Kreol" manual** | The readily indexed Peace Corps course is Seychelles Creole, not Haitian Creole. No verified Haitian Peace Corps course corpus found. |
| **Radio Haiti Inter** | Extraordinary source—more than 5,300 recordings—but item license is CC BY-NC-SA. A sample also warns that some third-party material is not covered. **QUARANTINE.** [DPLA item](https://dp.la/item/bd033a85719cf653a155c3dc99d0c066) |
| **IARPA Babel Haitian Creole** | 203 hours of conversation and scripted telephone speech, but distributed under LDC for-profit/nonprofit agreements, not an open corpus. [LDC catalog](https://catalog.ldc.upenn.edu/LDC2017S03) |
| **Kreyòl Pale textbook** | Excellent 29-chapter contemporary dialogue textbook, but explicitly **CC BY-NC**. **QUARANTINE**, though worth asking the authors for a separate license. [Open Textbook Library](https://open.umn.edu/opentextbooks/textbooks/kreyol-pale-a-haitian-creole-textbook-for-beginners) |
| **Ti Koze Kreyòl** | Conversation manual licensed **CC BY-NC-ND 3.0**. **QUARANTINE.** [MERLOT](https://www.merlot.org/merlot/viewMaterial.htm?id=661437) |
| **Partners in Literacy Haiti books** | Six easy readers, all translated/adapted. Site says it "does not allow the reselling or reprinting" of materials. **QUARANTINE/PERMISSION.** [PILH](https://www.haiti-literacy.org/kreyol-print/) |
| **dLOC/FIU collections** | Large Haiti holdings but very little item-level Haitian Creole full text; no substantial downloadable Kreyòl newspaper corpus found. Rights are per-item. |
| **Duke Radio Haiti "transcripts"** | Detailed trilingual metadata exists, but not full transcripts. The audio itself remains NC. |
| **TalkBank/CHILDES** | No Haitian Creole conversational corpus located; TalkBank's common NC terms would also be incompatible. |
| **UD Haitian Adolphe** | 71,734 tokens under CC BY-SA, but underlying text is primarily JW material. Because JW is a final hard quarantine, this derivative is not safe. |
| **UD Haitian Autogramm** | Only 3,279 tokens, mixed from Bible, novel, VOA, and NGO sources with uneven underlying rights. Too small and legally messy. |
| **HPLT v2 / CulturaX / Glot500 / MaLA-style collections** | No meaningful non-Common-Crawl Haitian source was identified; mostly redundant crawl or multilingual aggregation. |
| **Reddit, forums, and public Facebook posts** | Authored social Kreyòl exists, but platform visibility does not grant redistribution or model-training rights. Only viable through an explicit opt-in contribution drive. |
| **YouTube manual captions** | Captions remain creator-owned unless a specific video is published under a compatible CC license. No sizable, clearly licensed manual-caption collection found. |
| **Lyrics websites** | Contemporary konpa and rap lyrics are copyrighted and usually reproduced without a reusable license. Hard quarantine unless artists contribute directly. |
| **World Bank/UN-system Kreyòl PDFs** | Mostly translated institutional material. Several use noncommercial IGO terms; no high-value authored corpus with a compatible blanket license was found. |
| **Le Moniteur / MENFP websites** | No large, downloadable authored-Kreyòl text collection with clear reuse rights found. School materials remain permission candidates. |
| **Lingua Libre Haitian** | 558 CC-licensed pronunciation recordings—legally useful, but only isolated words and therefore negligible for LM text training. [Wikimedia Commons](https://commons.wikimedia.org/wiki/Category:Lingua_Libre_pronunciation-hat) |
| **1793 Kreyòl proclamation and Six Creole Folk-Songs** | Public-domain historical microtexts, collectively only a few thousand tokens. Preserve as heritage material, but they do not move corpus scale. |
| **HF second-pass miscellany** | `predika-org` is CC BY-NC-ND; `EdManZoeTech` has four rows and contradictory licensing; `zaydzuhri/kreyol-mt-cleaned` has no source/license documentation; xMIND is NC and translated. None is ingestible. |

## Partnership-shaped opportunities

| Partner | Partnership proposal | Why it is unusually valuable |
|---|---|---|
| **Woy Magazine** | License articles on an author-by-author opt-in basis; include original-language markers and author attribution | Contemporary essays, personal narrative, and diaspora writing |
| **EspasKreyòl** | Joint rights audit, per-item manifest, and explicit CC BY/BY-SA release for materials they control | Historical journalism, literature, culture, and intellectual Kreyòl |
| **Kreyòl & Beyond** | Run an opt-in corpus drive across its author/educator network | It describes itself as a "nonprofit literary collective" and explicitly seeks to "partner with organizations." [About](https://www.kreyolandbeyond.org/about) |
| **OtakuTòk** | Fund transcription and return reviewed transcripts/subtitles to the creator in exchange for a reusable license | Modern informal youth dialogue and humor |
| **Mimi / Learn Haitian Creole** | License scripts, transcripts, and selected dialogues; compensate editorial review | Clean everyday and instructional dialogue |
| **Indiana / CNHC investigators** | Digitally repatriate and responsibly license transcripts with participant safeguards | Rare natural Capois conversation and personal narrative |
| **Radio Haiti estate + Duke** | Seek a separate model-training and transcript license from the estate; exclude third-party audio | Potentially transformative archive of interviews, news, and ordinary voices |
| **FLA–UEH / Atlas Linguistique** | Community-governed transcription project with consent review and dialect metadata | Nationwide dialect coverage at a scale unavailable elsewhere |
| **Reading Room Haiti** | Help build and publish a per-title rights manifest; ingest only compatible items | Could expose hundreds of legitimately reusable children's books |
| **RTI / MENFP / USAID** | Ask for the original ToTAL source files under an explicit open-model license | High-quality scripted literacy and classroom language |
| **CreoleCentric** | License the human-authored dictionary subset and document provenance | Large structured lexicon and usage-example collection |
| **Haitian creators generally** | Launch an opt-in "Konbit Korpus" for essays, stories, transcripts, newsletters, sermons, and group-chat-style dialogue | The fastest path to authored social and conversational Kreyòl without scraping people who never consented |

