# Model-estimated audit (claude-opus-4-8) — pending human verification

Second-pass review of the 200-document stratified sample in
`ml/reports/audit_sample.jsonl`, judged by a fluent Kreyòl/French/Spanish/English
reader. Every document was read (text capped at ~2,200 chars for long docs). Labels
are in `ml/reports/audit_model_labels.csv`. **These are model estimates and still need
human verification** — start with the shortlist at the bottom.

Each doc was judged on three axes: **language** (primary language of the *prose* —
foreign proper nouns and normal code-switching do not count against `ht`),
**quality** (`ok` / `junk` / `stub`), and **translation_shaped** (`natural` = reads like
Kreyòl someone *wrote*; `translated` = reads like Kreyòl a machine/translator
*produced*; `unsure`; `na` for non-ht docs).

## Headline

- **True wrong-language rate: 0.5% (1 / 200)** — vs the machine's **25.5% (51/200)**
  upper bound. The machine's wrong-language flag is almost entirely false positives:
  of its 51 flags I **overturned 50** and **confirmed 1**. The only genuinely
  non-Kreyòl document is Thai gambling spam (`madlad_400_ht_clean:0105399`, which the
  machine had mislabeled `en`, not even `th`). Two further docs are **`mixed`**
  (half-and-half), not cleanly wrong-language.
- **Translation-shaped is the real corpus-quality problem, and it is concentrated in
  the crawl.** 70.0% of the crawl (madlad) documents read as translation-shaped
  (machine-translated / calqued), confirming the plan's hypothesis. Wikipedia is 0%
  translation-shaped but 71% empty stubs.
- **Junk: 17.2% of the crawl** (product listings, SEO/steroid spam, mojibake). Wikipedia
  has **0 junk** but is **71% stubs**.

## Per-source rates (my labels)

| source | n | true wrong-lang | junk | stub | translation-shaped¹ |
|---|---|---|---|---|---|
| `madlad_400_ht_clean` (web crawl) | 151 | **0.7%** (1) | **17.2%** (26) | 0% | **70.0%** (105/150) |
| `ht_wikipedia` (authored) | 48 | 0.0% | 0.0% | **70.8%** (34) | 0.0% (0/48) |
| `owned_proverbs` | 1 | 0.0% | 0.0% | 0.0% | 0.0% |
| **all** | 200 | **0.5%** (1) | 13.0% (26) | 17.0% (34) | 52.5% (105/199) |

¹ translation-shaped % = `translated` / (docs where an ht-prose judgment applies; excludes the 1 non-ht `na` doc). `unsure` (2 docs) not counted as translated.

Overall label counts: language ht=197, mixed=2, other-th=1 · quality ok=140, stub=34,
junk=26 · translation_shaped natural=92, translated=105, unsure=2, na=1 · confidence
high=186, med=11, low=3.

## The crawl translation-shaped finding (key project result)

Among the 150 crawl docs where an ht-prose call applies, **105 (70.0%) read as
translation-shaped** and only **43 (28.7%) read as natural** (2 unsure). This strongly
supports the plan's concern that crawl Kreyòl is heavily machine-produced. The
translation-shaped material is mostly: Chinese/Indian **product & vendor pages**
(steel, chemicals, medical devices, LED, etc.), **SEO/affiliate spam** (crypto,
steroids/"Mauritius" geo-spam, flash-game and dating portals), and **MT news / blogs**
(martech, tourism, tech) rendered from English/French/Russian/Italian, often with
tell-tale artifacts (untranslated English words, `&#39;` entities, `XNUMX`/`XNMX`
placeholders, calques like "Komisyon Konsèy" for "Board").

**Important counter-finding — the crawl is *not* uniformly translationese.** The 43
natural crawl docs are real, high-value Kreyòl and fall in two buckets:
- **Native Haitian-authored text** — journalism (Juno7, AyiboPost, VOA-style, Triboland
  on PHTK/Jovenel Moïse, Boric/Chile, Jamaica-2010, hurricanes, sports roundups),
  diaspora blogs/essays (BelNegès, windowsonhaiti forum posts, a Kreyòl-language forum
  debate), a Port-au-Prince church's own theology, the MIT-Haiti Initiative, a Kreyòl
  dictionary. Several of these were the docs the machine most badly mislabeled
  (e.g. the Moïse-assassination opinion blog `…:0069440` tagged `jv`; the forum debate
  `…:0029085` tagged `oc`; a Biden/Afghanistan report `…:0085277` tagged `tl`).
- **Professional translations that read native-quality** — the Haitian Bible and JW
  content (Bibles, Bible-story retellings, articles), and US government/community
  material translated for the diaspora (IRS ITIN guide, Polk County school COVID guide,
  a Pro-Se legal handbook). I labeled these `natural`; a stricter reviewer might call
  them `translated`. Flagged in the shortlist notes.

## Comparison vs the machine first pass

The machine flagged **51 docs `wrong_language=True`** (all its non-`ht` langid rows).

- **Overturned: 50** (I judge them Kreyòl or mixed). The langid classifier is
  systematically fooled by (i) short Wikipedia stubs, (ii) Kreyòl bios/stubs of foreign
  people/films where the body is foreign **proper nouns**, and (iii) crawl Kreyòl Bibles
  and native prose. Bogus codes assigned: `fr`×13, `jv`×10, `tl`×6, `es`×4, `vi`×3,
  `wa`/`diq`/`pms`/`oc`×2 each, `en`×2, `hr`/`nds`/`rm`/`ms`×1.
- **Confirmed: 1** — `madlad_400_ht_clean:0105399` (Thai gambling spam). Genuinely
  non-Kreyòl. Note the machine's *code* (`en`) was still wrong; the true language is
  Thai. I labeled it `other-th`.
- **Machine false negatives (said ht, actually non-ht): 0.** I found no wrong-language
  document that the machine had passed as `ht`. (I did downgrade 2 machine-`ht` docs to
  `mixed`.)

Net effect: the wrong-language problem in this corpus is **~50× smaller** than the
machine's flag suggests. The machine flag is useful only as a "needs-review" signal,
not as a drop list — dropping all 51 would discard 50 good Kreyòl docs, including
published Bibles and native Haitian journalism.

---

## Verification shortlist (human to-do)

### Priority — genuinely uncertain calls (read these first, ~8)

These are the borderline judgments where my label could be wrong and a fluent human
should decide.

| doc_id | source | machine langid | my label | why it needs review |
|---|---|---|---|---|
| `ht_wikipedia:74579` | wiki | fr (0.92) | **ht** / ok / natural / **low** | Kreyòl intro+headers but ~18k chars of **French** filmography incl. untranslated French role descriptions. Arguably `fr`/`mixed`. |
| `ht_wikipedia:71336` | wiki | fr (0.90) | **ht** / stub / natural / **low** | Same pattern (Flore Bonaventura): Kreyòl frame, all-French filmography body. Borderline `mixed`/`fr`. |
| `ht_wikipedia:77149` | wiki | fr (0.75) | **ht** / ok / natural / **low** | Same pattern (Jalil Lespert). Borderline `mixed`/`fr`. |
| `ht_wikipedia:68513` | wiki | fr | **ht** / ok / natural / med | Algerian-poet bio; body dominated by French bibliography titles. Same family as the three above. |
| `madlad_400_ht_clean:0066650` | crawl | ht | **mixed** / ok / translated / med | Poker-tells article that is genuinely ~half untranslated English, half Kreyòl-MT. |
| `madlad_400_ht_clean:0001651` | crawl | ht | **mixed** / junk / **unsure** | Scraped page: English discography listing + Jehovah's-Witness Kreyòl snippets mashed together. |
| `madlad_400_ht_clean:0099509` | crawl | ht | ht / ok / **unsure** | Haiti health-insurance news; native content but an MT artifact ("akòz gen ase dokte"). natural vs translated unclear. |
| `madlad_400_ht_clean:0105399` | crawl | en | **other-th** / junk / na | Confirm my one wrong-language call: this is Thai gambling spam (the machine's only correct wrong-lang flag, though it guessed `en`). |

Also worth a glance (med-confidence, not langid disagreements): `ht_wikipedia:88074`
(rank article, some calqued terms), `madlad_400_ht_clean:0050842` (garbled MT turbo-book
page), `madlad_400_ht_clean:0079953` (Boston/John Barros campaign page — professional
translation I called natural). And the `natural`-labeled professional translations noted
above (IRS/Polk-County/Pro-Se/Bible/JW) if you want to reclassify them as `translated`.

### (a) All machine-langid disagreements (53 rows)

The 8 priority rows above are part of this set. The remaining ~45 are **high-confidence
reclassifications** where the machine's non-`ht` code is clearly wrong; grouped below for
a fast spot-check (each `→` is my label). These drive the wrong-language rate from 25.5%
to 0.5%.

**Kreyòl bios of foreign figures — foreign titles are proper nouns → all ht/ok**
(the langid model read the Spanish/French titles as the doc language):
`ht_wikipedia:67841` (es), `:73561` (fr), `:13298` (fr), `:67976` (fr), `:66301` (es),
`:74716` (es). *(the 3 low-conf + 1 med French-filmography bios above are the harder
members of this family.)*

**Film/TV stubs — foreign title/cast are proper nouns → all ht/stub:**
`ht_wikipedia:83317` (en), `:82397` (fr), `:76445` (fr), `:82453` (fr), `:81220` (fr, a
*Haitian* film), `:84682` (vi), `:75649` (pms), `:82726` (vi), `:81434` (diq), `:76489`
(fr), `:79830` (oc).

**Number / year / calendar stubs → all ht/stub** (all tagged `jv`):
`ht_wikipedia:7319`, `:3465`, `:7385`, `:6780`, `:7284`.

**Published Bible passages in the crawl → all ht/ok/natural:**
`madlad_400_ht_clean:0026847` (wa, 1 Samuel), `:0065039` (fr, Habakkuk), `:0091845`
(wa, Genesis), `:0013477` (pms, Ruth), `:0099925` (tl, JW 1 Thessalonians).

**Crawl Kreyòl (native or MT) where langid misfired:**
`madlad_400_ht_clean:0069440` (jv → **native** Moïse-assassination blog),
`:0055375` (tl → **native** MIT-Haiti/Limyè Lavi NGO),
`:0106980` (es → **native** diaspora forum post),
`:0085277` (tl → **native** Biden/Afghanistan news),
`:0088224` (tl → **native** Haitian radio/political report),
`:0029085` (oc → **native** Kreyòl-language forum debate),
`:0101349` (jv → MT PVC-flooring), `:0059805` (jv → MT plastics), `:0017303`
(jv → MT martech), `:0087326` (ms → MT kitchen cabinet),
`:0045621` (hr → junk, garbled MT slit-lamp), `:0069741` (tl → junk, steel catalog),
`:0044349` (en → junk, scraped Wikipedia footer only), plus wiki
`:89177` (tl → PC article), `:65098` (diq → play stub), `:90683` (jv → sexual-violence
def), `:58271` (rm → software stub), `:21701` (nds → near-empty stub),
`owned_proverbs:25` (vi → the Haitian proverb).

**Mixed (machine said ht):** `madlad_400_ht_clean:0001651`, `:0066650` (both in priority).

### (b) Low confidence or translation_shaped=unsure

Low confidence (3): `ht_wikipedia:74579`, `:71336`, `:77149` (the French-filmography
bios). Unsure (2): `madlad_400_ht_clean:0001651`, `:0099509`. All are already in the
priority list above.

---

## Human verification (2026-07-22)

The fluent human reviewer (CreolePro) reviewed the verification shortlist above — the
langid-disagreement and low-confidence/unsure rows — and **concurred with the model
labels**. Concurrence is recorded per-row in `audit_human_labels.csv` (blank rows =
model-label-only, filled rows = human-reviewed). Status of the headline rates is
therefore: **model-labeled (claude-opus-4-8), human-verified on the contested set** —
true wrong-language ≈ 0.5% (1/200, Thai spam), crawl junk ≈ 17%, crawl
translation-shaped ≈ 70% (52.5% overall), Wikipedia stubs ≈ 71%. These supersede the
machine first-pass upper bounds in `corpus_v0.md` §Quality audit; that section's
lid.176 rates remain documented as the tooling-gap finding. These 200 labels are the
ground-truth test set for the Kreyòl language-ID side-quest (docs/phase-0.md).
