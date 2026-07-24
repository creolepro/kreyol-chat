# Data — corpus v0.2 + SFT sources (Workstream J & the Workstream I data plan)

*Drafted 2026-07-24 from a verified web survey (every license verdict below is backed by a quoted terms page or dataset card, not guessed). Status vocabulary matches `ml/corpus/rights.yaml`: **TRAIN-OK** / **PERMISSION-ROUTE** / **QUARANTINE** / **EVAL-ONLY**.*

## Why v0.2

Corpus v0.1 is 112M unique tokens: 91.7% web crawl (audited ~70% translation-shaped) + 8.3% Wikipedia (71% bot-stubs). Model C v0 proved the pipeline; its ceiling is now data, three ways:

1. **Authored share ~8%** — the model's voice is translationese web register.
2. **No register diversity** — zero journalism, legal, health, financial, children's literature, or dialogue.
3. **Workstream I (midtraining + SFT)** needs instruction/dialogue data that v0.1 contains none of.

Survey outcome in one line: there is a clean-rights path to **~150–180M unique tokens with the authored share roughly tripled** — one big unlock (VOA Nouvèl) plus a tail of small clean sources — and off-the-shelf Apache-2.0 Kreyòl **instruction** data already exists (kakugo-hat, MURI-IT), which converts Workstream I's biggest risk into an audit task.

---

## §1 Source registry — verified verdicts

### TRAIN-OK (to be added to `rights.yaml` with evidence links)

| Source | Register | Authored? | Est. tokens | Evidence (quoted at survey time) | Acquisition |
|---|---|---|---|---|---|
| **VOA Nouvèl** (voanouvel.com) | journalism, current affairs | **yes — native-Kreyòl journalists** | **~20–60M** (est. 30–80k articles, 2014→; confirm via sitemap) | "All text, audio and video material produced exclusively by the Voice of America is public domain" (US federal work). **Carve-out:** licensed AFP/AP/Reuters material inside articles is NOT PD — must be filtered | Polite crawler (robots.txt, browser headers, rate-limited); naive fetch gets Cloudflare 403 |
| US federal agency PDFs — CDC Stacks, USCIS, IRS, SSA, FEMA, EPA, HHS, OSHA, CFPB | health, immigration, tax, disaster, financial | professional translation | ~1–5M | 17 U.S.C. §105 — federal works are public domain ("The U.S. government work designation does not apply to works of state and local governments") | Per-agency PDF harvest; CDC Stacks supports collection harvesting |
| **IRS Pub 850 (en-ht)** | tax terminology | translation | ~50–100k (~1,500 term pairs) | Federal PD | Single PDF; also feeds SFT translation turns |
| **CFPB HT financial glossary** (81 pp, 2024 — copy in the family-contributed set) | financial terminology + style guide | translation | ~30–60k | Federal PD | Already in hand |
| **Bib La 1985** (eBible.org) | religious (archaic-leaning) | human translation | ~1M | "In the Public Domain due to publication without copyright notice in 1985, and without subsequent registration within 5 years" (ebible.org/hat/copyright.htm) | Direct download (USFM/ePub); strip verse markup; heavy dedup vs crawl (Bible text recurs in many corpora) |
| **Konstitisyon 1987** (Kreyòl) | legal, canonical | 1987 constitutional text in Kreyòl (whether the Kreyòl version is the official enactment or a later translation is disputed — survey sources attribute different translators) | ~20–60k | The Wikisource copy is CC BY-SA regardless of the official-status question | Single document |
| **MedlinePlus HT — federal-producer subset only** (FDA, CDC, CMS, FEMA docs) | patient health | translation | small | MedlinePlus is a **router, not a source**: its own Kreyòl text doesn't exist; ~120 aggregated PDFs follow each producer's rights. "Works produced by the federal government are not copyrighted" | Follow each PDF to the producing agency; prefer stacks.cdc.gov originals |
| **Bloom Library (sil-ai/bloom-lm) hat, CC-BY/CC-BY-SA entries only** | children's storybooks | **substantially authored** (SIL Haiti literacy initiative, local authors) | ~50–100k (281 live books; 260 in the HF dataset pre-filter) | Per-entry license field ("cc-by-4.0, cc-by-nc-4.0, cc-by-nd-4.0, cc-by-sa"); **keep BY/BY-SA, drop NC/ND** | HF `sil-ai/bloom-lm` (gated) or OPDS API (key required); bilingual books need language-tagged extraction |
| **Storybooks Haiti** (global-asp.github.io/storybooks-haiti) | children's stories | human translation | ~10–15k (~44 stories) | Footer: "Some rights reserved" → **CC BY 4.0** | Static GitHub Pages — clone the repo; separate Kreyòl from parallel French |
| fineweb-2 `hat_Latn` (ODC-BY) | web crawl | translation-shaped | ~300MB parquet raw; expect heavy MADLAD overlap — treat as a dedup-merge experiment, not new volume | Card license odc-by | HF; dedup vs MADLAD before counting anything as new |
| Tatoeba hat | everyday sentences | authored | negligible (162 sentences) | CC-BY 2.0 FR, attribution | Seed/eval phrases only |
| rmunro/disaster_response_messages | disaster SMS (sanitized) | authored | small, mixed-language | CC-BY-4.0; PII "stripped… reviewed by at least 3 people" | Marginal value; classification-shaped |

Already in v0.1 and unchanged: MADLAD-400 hat (CC-BY-4.0), ht.wikipedia (CC-BY-SA; 72,039 articles, stub/bot-heavy).

### PERMISSION-ROUTE (a letter is plausible and worth it)

| Source | Why it matters | The ask |
|---|---|---|
| **Healthy Roads Media** | patient-education register | CC BY-NC-ND 3.0 whose own text invites it: "Any of these restrictions can be waived if you ask permission" — cheapest ask on the list |
| **Akademi Kreyòl Ayisyen** | THE normative source: official orthography/grammar (footer says "PA KOPYE © AKA") | Partnership ask with public-mission framing (enfomasyon@akademikreyol.net); strategically valuable beyond tokens |
| **Ayibopost / Juno7 / Rezo Nòdwès** | best contemporary *authored* journalism register after VOA | Mission-driven outlets; data-partnership ask realistic |
| NYC.gov (Local Law 30 corpus) / City of Boston | large civic translation corpora | copyright@doitt.nyc.gov / LCA@boston.gov; all-rights-reserved until granted |
| Lakou Kajou | children's educational media (video-first; "© TSNE. All rights reserved") | Mission-aligned nonprofit; a partnership could unlock scripts/transcripts — no harvestable text today |
| Educa Vision | 2,000+ professionally edited Kreyòl titles | Commercial licensing conversation only |
| Kreyòl-MT monolingual (unreleased) | aggregated monolingual HT | Maintainer invites contact |
| NJ Judiciary legal glossary + MA DESE IEP forms (family-contributed copies) | legal/education terminology | State works ≠ PD (the stricter read is quarantine); quick terms-check/permission email; **interim: EVAL-ONLY, local probes only** |
| **Florida state HT materials** (DOH, WIC) | public health/benefits translation corpus | **CONTESTED — held out of training.** FL public-records law + *Microdecisions v. Skinner* suggest state materials are PD, but myflorida.com terms assert "personal, non-commercial use only." Statute-vs-site-terms conflict → resolve (or ask) before ingesting |

### QUARANTINE (no training, no redistribution)

- **JW.org** — largest single Kreyòl corpus in existence, and explicitly anti-scraping ("…may not create… tools… specifically made to collect, copy… or scrape data"). Avoid.
- **Mission 4636 / WMT11 raw earthquake SMS** — excluded on **ethics**, independent of license: real disaster-victim messages, organizers state "the anonymization may be incorrect or incomplete." Not for a generative model.
- **MedlinePlus third-party producers** (American Cancer Society, Immunization Action Coalition, Mass DPH, Mass General), **NY State agencies** (non-commercial/credit terms), **Potomitan** (per-author rights, clearance impractical), **TaCo/saillab alpaca-haitian** (CC BY-NC), **Freeman Medical Dictionary** (published copyrighted work; human reference only), **Kreyòl-MT bundle** (license "other"; use its per-component index, never the blob), **MIT-Haiti** (standing quarantine — see §5 open item).

### EVAL-ONLY (standing)

FLORES+ (unchanged), CreoleVal religious-MT component ("Copyrighted"), and the family-contributed NJ/MA/BMC documents pending rights checks.

---

## §2 Workstream J — corpus v0.2 acquisition + ingestion

Ordered by value-per-effort. Everything lands under `ml/data/` (never committed); registries (`rights.yaml`, `splits.yaml`) are updated **before** ingestion, per standing policy.

**J1 — VOA Nouvèl crawl (the big one).**
Sitemap-driven, robots.txt-respecting, rate-limited crawler with browser headers (site 403s naive fetchers). Per-article provenance: URL, date, byline. **Wire-content policy:** drop articles bylined to AFP/AP/Reuters entirely; strip wire-credited paragraphs elsewhere; when in doubt, drop — PD applies only to VOA-produced text. MinHash-dedup against MADLAD (which already contains partial VOA). Genre tag `journalism_authored`. Deliverable: doc/token counts + dedup overlap in `ml/reports/corpus_v0_2.md`.

**J2 — Federal PDF harvest.**
CDC Stacks HT collection, USCIS multilingual center (incl. I-589 asylum instructions), IRS (Pub 850 + HT pages), SSA, FEMA/Ready.gov, EPA, HHS/OSHA, CFPB. PDF extraction with multi-column handling; spot-check extraction quality per producer; per-doc provenance (agency, URL). Registers: legal/health/tax/disaster/financial.

**J3 — Family-contributed set (local; never committed).**
Six documents contributed from a family member's professional interpreter/translator reference collection. Per-file plan:

| File | Rights verdict | Action | Allocation |
|---|---|---|---|
| CFPB financial glossary | TRAIN-OK (federal PD) | extract; also mine EN↔HT pairs | pretrain + SFT translation turns |
| NJ Judiciary court-terms glossary | PERMISSION-ROUTE (state) | terms-check email; extract meanwhile | **eval**: legal-terminology probe set |
| MA DESE IEP form (HT) | PERMISSION-ROUTE (state) | terms-check; mostly labels/blanks | eval at most; tiny |
| Labor-&-birth patient handout (bilingual) | **pending** — identify producer from footer | classify once producer known | TBD |
| BMC clinical-trial glossary | PERMISSION-ROUTE ("Courtesy" ≠ license) | ask; **orthography audit** — uses nonstandard French-style accents (santé/rivé); tag, don't silently normalize | eval/quarantine |
| Freeman Medical Dictionary | QUARANTINE (published, copyrighted) | human reference only | none |

Contributor record per plan §10 governance: this contribution is **chain-of-custody of third-party works**, not a rights grant — document the contributor, date, and provenance; the copyrighted items stay quarantined regardless of possession. (First contributor record in the project — the governance template gets exercised for real.)

**J4 — Small clean wins.** Bib La (strip USFM; dedup; **cap religious register ≤~2% of corpus** so 1M tokens don't distort the mix), Konstitisyon 1987, Bloom CC-BY/BY-SA subset, Storybooks Haiti, fineweb-2 hat (dedup vs MADLAD first; ingest only what survives). Florida moved to contested-hold pending the statute-vs-terms resolution.

**J5 — Permission letters (human action, parallel).** Healthy Roads waiver → AKA → Ayibopost → NYC/Boston, in that order of expected yield. Draft letters on request.

**J6 — Registry + build.** rights.yaml/splits.yaml entries with evidence links; junk + langid pass per source (per-source thresholds — government PDFs junk differently than crawl); cross-source MinHash dedup; rebuild train mix with targets **authored ≥30%** (from 8.3%), religious ≤2%; nutrition label v2.

**Eval upgrade rides along:** hold out a temporal VOA slice (e.g., 2026 articles) as `authored_eval_v2` — this simultaneously fixes the standing fertility-report TODO (no authored set large enough for the translated-vs-authored fertility comparison) and gives BPB slices a real authored-journalism axis. Terminology probe sets come from the eval-allocated glossaries (NJ court terms; or a held-out IRS Pub 850 split if rights lag).

Estimated v0.2: **~150–180M unique tokens, authored ~30–40%.** Micro-fleet Q2 (authored upweighting) and Q5 (epoch stretch) directly inform how G-v1 should consume it.

---

## §3 Workstream I — SFT data plan (research-updated)

The literature is consistent (LIMA; MURI, TACL 2025; Aya, ACL 2024; SEA-LION): for a small base model, **bulk data teaches format; a small native-quality set teaches voice**; machine-translated instruction data carries translationese that measurably hurts. Three layers:

**Layer 1 — Midtraining (format, bulk-but-structured).**
- **`Kreyol/kakugo-hat`** — 41,264 synthetic HT multi-turn conversations, Apache-2.0, built specifically for Kreyòl SLMs. **Gate: native-speaker audit of a 150–200 stratified sample first** (the card itself warns "NOT PERFECT!"). Grades well → biggest schedule win of the phase; grades badly → mine it for structure only.
- Filtered/deduped `aya_collection` haitian + `xP3x` hat_Latn (Apache-2.0) — templated/translated bulk; cap volume, drop degenerate templates.
- Translation-task turns from **PD glossaries** (IRS Pub 850, CFPB) + synthetic pairs per the existing phase-1 §I.1 rights-clear route.

**Layer 2 — Corpus-grounded generation (the quality core, 15–30k target).**
MURI-style reverse instructions (write the instruction whose answer IS an authentic corpus passage — output stays native Kreyòl) + SEA-LION-style doc→conversation agent loop (can-this-seed-a-dialogue check → outline → draft → self-critique) over v0.2 **authored** passages (VOA etc.), frontier-model teacher. Stratified native review on a 1–5 rubric; drop <4.

**Layer 3 — SFT cap (small and excellent, per phase-1 §I.2's 1–10k).**
`akoksal/muri-it` hat (9,876 native-output pairs, Apache-2.0) + `aya_dataset` hat (**106** gold human examples — the entire world supply; also eval anchors) + best of Layer 2 + proverb/dictionary QA + glossary translation turns.

Rights notes: kakugo/muri-it/aya are Apache-2.0 (TRAIN-OK); TaCo's 52k translated Alpaca is CC BY-NC (QUARANTINE). Native review remains the long pole (phase-1 risk list) — schedule reviewer time first, not last.

---

## §4 Open items

- **MIT-Haiti license conflict:** CreoleVal's repo lists the MIT-Haiti corpus as CC-BY-4.0; our registry quarantines it from an earlier review. A third-party README doesn't override the source's own terms — verify at the platform itself before any status change. If CC-BY holds, it's the best authored STEM Kreyòl available.
- **Florida statute-vs-terms conflict:** FL public-records law (+ *Microdecisions v. Skinner*) points to PD, but myflorida.com's terms restrict to personal non-commercial use. Held out of training until resolved — a per-agency permission note is the pragmatic path.
- **Wire-filter recall (VOA):** measure on a labeled sample of ~100 articles; err toward dropping.
- **fineweb-2 hat size** unmeasured — measure at ingestion before planning around it.
- **New lead: readingroomhaiti.org** — claims ~1,000 CC-licensed books translated into Kreyòl (an aggregator; per-source license verification needed). Potentially the biggest children's/educational source if it checks out.
- **Confirmed dead ends (do not re-search):** Global Voices (CC-BY but zero Kreyòl articles — no HT Lingua edition), Global Digital Library (no HT titles), StoryWeaver (3 rough titles), Le Nouvelliste (French outlet), ht.wikisource/wiktionary/wikibooks (empty/Incubator), OPUS-100 (no HT pair), Tatoeba beyond seeds (162 sentences), radio-station sites (audio-first), MedlinePlus-as-a-source (it's a router), NLM-authored Kreyòl (doesn't exist).
- **PDF extraction quality** varies by producer (multi-column flyers) — spot-check each.
- **Orthography variance** (French-style accents in some medical materials) — tag by file, never silently normalize; the audit norm is measure-don't-filter.
