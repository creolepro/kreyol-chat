# Corpus v0 — quality audit sample (machine first pass)

*Deterministic stratified sample of 200 docs (seed 20260720), across sources × length bands. Build: `full`. Snapshot 2026-07-20.*

**These flags are MACHINE-ESTIMATED (fasttext lid.176 + heuristics) and PENDING HUMAN REVIEW.** langid is unreliable on short docs and runs modest on Kreyòl, so `wrong_language` is an upper bound. `translation_shaped` is left blank — it needs a human ear.

> **MADLAD snippets are redacted.** rights.yaml marks MADLAD redistribution *unresolved* (CC-BY vs ODC-BY), so its text is not excerpted in a committed file. Full text for every row (MADLAD included) is in the git-ignored `ml/reports/audit_sample.jsonl` for local human review.

## Machine-estimated rates

`wrong_lang` is the raw langid flag; `wrong_lang≥300` restricts to docs ≥300 chars (rules out the short-doc confound). Both are UPPER BOUNDS: eyeballing shows most flagged docs are genuinely Kreyòl encyclopedic stubs dense with French/Spanish proper nouns (foreign names, film titles) that fool lid.176 — i.e. this mostly measures langid's weakness on Kreyòl, not contamination. Human review needed.

| scope | n | wrong_lang | wrong_lang≥300 (n) | low_conf_ht | boilerplate | unreadable |
|---|--:|--:|--:|--:|--:|--:|
| **overall** | 200 | 25.5% | 20.7% (174) | 71.0% | 0.0% | 0.0% |
| madlad_400_ht_clean | 151 | 12.6% | 12.0% (150) | 84.8% | 0.0% | 0.0% |
| ht_wikipedia | 48 | 64.6% | 75.0% (24) | 29.2% | 0.0% | 0.0% |
| owned_proverbs | 1 | 100.0% | — | 0.0% | 0.0% | 0.0% |

## Sampled documents

| doc_id | source | genre | band | langid (conf) | flags | snippet |
|---|---|---|---|---|---|---|
| `ht_wikipedia:13298` | ht_wikipedia | encyclopedic | l | fr (0.48) | wrong_language | Odette Roy Fombrun, ki fèt 13 jen 1917 nan Pòtoprens (Ayiti) epi ki mouri 23 desanm 2022 nan Petyonvil (Ayiti), se yon ekriven ayisyen.  Biyografi Li te fèt Pòtoprens ann Ayiti. An 1935 li te gen yon  |
| `ht_wikipedia:13824` | ht_wikipedia | encyclopedic | s | ht (0.13) | low_confidence_ht | Bayonèt, Yon kouto long, de bò file, pwenti, moun lontan yo te konnen itilize pou batay nan lagè. Moun sa yo pate gen zam ki genyen konnye a pou batay, se ki fè yo te konnen itilize bayonèt.  Tradiksy |
| `ht_wikipedia:21701` | ht_wikipedia | encyclopedic | xs | nds (0.36) | wrong_language | Dat li yo : ( 2103–1905 ) .  Byografi Dat enpòtan yo  Zèv li yo  Referans |
| `ht_wikipedia:22170` | ht_wikipedia | encyclopedic | s | ht (0.74) | — | se yon komin nan pwovens Cadix, nan kominote otonòm Andalouzi, nan peyi Espay. right\|Komin yo ki nan pwovens Cadix  Li genyen (2002) 187.087 moun. Sipèfisi l se 196.275 Km^2  Istwa Istwa  referans  Kè |
| `ht_wikipedia:22597` | ht_wikipedia | encyclopedic | xs | ht (0.24) | low_confidence_ht | right\|Komin yo ki nan pwovens Jaén  Li genyen (2002) 1 322 moun.  Istwa Istwa  referans  Kèk lyen |
| `ht_wikipedia:29700` | ht_wikipedia | encyclopedic | xs | ht (0.18) | low_confidence_ht | Chèf-lye li se ?  Nan lane 2004, popilasyon l te konte 990 moun. Lòt lane yo te genyen : 2003 (995), Istwa Istwa  referans  Kèk lyen |
| `ht_wikipedia:3465` | ht_wikipedia | encyclopedic | s | jv (0.39) | wrong_language | 29 me se 149m jou lane (oubyen 150m nan lane bisektil) nan almanak gregoryen.  Evènman yo  Dat li fèt yo 1917 : John Fitzgerald Kennedy, prezidan Etazini (+ 1963)  Dat li mouri yo  Selebrasyon yo Fèt  |
| `ht_wikipedia:38427` | ht_wikipedia | encyclopedic | s | ht (0.52) | — | Dover se yon vil nan eta Ohio , nan Etazini. Popilasyon l nan lane 2000 te genyen 12,210 moun. Li nan konte, rejyon Tuscarawas . Istwa Istwa  Relasyon ak Ayiti Jewografi Ekonomi Devlòpman Politik Edik |
| `ht_wikipedia:38952` | ht_wikipedia | encyclopedic | xs | ht (0.35) | low_confidence_ht | Lucas se yon vil nan eta Ohio , nan Etazini. Li nan konte, rejyon Richland . Istwa Istwa  Relasyon ak Ayiti Jewografi  Ekonomi  Devlòpman  Politik  Edikasyon  Anviwònman  referans  Kèk lyen |
| `ht_wikipedia:44398` | ht_wikipedia | encyclopedic | s | ht (0.35) | low_confidence_ht | Irasburg se yon vil nan eta Vermont, nan Etazini. Li nan konte, rejyon Orleans Nan lane 2000, popilasyon l te konte 1077 moun. Sifas li se 105.1 km² Istwa Istwa  Relasyon ak Ayiti Jewografi  Ekonomi   |
| `ht_wikipedia:44788` | ht_wikipedia | encyclopedic | s | ht (0.31) | low_confidence_ht | Auburn se yon vil nan eta Washington, nan Etazini. Nan lane 2000, popilasyon l te konte 40,314 moun. Istwa Istwa  Relasyon ak Ayiti Jewografi Ekonomi Devlòpman Politik Edikasyon Anviwònman referans Kè |
| `ht_wikipedia:5776` | ht_wikipedia | encyclopedic | s | ht (0.25) | low_confidence_ht | 570 se yon nonm. Pou ekri li, yo itilize chif arab yo. Matematik Nan sistèm desimal Se nonm antye natirèl ant 569 ak 571 nan sistèm desimal.  Nan lòt sistèm yo Se nonm antye natirèl ant 567 ak 571 nan |
| `ht_wikipedia:5790` | ht_wikipedia | encyclopedic | s | ht (0.21) | low_confidence_ht | 556 se yon nonm. Pou ekri li, yo itilize chif arab yo. Matematik Nan sistèm desimal Se nonm antye natirèl ant 555 ak 557 nan sistèm desimal.  Nan lòt sistèm yo Se nonm antye natirèl ant 555 ak 557 nan |
| `ht_wikipedia:58271` | ht_wikipedia | encyclopedic | xs | rm (0.09) | wrong_language | Okawix se yon sistèm enfòmatik ki ap fè yon òdinatè (PC, Mac) mache byen, li ap itilize tout resous materyèl sistèm an pou li eksplwate li.  wikipedya é wikisource san entenet.  okawix |
| `ht_wikipedia:59069` | ht_wikipedia | encyclopedic | xs | ht (0.32) | low_confidence_ht | thumb\|Òj (Hordeum).  vignette\|Champ d'Orge carrée (Hordeum vulgare) nan mwa jen nan nò Frans Òj se yon sereyal. |
| `ht_wikipedia:65098` | ht_wikipedia | encyclopedic | s | diq (0.12) | wrong_language | El crimen del Padre Amaro se yon pyès teyat an sòti 2005.  Aktè ak ekip Julián Gil kòm Padre Amaro Marilyn Pupo kòm Dionisia Sara Pastor kòm Sanjuanera Giovanni Haddock kòm Juan Eduardo Marcos Garay k |
| `ht_wikipedia:66301` | ht_wikipedia | encyclopedic | m | es (0.48) | wrong_language | Vanessa Saba Zarzar, rele Vanessa Saba (ki fèt 23 jen 1975 nan Lima) se yon aktris pewouvyen. Biyografi Zèv li yo Fim 2005 : Un día sin sexo : Daniela 2007 : Una sombra al frente : Doris Beltrán 2009  |
| `ht_wikipedia:67481` | ht_wikipedia | encyclopedic | s | ht (0.52) | — | Ivan Rakitić, ki fèt sou 10 mas 1988 nan Rheinfelden (Swis), se yon foutbalè entènasyonal kroyat ki jwe kòm yon milye santral oswa atakan nan FC Barcelone ak nan ekip nasyonal Kroasi la. Biyografi Zèv |
| `ht_wikipedia:6780` | ht_wikipedia | encyclopedic | xs | jv (0.19) | wrong_language | 424 se 469m ane nan almanak jilyen.  Gade osi nonm 424.  divètisman  la  matematik  relijyon  sosyete  syans imèn  syans natirèl  teknoloji |
| `ht_wikipedia:67841` | ht_wikipedia | encyclopedic | m | es (0.54) | wrong_language | María de Jesús Rubio Tejero, rele María Rubio, ki fèt 21 septanm 1934 nan Tijuana, Baja California; epi ki mouri 1e mas 2018 nan Meksiko, te yon aktris meksikèn sinema, teyat ak televizyon. Biyografi  |
| `ht_wikipedia:67976` | ht_wikipedia | encyclopedic | m | fr (0.18) | wrong_language | Anna Mae Bullock, rele Tina Turner, ki fèt 26 novanm 1939 epi ki mouri 24 me 2023, se yon chantèz, dansèz, aktris ak konpozitris orijinè amerikèn ak natiralize swis.  Biyografi Tina Turner te marye av |
| `ht_wikipedia:68451` | ht_wikipedia | encyclopedic | m | ht (0.23) | low_confidence_ht | Sentaks, se yon branch nan yon lang nan ki etidye fason mo yo mete ansanm pou fòme yon fraz.  Egzanp :  Mwen ak li twa mo sa yo pa relye sa vle di gen fot sentaks mwen avè l oswa mwen avè li mo sa yo  |
| `ht_wikipedia:68513` | ht_wikipedia | encyclopedic | m | fr (0.53) | wrong_language | Hamid Nacer-Khodja, ki fèt 25 janvye 1953 nan Lakhdaria (Aljeri) epi ki mouri 16 septanm 2016 nan Djelfa, se yon ekriven ak powèt aljeryen. Biyografi Zèv li yo Pwezi 2015 ː La Profonde terre du verbe  |
| `ht_wikipedia:71336` | ht_wikipedia | encyclopedic | m | fr (0.90) | wrong_language | Flore Bonaventura, ki fèt 12 oktòb 1988, se yon aktris fransèz. Biyografi Zèv li yo Sinema 2012 : Comme des frères d'Hugo Gélin : Cassandre 2012 : White City Spleen, kout fim d'Alfred Rambaud : sè 201 |
| `ht_wikipedia:7284` | ht_wikipedia | encyclopedic | xs | jv (0.18) | wrong_language | 928 se 973m ane nan almanak jilyen.  Gade osi nonm 928.  divètisman  la  matematik  relijyon  sosyete  syans imèn  syans natirèl  teknoloji |
| `ht_wikipedia:7319` | ht_wikipedia | encyclopedic | xs | jv (0.18) | wrong_language | 963 se 1008m ane nan almanak jilyen.  Gade osi nonm 963.  divètisman  la  matematik  relijyon  sosyete  syans imèn  syans natirèl  teknoloji |
| `ht_wikipedia:73561` | ht_wikipedia | encyclopedic | s | fr (0.23) | wrong_language | Jorge Aguilera, ki fèt nan Meksiko (Meksik), se yon reyalizatè, senaris, pwodiktè ak montè sinema meksiken. Biyografi Zèv li yo Tankou reyalizatè 1991 : Pero se sigue viviendo 1993 : Juguete, arte obj |
| `ht_wikipedia:7385` | ht_wikipedia | encyclopedic | xs | jv (0.18) | wrong_language | 1029 se 1074m ane nan almanak jilyen.  Gade osi nonm 1029.  divètisman  la  matematik  relijyon  sosyete  syans imèn  syans natirèl  teknoloji |
| `ht_wikipedia:74180` | ht_wikipedia | encyclopedic | s | ht (0.18) | low_confidence_ht | Miami En Action se yon fim ayisyen soti an 2004. Rezime  Ekip teknik  Aktè Fritz Buissereth kòm Joe Hubermann Saintil kòm Manno Sheila Mocombe kòm Rachel  Referans  Lyen deyò Miami En Action sou movie |
| `ht_wikipedia:74579` | ht_wikipedia | encyclopedic | l | fr (0.92) | wrong_language | Émile Cohen-Zardi, rele Dominique Zardi, ki fèt 2 mas 1930 nan PariNotice sur Les Gens du cinéma, epi ki mouri 13 desanm 2009Mort de l'acteur Dominique Zardi dépêche AFP, se yon aktè, jounalis, ekrive |
| `ht_wikipedia:74716` | ht_wikipedia | encyclopedic | m | es (0.61) | wrong_language | Georgina García Tamargo, rele Gina Romand, ki fèt 15 fevriye 1938 nan La Avàn (Kiba), se yon aktris meksikèn. Biyografi Zèv li yo Fim 1986 : Chiquita pero picosa : Victoria Blanco 1985 : Gavilán o pal |
| `ht_wikipedia:75649` | ht_wikipedia | encyclopedic | s | pms (0.21) | wrong_language | Master's Sun (an koreyen : 주군의 태양, Jugun-ui Taeyang) se yon seri televizyon sidkoreyèn difize an 2013 sou SBS.  Aktè Gong Hyo-jin kòm Tae Gong-shil So Ji-sub kòm Joo Joong-won Seo In-guk kòm Kang Woo  |
| `ht_wikipedia:75887` | ht_wikipedia | encyclopedic | s | ht (0.14) | low_confidence_ht | Hospital Playlist (an koreyen : 슬기로운 의사생활, Seulgiroun Euisasaenghal) se yon seri televizyon sidkoreyèn difize depi 2020 sou tvN.  Aktè Jo Jung-suk kòm Lee Ik-joon Yoo Yeon-seok kòm Ahn Jung-won Jung K |
| `ht_wikipedia:76445` | ht_wikipedia | encyclopedic | s | fr (0.19) | wrong_language | Grabuge ! se yon fim franse reyalize pa Jean-Pierre Mocky ak soti an 2005.  Rezimafasaaaaaaaaaaaaaaaaaaaaaaaaaaaikkjjjgghhhfvvhyyiopplnbvffftttyyyyy ehjgopnblf  Ekip teknik  Aktè Michel Serrault : kom |
| `ht_wikipedia:76489` | ht_wikipedia | encyclopedic | s | fr (0.54) | wrong_language | Le Cri du hibou se yon fim franse reyalize pa Claude Chabrol ak soti an 1987.  Rezime  Ekip teknik  Aktè Christophe Malavoy : Robert Mathilda May : Juliette Jacques Penot : Patrick Jean-Pierre Kalfon  |
| `ht_wikipedia:77149` | ht_wikipedia | encyclopedic | m | fr (0.75) | wrong_language | Jalil Lespert, ki fèt 11 me 1976 nan Pari (Frans), se yon aktè ak reyalizatè franse. Biyografi  Zèv li yo Tankou aktè Fim 1998 : Nos vies heureuses de Jacques Maillot : Étienne 1999 : Un dérangement c |
| `ht_wikipedia:79468` | ht_wikipedia | encyclopedic | s | ht (0.36) | low_confidence_ht | GMM 25 se yon chanèl dijital televizyon terrestres nan Tayilann ki posede pa GMM Grammy. Rezo a ofri yon varyete de kontni tankou dram, mizik, nouvèl ak pwogram amizman vize adolesan.  GMM 25 te lanse |
| `ht_wikipedia:79830` | ht_wikipedia | encyclopedic | m | oc (0.28) | wrong_language | Le Bal des actrices se yon fò dokimantè franse ekri ak reyalize pa Maïwenn ak soti 28 janvye 2009.  Rezime  Ekip teknik  Aktè Aktris Jeanne Balibar kòm li menm Romane Bohringer kòm li menm Julie Depar |
| `ht_wikipedia:81220` | ht_wikipedia | encyclopedic | s | fr (0.11) | wrong_language | The Green Card se yon fim ayisyen reyalize pa Jean Gardy Bien-Aimé ak soti an 2007.  Rezime  Ekip teknik Reyalizatè : Jean Gardy Bien-Aimé Direktè fotografi : Telio Deetjen Dat soti : 2007  Aktè Claud |
| `ht_wikipedia:81434` | ht_wikipedia | encyclopedic | s | diq (0.18) | wrong_language | Les Gagnants se yon fim franse reyalize pa AZ ak Laurent Junca e soti an 2022.  Rezime  Ekip teknik  Aktè JoeyStarr kòm Tom Leroy Alban Ivanov kòm Nicolas AZ kòm Nabil Adèle Galloy kòm Tania Samuel Ba |
| `ht_wikipedia:82397` | ht_wikipedia | encyclopedic | xs | fr (0.22) | wrong_language | Le Cochon Danseur se yon fim franse soti an 1907.THE DANCING PIGThe Dancing Pig  Referans  Lyen deyò Le Cochon Danseur sou IMDb |
| `ht_wikipedia:82453` | ht_wikipedia | encyclopedic | xs | fr (0.30) | wrong_language | La Fée Libellule ou le Lac enchanté se yon fim franse reyalize pa Georges Méliès ak soti an 1908.  Rezime  Ekip teknik  Aktè  Referans  Lyen deyò |
| `ht_wikipedia:82726` | ht_wikipedia | encyclopedic | xs | vi (0.11) | wrong_language | Delightful Dolly se yon fim ameriken soti an 1910.DELIGHTFUL DOLLY  Aktè Marie Eline kòm Marie Eline  Referans |
| `ht_wikipedia:83317` | ht_wikipedia | encyclopedic | s | en (0.33) | wrong_language | Matrimony's Speed Limit se yon fim ameriken reyalize pa Alice Guy-BlachéMatrimony’s Speed LimitSarah Janssen. The World Almanac and Book of Facts 2013 ak soti an 1913.  Referans  Lyen deyò Matrimony's |
| `ht_wikipedia:84682` | ht_wikipedia | encyclopedic | s | vi (0.14) | wrong_language | Langdon's Legacy se yon fim ameriken reyalize pa Otis TurnerLangdon’s LegacyLANGDON'S LEGACY ak soti an 1916.  Aktè J. Warren Kerrigan kòm Jack Langdon Bertram Grassby kòm Juan Maria Barada Lois Wilso |
| `ht_wikipedia:88074` | ht_wikipedia | encyclopedic | m | ht (0.42) | low_confidence_ht | Kont-amiral se yon ran militè nan yon kantite marin. Malgre ke yo pote menm non, klas sa yo pa ekivalan nan yerachi a: kont admiral oswa kont admiral nan Royal Canadian Navy se, dapre standardizasyon  |
| `ht_wikipedia:89177` | ht_wikipedia | encyclopedic | s | tl (0.16) | wrong_language | Yon òdinatè pèsonèl se yon tip òdinatè ki konsevwa pou itilizasyon endividyèl, swa nan yon enstitisyonèl oswa domestik. Jeneralman li sèvi pou aktivite jou wè tretman tèks, navigasyon sou entènet, Imè |
| `ht_wikipedia:90683` | ht_wikipedia | encyclopedic | s | jv (0.13) | wrong_language | Vyolans seksyèl se tout ak seksyèl oubyen tout tentativ ki arive fèt pou gen rapò seksyèl ak yon moun pandan ke youn nan moun yo pa pataje konsantman li pou sa. Òganizasyon Mondyal Sante a ( OMS) ajou |
| `madlad_400_ht_clean:0000490` | madlad_400_ht_clean | web | m | ht (0.17) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0001420` | madlad_400_ht_clean | web | m | ht (0.23) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0001637` | madlad_400_ht_clean | web | m | ht (0.10) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0001651` | madlad_400_ht_clean | web | m | ht (0.10) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0002260` | madlad_400_ht_clean | web | l | ht (0.24) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0002350` | madlad_400_ht_clean | web | m | ht (0.17) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0003331` | madlad_400_ht_clean | web | m | ht (0.11) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0005260` | madlad_400_ht_clean | web | m | ht (0.34) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0005573` | madlad_400_ht_clean | web | m | ht (0.42) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0006511` | madlad_400_ht_clean | web | m | ht (0.21) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0009715` | madlad_400_ht_clean | web | l | ht (0.24) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0009933` | madlad_400_ht_clean | web | l | ht (0.15) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0009952` | madlad_400_ht_clean | web | m | ht (0.44) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0010945` | madlad_400_ht_clean | web | m | ht (0.32) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0011099` | madlad_400_ht_clean | web | m | ht (0.19) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0011404` | madlad_400_ht_clean | web | m | ht (0.12) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0011736` | madlad_400_ht_clean | web | m | ht (0.42) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0011775` | madlad_400_ht_clean | web | s | ht (0.38) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0012064` | madlad_400_ht_clean | web | m | ht (0.23) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0012210` | madlad_400_ht_clean | web | s | ht (0.10) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0013212` | madlad_400_ht_clean | web | m | ht (0.16) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0013477` | madlad_400_ht_clean | web | m | pms (0.10) | wrong_language | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0013578` | madlad_400_ht_clean | web | m | ht (0.18) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0014888` | madlad_400_ht_clean | web | m | ht (0.27) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0016083` | madlad_400_ht_clean | web | m | ht (0.18) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0017201` | madlad_400_ht_clean | web | l | ht (0.31) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0017303` | madlad_400_ht_clean | web | m | jv (0.15) | wrong_language | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0017335` | madlad_400_ht_clean | web | m | ht (0.17) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0017665` | madlad_400_ht_clean | web | m | ht (0.21) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0019559` | madlad_400_ht_clean | web | m | ht (0.32) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0020203` | madlad_400_ht_clean | web | m | ht (0.26) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0020210` | madlad_400_ht_clean | web | m | ht (0.27) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0023890` | madlad_400_ht_clean | web | m | ht (0.41) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0024766` | madlad_400_ht_clean | web | m | ht (0.20) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0025363` | madlad_400_ht_clean | web | l | ht (0.50) | — | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0025487` | madlad_400_ht_clean | web | l | ht (0.30) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0025517` | madlad_400_ht_clean | web | m | ht (0.19) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0026847` | madlad_400_ht_clean | web | l | wa (0.07) | wrong_language | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0027479` | madlad_400_ht_clean | web | l | ht (0.17) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0028743` | madlad_400_ht_clean | web | l | ht (0.22) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0028911` | madlad_400_ht_clean | web | m | ht (0.14) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0029085` | madlad_400_ht_clean | web | l | oc (0.08) | wrong_language | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0030473` | madlad_400_ht_clean | web | m | ht (0.20) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0032217` | madlad_400_ht_clean | web | m | ht (0.29) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0033201` | madlad_400_ht_clean | web | l | ht (0.35) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0034494` | madlad_400_ht_clean | web | m | ht (0.24) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0035843` | madlad_400_ht_clean | web | l | ht (0.32) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0035976` | madlad_400_ht_clean | web | m | ht (0.20) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0036192` | madlad_400_ht_clean | web | l | ht (0.19) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0036366` | madlad_400_ht_clean | web | m | ht (0.23) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0037421` | madlad_400_ht_clean | web | m | ht (0.13) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0038876` | madlad_400_ht_clean | web | l | ht (0.17) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0041397` | madlad_400_ht_clean | web | l | ht (0.29) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0044057` | madlad_400_ht_clean | web | m | ht (0.19) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0044349` | madlad_400_ht_clean | web | xs | en (0.13) | wrong_language | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0045621` | madlad_400_ht_clean | web | m | hr (0.16) | wrong_language | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0045855` | madlad_400_ht_clean | web | m | ht (0.28) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0048885` | madlad_400_ht_clean | web | m | ht (0.23) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0049124` | madlad_400_ht_clean | web | l | ht (0.21) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0050842` | madlad_400_ht_clean | web | m | ht (0.08) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0050946` | madlad_400_ht_clean | web | l | ht (0.36) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0052240` | madlad_400_ht_clean | web | s | ht (0.52) | — | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0054752` | madlad_400_ht_clean | web | m | ht (0.23) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0054934` | madlad_400_ht_clean | web | m | ht (0.18) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0055375` | madlad_400_ht_clean | web | m | tl (0.23) | wrong_language | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0056022` | madlad_400_ht_clean | web | l | ht (0.17) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0056198` | madlad_400_ht_clean | web | m | ht (0.19) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0056774` | madlad_400_ht_clean | web | l | ht (0.15) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0058993` | madlad_400_ht_clean | web | m | ht (0.50) | — | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0059805` | madlad_400_ht_clean | web | l | jv (0.09) | wrong_language | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0061412` | madlad_400_ht_clean | web | m | ht (0.15) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0062312` | madlad_400_ht_clean | web | l | ht (0.23) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0062373` | madlad_400_ht_clean | web | m | ht (0.10) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0062463` | madlad_400_ht_clean | web | m | ht (0.14) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0062566` | madlad_400_ht_clean | web | m | ht (0.22) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0063433` | madlad_400_ht_clean | web | s | ht (0.32) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0063441` | madlad_400_ht_clean | web | l | ht (0.16) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0064277` | madlad_400_ht_clean | web | m | ht (0.44) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0064922` | madlad_400_ht_clean | web | l | ht (0.38) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0064942` | madlad_400_ht_clean | web | m | ht (0.26) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0064964` | madlad_400_ht_clean | web | m | ht (0.29) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0065039` | madlad_400_ht_clean | web | s | fr (0.10) | wrong_language | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0066331` | madlad_400_ht_clean | web | m | ht (0.21) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0066446` | madlad_400_ht_clean | web | m | ht (0.37) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0066650` | madlad_400_ht_clean | web | l | ht (0.30) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0067492` | madlad_400_ht_clean | web | m | ht (0.14) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0068706` | madlad_400_ht_clean | web | l | ht (0.20) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0069067` | madlad_400_ht_clean | web | l | ht (0.26) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0069089` | madlad_400_ht_clean | web | l | ht (0.13) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0069427` | madlad_400_ht_clean | web | m | ht (0.12) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0069440` | madlad_400_ht_clean | web | l | jv (0.08) | wrong_language | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0069741` | madlad_400_ht_clean | web | m | tl (0.16) | wrong_language | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0069974` | madlad_400_ht_clean | web | s | ht (0.14) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0069990` | madlad_400_ht_clean | web | m | ht (0.35) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0070342` | madlad_400_ht_clean | web | m | ht (0.23) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0071457` | madlad_400_ht_clean | web | m | ht (0.29) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0071523` | madlad_400_ht_clean | web | m | ht (0.22) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0072405` | madlad_400_ht_clean | web | l | ht (0.16) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0072677` | madlad_400_ht_clean | web | m | ht (0.21) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0073315` | madlad_400_ht_clean | web | m | ht (0.44) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0073480` | madlad_400_ht_clean | web | l | ht (0.14) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0074277` | madlad_400_ht_clean | web | m | ht (0.25) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0075494` | madlad_400_ht_clean | web | m | ht (0.35) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0075561` | madlad_400_ht_clean | web | l | ht (0.27) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0076780` | madlad_400_ht_clean | web | m | ht (0.33) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0076987` | madlad_400_ht_clean | web | m | ht (0.53) | — | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0077602` | madlad_400_ht_clean | web | m | ht (0.33) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0078057` | madlad_400_ht_clean | web | l | ht (0.42) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0078206` | madlad_400_ht_clean | web | l | ht (0.12) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0078503` | madlad_400_ht_clean | web | l | ht (0.18) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0078951` | madlad_400_ht_clean | web | m | ht (0.16) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0079808` | madlad_400_ht_clean | web | l | ht (0.26) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0079953` | madlad_400_ht_clean | web | l | ht (0.24) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0080110` | madlad_400_ht_clean | web | m | ht (0.39) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0082009` | madlad_400_ht_clean | web | m | ht (0.11) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0082374` | madlad_400_ht_clean | web | l | ht (0.25) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0082473` | madlad_400_ht_clean | web | l | ht (0.49) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0085277` | madlad_400_ht_clean | web | m | tl (0.17) | wrong_language | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0085326` | madlad_400_ht_clean | web | m | ht (0.12) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0086512` | madlad_400_ht_clean | web | l | ht (0.21) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0086524` | madlad_400_ht_clean | web | m | ht (0.15) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0086711` | madlad_400_ht_clean | web | m | ht (0.25) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0087314` | madlad_400_ht_clean | web | m | ht (0.34) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0087326` | madlad_400_ht_clean | web | m | ms (0.11) | wrong_language | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0088224` | madlad_400_ht_clean | web | m | tl (0.08) | wrong_language | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0091098` | madlad_400_ht_clean | web | m | ht (0.22) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0091607` | madlad_400_ht_clean | web | m | ht (0.39) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0091845` | madlad_400_ht_clean | web | s | wa (0.22) | wrong_language | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0092367` | madlad_400_ht_clean | web | m | ht (0.16) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0092458` | madlad_400_ht_clean | web | l | ht (0.28) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0092800` | madlad_400_ht_clean | web | l | ht (0.22) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0094956` | madlad_400_ht_clean | web | m | ht (0.24) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0095274` | madlad_400_ht_clean | web | l | ht (0.17) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0096709` | madlad_400_ht_clean | web | m | ht (0.27) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0097032` | madlad_400_ht_clean | web | m | ht (0.22) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0097861` | madlad_400_ht_clean | web | s | ht (0.44) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0099509` | madlad_400_ht_clean | web | m | ht (0.45) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0099925` | madlad_400_ht_clean | web | m | tl (0.15) | wrong_language | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0100054` | madlad_400_ht_clean | web | m | ht (0.14) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0100322` | madlad_400_ht_clean | web | l | ht (0.27) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0101349` | madlad_400_ht_clean | web | m | jv (0.11) | wrong_language | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0101375` | madlad_400_ht_clean | web | l | ht (0.24) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0105399` | madlad_400_ht_clean | web | l | en (0.14) | wrong_language | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0105458` | madlad_400_ht_clean | web | m | ht (0.19) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0105615` | madlad_400_ht_clean | web | s | ht (0.28) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0106775` | madlad_400_ht_clean | web | m | ht (0.27) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0106895` | madlad_400_ht_clean | web | l | ht (0.21) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0106980` | madlad_400_ht_clean | web | m | es (0.07) | wrong_language | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0107306` | madlad_400_ht_clean | web | m | ht (0.26) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0109954` | madlad_400_ht_clean | web | l | ht (0.34) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `madlad_400_ht_clean:0110401` | madlad_400_ht_clean | web | m | ht (0.29) | low_confidence_ht | «redacted — MADLAD redistribution unresolved (see git-ignored jsonl)» |
| `owned_proverbs:25` | owned_proverbs | proverb | xs | vi (0.84) | wrong_language | Pinga ou mete men ou nan vant moun. |
