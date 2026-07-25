# Heritage exhibit — the oldest text connected to the model

> **« Le Loup, la Chèvre et le Chevreau »** — the wolf-at-the-door fable, from
> **Georges Sylvain, _Cric? Crac!_ (1901)** — the first collection of poems
> published by a Haitian writer, La Fontaine's fables retold in Kreyòl verse by
> "a Haitian montagnard." This single fable is held out of the training corpus
> **entirely** so the exhibit claim stays honest: the model never read it.

- **Source:** Digital Library of the Caribbean, [UF00076576](https://dloc.com/UF00076576/00001) — the 1901 first edition (Ateliers haïtiens; OCLC 06991790; held by the University of Florida). US public domain (published before 1929).
- **Provenance:** dLOC's own page-level OCR, pageorders 38–41. Text reproduced **exactly as digitized** — no orthographic normalization.
- **Register / genre:** `historical_literary` / `historical`, orthography `pre_reform`.

## The fable (1901 Kreyòl, exactly as OCR'd)

```
Loup, ti Cabritt avec manman Cabritt
Nan toutt temps, cabritt ac loup,
C6 tancou laitt ac citron :
Cabritt, ce bett qui pas capon :
Gnou sanm'di, bien h-bon-nh&*,
Gnou manman-cabritt, qui td
Rhlde pititt-li :
<< Chita* 1, jouq' moin vini.
< M'a batt ou, si 9a rivd...
< Moin derhb*: quimbd* cb ou >
Rould gnou roch' pa deye;
Et pi, alley fait routt li.
Li pas t'encb divir
Gnou gros loup, qui td cache
Dey6 gnou ragd-piquant*,
Pou coutd 9a l'ap6 dit,
C6h! c6ouh Pessonn pas reponn.
Quand li oue Ca, con cabritt
Li prend gnou ti voix bdgu6,
Pou li dit : < Rh6y'* louvri vitt!
< Che... ti... Sb moin !... Main... gnou loup,
<< Qu' ap6 t'... toui' fanmil' ou !...
< Satan... pdtdfiel toutt loup >
Ti cabritt-la, nan khe-li
Et pi, pou loup-la li dit
< Moin louvri pott pou moun' qui
<< Pas gangnin patt blanc con nous.
<< Cd pou tett 9a*, moin big6
<< (a pas nui pbssonn, pas vrd ? >
```

## A note on the orthography

This is **pre-reform Haitian Creole**, written in a French-based spelling decades before the standard IPN/1979 orthography. It is preserved verbatim: it is a primary historical artifact, and any modernized rendition would be a separately-reviewed derivative, never a replacement (docs/data.md §4). The text also carries **1901-print OCR artifacts** — e.g. `9a` for _ça/sa_, `6` for _é_, `td`/`t6` for _te_, `gnou` for _yon_, `cé` for _se_, stray `<`/`>` for the French guillemets « ». In modern orthography the opening reads roughly: _"Nan tout tan, kabrit ak lou, se tankou lèt ak sitwon…"_ ("In all times, goat and wolf are like milk and lemon…"). The mother goat, leaving for market, warns her kid to open the door to no one — until the wolf, hiding behind a thorn-bush, apes her voice at the door. The kid, unfooled, refuses. It is small, funny, and unmistakably Haitian — the reason it makes a good first stone in the model's lineage.

*Held out of training in `corpus/build_v0_2_1.py` (heritage-fable leak-check asserted at build). Registered in `splits.yaml` → `heritage_exhibits`.*