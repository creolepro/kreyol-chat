# Blinded naturalness review — base-model probe

_10 fixed Kreyòl completion prompts. Each model's greedy completion is shown de-identified (labels are shuffled independently per prompt). The key is committed separately (`probe/naturalness_key.json`) so this sheet stays blind._

**Rubric (score each labelled output 1–3):** **1 = unusable** (not Kreyòl / word-salad); **2 = degraded but Kreyòl** (recognizably Kreyòl, errors/odd); **3 = plausible Kreyòl** (a fluent speaker could have written it). Write your score next to each label.

## 1. _greeting_

**Prompt (given to the model):** `Bonjou, zanmi m! Jodi a`

- **A** ⟶ kajn, kajn, kajn, kajn, kajn, kajn, kajn, kajn, kajn, kajn, kajn, kajn, kajn, kajn, kajn, kajn, kajn, kajn, kajn, kajn,  
  _score: ____
- **B** ⟶ mwen pa lapli, mwen ap fè yon blog pou tout moun. Mwen ap fè yon blog pou tout moun. Mwen ap fè yon blog pou tout moun. Mwen ap fè yon blog pou tout moun. Mwen ap fè yon blog pou tout moun. Mwen ap fè yon blog pou  
  _score: ____
- **C** ⟶ , nou pral pale sou yon pwoblèm ki gen rapò ak konvèsasyon ak moun ki gen yon gwo gwo. Nou pral pale sou yon pwoblèm ki gen rapò ak konvèsasyon ak moun ki gen yon gwo gwo. Nou pral pale sou yon pwoblèm ki gen rapò ak konvèsasyon ak moun ki gen yon g  
  _score: ____
- **D** ⟶ te fèt la, jodi a te fèt la, jodi a te fèt la, jodi a te fèt la, jodi a te fèt la, jodi a te fèt la, jodi a te fèt la, jodi a te fèt la, jodi a te fèt la, jodi  
  _score: ____
- **E** ⟶ ri, mwen te fè yon konsèyman, mwen te fè yon konsèyman, mwen te fè yon konsèyman, mwen te fè yon konsèyman, mwen te fè yon konsèyman, mwen te fè yon konsèyman,  
  _score: ____

## 2. _greeting_

**Prompt (given to the model):** `Alo, koman ou ye? Mwen menm, mwen`

- **A** ⟶ menm, mwen menm, mwen menm, mwen menm, mwen menm, mwen menm, mwen menm, mwen menm, mwen menm, mwen menm, mwen menm, mwen menm, mwen menm, mwen menm, mwen menm, mwen menm, mwen  
  _score: ____
- **B** ⟶ te koumen. Mwen te koumen, mwen te koumen. Mwen te koumen, mwen te koumen. Mwen te koumen, mwen te koumen. Mwen te koumen, mwen te koumen. Mwen te koumen, mwen te koumen. Mwen te kou  
  _score: ____
- **C** ⟶ te koumanse ak mwen te koumanse. Mwen te koumanse ak mwen te koumanse. Mwen te koumanse ak mwen te koumanse. Mwen te koumanse ak mwen te koumanse. Mwen te koumanse ak mwen te koumanse. Mwen  
  _score: ____
- **D** ⟶ genyen. Mwen genyen, mwen genyen, mwen genyen. Mwen genyen, mwen genyen, mwen genyen. Mwen genyen, mwen genyen, mwen genyen. Mwen genyen, mwen genyen, mwen genyen. Mwen genyen, mwen genyen, mwen genyen. Mwen  
  _score: ____
- **E** ⟶ te gen yon ti kras nan yon ti kras nan yon ti kras nan yon ti kras nan yon ti kras nan yon ti kras nan yon ti kras nan yon ti kras nan yon ti kras nan yon ti kras nan yon ti kras nan yon ti kras nan yon ti kras nan yon ti kras nan yon ti kras nan yon ti k  
  _score: ____

## 3. _simple question_

**Prompt (given to the model):** `Kisa ou renmen manje? Mwen renmen`

- **A** ⟶ manje, mwen renmen kisa ou renmen manje? Mwen renmen kisa ou renmen manje? Mwen renmen kisa ou renmen manje? Mwen renmen kisa ou renmen manje? Mwen renmen kisa ou renmen manje? Mwen renmen kisa ou renmen manje? Mwen renmen k  
  _score: ____
- **B** ⟶ kisa? Kisa ou renmen manje? Mwen renmen kisa? Kisa ou renmen manje? Mwen renmen kisa? Kisa ou renmen manje? Mwen renmen kisa? Kisa ou renmen manje? Mwen renmen kisa? Kisa ou renmen manje? Mwen renmen kisa? Kisa  
  _score: ____
- **C** ⟶ manje kisa. Kisa ou renmen manje? Mwen renmen manje kisa. Kisa ou renmen manje? Mwen renmen manje kisa. Kisa ou renmen manje? Mwen renmen manje kisa. Kisa ou renmen manje? Mwen renmen manje kisa. Kisa ou renmen manje  
  _score: ____
- **D** ⟶ manje. Kisa ou renmen manje? Mwen renmen manje. Kisa ou renmen manje? Mwen renmen manje. Kisa ou renmen manje? Mwen renmen manje. Kisa ou renmen manje? Mwen renmen manje. Kisa ou renmen manje? Mwen renmen manje. Kisa ou renmen manje?  
  _score: ____
- **E** ⟶ manje. Mwen renmen manje. Mwen renmen manje. Mwen renmen manje. Mwen renmen manje. Mwen renmen manje. Mwen renmen manje. Mwen renmen manje. Mwen renmen manje. Mwen renmen manje. Mwen renmen manje. Mwen renmen manje.  
  _score: ____

## 4. _simple question_

**Prompt (given to the model):** `Poukisa lapli tonbe? Paske`

- **A** ⟶ , kui on kõik kõik, siis on kõik kõik. Kui on kõik kõik, siis on kõik kõik. Kui on kõik kõik, siis on kõik kõik. Kui on kõik kõik, siis on kõik kõik  
  _score: ____
- **B** ⟶ i, kai kai kai kai kai kai kai kai kai kai kai kai kai kai kai kai kai kai kai kai kai kai kai kai kai kai kai kai kai kai kai kai kai kai kai kai kai kai kai  
  _score: ____
- **C** ⟶ li te gen yon gwo kote pou li te ka mete yon gwo kote pou li te ka mete yon gwo kote pou li te ka mete yon gwo kote pou li te ka mete yon gwo kote pou li te ka mete yon gwo kote pou li te ka mete yon gwo kote pou li te ka mete yon gwo kote pou li  
  _score: ____
- **D** ⟶ vicius, V. (2017). Poukisa lapli tonbe?. In V. Paskevicius (Ed.), 2017 m. Europos Sąjungos Parlamentas: Europos Sąjungos Parlamentas ir Lietuvos Respublikos Seimas (pp. 1-2). Lietuvos Res  
  _score: ____
- **E** ⟶ tonbe karekare, tonbe karekare, tonbe karekare, tonbe karekare, tonbe karekare, tonbe karekare, tonbe karekare, tonbe karekare, tonbe karekare, tonbe karekare, tonbe karekare, tonbe k  
  _score: ____

## 5. _short translation_

**Prompt (given to the model):** `An kreyòl, "Good morning, how are you?" vle di:`

- **A** ⟶ "Mwen kreyòl, mwen kreyòl, mwen kreyòl, mwen kreyòl, mwen kreyòl, mwen kreyòl, mwen kreyòl, mwen kreyòl, mwen kreyòl, mwen kreyòl, mwen kreyòl, mwen  
  _score: ____
- **B** ⟶ "Bonjou, ki jan ou ye?"  
  _score: ____
- **C** ⟶ "Bon matin, kòm te?" "Bon matin, kòm te?" vle di: "Good morning, how are you?" "Bon matin, kòm te?" vle di: "Good morning, how are you?" "Bon matin, kòm te?" vle di: "Good morning, how are you?" "Bon matin  
  _score: ____
- **D** ⟶ "Bonswa, ki moun yo?" (bon-swah, kee moun-yo) or "Bonswa, ki moun yo?" (bon-swah, kee moun-yo). In English, "Good morning, how are you?" is "Good morning, how are you?" or "Good morning, how are you?" or "Good morning, how  
  _score: ____
- **E** ⟶ "Bonjou, konn ou pa konn?". An kreyòl, "Good afternoon, how are you?" vle di: "Bonapè, konn ou pa konn?". An kreyòl, "Good evening, how are you?" vle di: "Bonsoir, konn ou pa konn?". An kreyòl  
  _score: ____

## 6. _short translation_

**Prompt (given to the model):** `Fraz sa a: "I love my country very much" vin di an kreyòl:`

- **A** ⟶ "Mwen pral moun ki genyen bon sens" (I love my country very much) (English translation) Fraz sa a: "I love my country very much" vin di an kreyòl: "Mwen pral moun ki genyen bon sens" (I love my country very much) (English translation) Fraz sa a: "I love my  
  _score: ____
- **B** ⟶ "Mwen renmen peyi mwen an" Fraz sa a se yon fraz ki gen yon konplèksite nan kreyòl. Li gen yon fraz ki gen yon fraz ki gen yon fraz ki gen yon fraz ki gen yon fraz ki gen yon fraz ki gen yon fraz ki gen yon fraz ki gen yon fraz ki gen  
  _score: ____
- **C** ⟶ "Mwen genyen kreyòl" (I speak Creole) vin di an kreyòl: "Mwen genyen kreyòl" (I speak Creole) vin di an kreyòl: "Mwen genyen kreyòl" (I speak Creole) vin di an kreyòl: "Mwen genyen kreyò  
  _score: ____
- **D** ⟶ "Mwen teyen mwen kouman mwen teyen mwen kouman mwen teyen mwen kouman mwen teyen mwen kouman mwen teyen mwen kouman mwen teyen mwen kouman mwen teyen mwen kouman mwen teyen mwen kouman mwen teyen mwen k  
  _score: ____
- **E** ⟶ "I love my country very much" vin di an kreyòl.  
  _score: ____

## 7. _register shift (formal)_

**Prompt (given to the model):** `Mesyedam, se yon gwo onè pou mwen prezante`

- **A** ⟶ mwen anpil moun ki mwen aksepte pou mwen aksepte mwen. Mwen aksepte mwen pou mwen aksepte mwen. Mwen aksepte mwen pou mwen aksepte mwen. Mwen aksepte mwen pou mwen aksepte mwen. Mwen aksepte mwen pou mwen  
  _score: ____
- **B** ⟶ nan yon manm ki gen yon pwogram ki fèt nan yon manm ki gen yon pwogram ki fèt nan yon manm ki gen yon pwogram ki fèt nan yon manm ki gen yon pwogram ki fèt nan yon manm ki gen yon pwogram ki fèt nan yon manm  
  _score: ____
- **C** ⟶ nou yon nouvo pwodiksyon ki te fèt nan yon ti kras nan yon ti kras nan yon ti kras nan yon ti kras nan yon ti kras nan yon ti kras nan yon ti kras nan yon ti kras nan yon ti kras nan yon ti kras nan yon ti kras nan yon ti kras nan yon ti kras nan yon ti  
  _score: ____
- **D** ⟶ , mwen te souvènje mwen te te souvènje mwen te te souvènje mwen te te souvènje mwen te te souvènje mwen te te souvènje mwen te te souvènje mwen te te souvènje mwen te te souvèn  
  _score: ____
- **E** ⟶ mwen ak mwen te fè mwen prezante mwen ak mwen te fè mwen prezante mwen ak mwen te fè mwen prezante mwen ak mwen te fè mwen prezante mwen ak mwen te fè mwen prezante mwen ak mwen te fè mwen prezante mwen ak mwen te fè  
  _score: ____

## 8. _register shift (casual)_

**Prompt (given to the model):** `Frè m, kite m di w yon bagay:`

- **A** ⟶ yon bagay ki te fèt pou yon bagay ki te fèt pou yon bagay ki te fèt pou yon bagay ki te fèt pou yon bagay ki te fèt pou yon bagay ki te fèt pou yon bagay ki te fèt pou yon bagay ki te fèt pou  
  _score: ____
- **B** ⟶ a. to be in the middle of something; to be in the middle of something; to be in the middle of something; to be in the middle of something; to be in the middle of something; to be in the middle of something; to be in the middle of something; to be in the middle of something; to be in the middle of something; to be in the middle of  
  _score: ____
- **C** ⟶ Frè m, kite m di w yon bagay: Frè m, kite m di w yon bagay: Frè m, kite m di w yon bagay: Frè m, kite m di w yon bagay: Frè m, kite m di w yon bagay: Frè m, kite m di w yon bagay: Frè  
  _score: ____
- **D** ⟶ 1000000000000000000000000000000000000000000000000000000000000000000000000000000  
  _score: ____
- **E** ⟶ si ou pa gen yon fason pou w konnen si yon moun ki gen yon fason pou w konnen si yon moun ki gen yon fason pou w konnen si yon moun ki gen yon fason pou w konnen si yon moun ki gen yon fason pou w konnen si yon moun ki gen yon fason pou w konnen si yon moun ki gen yon fason pou  
  _score: ____

## 9. _continuation (narrative)_

**Prompt (given to the model):** `Te gen yon fwa, yon ti gason yo te rele Ti Jan. Chak maten, li`

- **A** ⟶ te fèt yon fwa. Li te fèt yon fwa nan yon koumanasyon, yon fwa nan yon koumanasyon, yon fwa nan yon koumanasyon, yon fwa nan yon koumanasyon, yon fwa nan yon koumanasyon, y  
  _score: ____
- **B** ⟶ te rele ti Jan. Yon ti gason yo te rele Ti Jan. Chak maten, li te rele ti Jan. Yon ti gason yo te rele Ti Jan. Chak maten, li te rele ti Jan. Yon ti gason yo te rele Ti Jan. Chak maten, li te rele ti Jan. Yon ti gason yo te rele Ti  
  _score: ____
- **C** ⟶ te ale nan kay laprès yo, li te ale nan kay laprès yo, li te ale nan kay laprès yo, li te ale nan kay laprès yo, li te ale nan kay laprès yo, li te ale nan kay laprès yo, li te ale nan kay laprès yo, li te ale nan kay laprès yo, li  
  _score: ____
- **D** ⟶ te rele Ti Jan. Ti Jan, Ti Jan, Ti Jan, Ti Jan, Ti Jan, Ti Jan, Ti Jan, Ti Jan, Ti Jan, Ti Jan, Ti Jan, Ti Jan, Ti Jan, Ti Jan, Ti Jan, Ti Jan, Ti Jan, Ti Jan, Ti Jan, Ti Jan, Ti Jan, Ti Jan, Ti Jan, Ti Jan, Ti Jan,  
  _score: ____
- **E** ⟶ te rele Ti Jan. Chak maten, li te rele Ti Jan. Chak maten, li te rele Ti Jan. Chak maten, li te rele Ti Jan. Chak maten, li te rele Ti Jan. Chak maten, li te rele Ti Jan. Chak maten, li te rele Ti Jan. Chak maten, li te rele Ti  
  _score: ____

## 10. _continuation (expository)_

**Prompt (given to the model):** `Ayiti se yon peyi ki gen yon istwa rich. Kilti li`

- **A** ⟶ se yon melanj de tradisyon afrikan ak franse. Ayiti se yon peyi ki gen yon istwa rich. Kilti li se yon melanj de tradisyon afrikan ak franse. Ayiti se yon peyi ki gen yon istwa rich. Kilti li se yon melanj de tradisyon afrikan ak franse. Ayiti se yon peyi ki  
  _score: ____
- **B** ⟶ te gen yon koumanse ki te fèt nan 1492, kote yon kòmpani franse te vin nan yon lòt peyi ki te gen yon nom ki te gen yon koumanse ki te fèt nan 1492, kote yon kòmpani franse te  
  _score: ____
- **C** ⟶ te genyen yon peyi ki gen yon istwa rich. Kilti li te genyen yon peyi ki gen yon istwa rich. Kilti li te genyen yon peyi ki gen yon istwa rich. Kilti li te genyen yon peyi ki gen yon istwa rich. Kilti li te genyen y  
  _score: ____
- **D** ⟶ yon peyi ki gen yon peyi ki gen yon peyi ki gen yon peyi ki gen yon peyi ki gen yon peyi ki gen yon peyi ki gen yon peyi ki gen yon peyi ki gen yon peyi ki gen yon peyi ki gen yon peyi ki gen yon peyi ki gen yon  
  _score: ____
- **E** ⟶ gen yon peyi ki gen yon peyi ki gen yon peyi ki gen yon peyi ki gen yon peyi ki gen yon peyi ki gen yon peyi ki gen yon peyi ki gen yon peyi ki gen yon peyi ki gen yon peyi ki gen yon peyi ki gen yon peyi ki gen y  
  _score: ____
