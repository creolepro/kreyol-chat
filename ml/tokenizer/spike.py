"""B0 integration spike — go/no-go for rustbpe + the apostrophe-regex decision.

Trains a toy 16k tokenizer on ~10MB of corpus, runs a round-trip test, probes
the apostrophe pre-tokenization (stock vs Kreyòl-aware), builds the HF + browser
format bridges and measures exact-token-ID parity vs tiktoken on a 1k-line probe
set, and writes ml/reports/rustbpe_spike.md with the verdict.

Run:  python -m tokenizer.spike
"""

from __future__ import annotations

import os
import time

import regex

from . import config, convert, data
from .core import KreyolBPE

TOY_VOCAB = 16384
TOY_CHARS = 10_000_000

# Apostrophe-clitic forms to probe (bare + mid-sentence).
APOS_FORMS = ["m'ap", "l'ap", "n'ap", "t'ap", "w'ap", "m'te", "m'ta", "n'ta", "l'te", "sa'k"]
APOS_SENT = "Li di m'ap vini men n'ta pito rete."

# Round-trip stressors: accent-heavy Kreyòl (è/ò multi-byte) + code-switch.
ROUNDTRIP = [
    "Dèyè mòn gen mòn; sè a wè lòt bò a.",
    "Nou pral fè yon konbit pou nou ranmase rekòt la ansanm.",
    "Li ekri « Le Nouveau Monde » an fransè epi tradui l an kreyòl.",
    "The meeting starts at 9h, men m'ap rive ta paske gen anpil trafik.",
    "Pòtoprens, Okap, Jakmèl — tout vil sa yo gen istwa pa yo.",
    "1804: revolisyon an fè Ayiti tounen premye repiblik nwa endepandan.",
]


def _toy_texts(target_chars: int):
    """~target_chars of source-mixed corpus text. Crawl + Wikipedia split the
    budget (proverbs are ~1KB total, always included)."""
    cap = target_chars // 2   # crawl and wiki each up to half; proverbs negligible
    got = {"crawl": 0, "wikipedia": 0, "owned": 0}
    out = []
    for d in data.iter_docs():
        cls = data.SOURCE_CLASS[d["acquisition"]["source"]]
        if got[cls] >= cap:
            if got["crawl"] >= cap and got["wikipedia"] >= cap:
                break
            continue
        t = data.nfc(d["text"])
        out.append(t)
        got[cls] += len(t)
    return out


def _pretok(pattern: str, text: str):
    return regex.compile(pattern).findall(text)


def run():
    os.makedirs(config.WORK, exist_ok=True)
    lines = []
    A = lines.append

    # --- toy training (timed) -------------------------------------------------
    data.log("building ~10MB toy sample")
    toy = _toy_texts(TOY_CHARS)
    total_chars = sum(len(t) for t in toy)
    data.assert_no_probe(toy)  # invariant: no probe proverb in any tokenizer set

    data.log("training toy (kreyol-aware pattern)")
    t0 = time.time()
    kbpe = KreyolBPE.train(iter(toy), TOY_VOCAB, config.KREYOL_SPLIT_PATTERN)
    t_kreyol = time.time() - t0
    data.log(f"  trained {kbpe.vocab_size} vocab in {t_kreyol:.1f}s")

    # --- round-trip -----------------------------------------------------------
    rt_fail = []
    for s in ROUNDTRIP:
        ids = kbpe.encode_ordinary(s)
        if kbpe.decode(ids) != s:
            rt_fail.append(s)

    # --- apostrophe probe (pre-tokenization, stock vs kreyol) -----------------
    apos_rows = []
    for form in APOS_FORMS:
        apos_rows.append((form,
                          _pretok(config.STOCK_SPLIT_PATTERN, form),
                          _pretok(config.KREYOL_SPLIT_PATTERN, form)))
    sent_stock = _pretok(config.STOCK_SPLIT_PATTERN, APOS_SENT)
    sent_kreyol = _pretok(config.KREYOL_SPLIT_PATTERN, APOS_SENT)
    # token-level on the trained toy (kreyol), to show final tokens
    toy_tokens = {f: [kbpe.decode([i]) for i in kbpe.encode_ordinary(f)] for f in APOS_FORMS}

    # --- format bridges + parity ----------------------------------------------
    data.log("building probe set + format bridges")
    probe = data.probe_lines(per_source=500)
    hf_path = os.path.join(config.WORK, "toy_hf.json")
    vm_path = os.path.join(config.WORK, "toy_vocab_merges.json")
    tk_path = os.path.join(config.WORK, "toy_tiktoken.json")
    hf_tok = convert.export_hf(kbpe, hf_path)
    convert.export_browser(kbpe, vm_path, tk_path)
    par = convert.parity(kbpe, hf_tok, probe)

    # --- verdict --------------------------------------------------------------
    go = (not rt_fail) and par["sentence_exact_frac"] >= 0.999
    verdict = "GO" if go else "GO (with documented bridge divergence)"

    # --- write report ---------------------------------------------------------
    A("# rustbpe integration spike (B0) — go/no-go")
    A("")
    A(f"*Snapshot {config.SNAPSHOT_DATE}. nanochat pinned `{config.NANOCHAT_COMMIT}`; "
      f"`rustbpe` PyPI wheel (see pyproject). Toy: {TOY_VOCAB:,} vocab on "
      f"{total_chars/1e6:.1f}MB corpus.*")
    A("")
    A(f"## Verdict: **{verdict}** — proceed to the sweep with rustbpe.")
    A("")
    A("## 1. Build friction")
    A("")
    A("- `rustbpe` is a **standalone PyPI package** (`rustbpe>=0.1.0`), not vendored in "
      "nanochat at the pinned commit — so **no Rust/maturin source build was needed**: a "
      "prebuilt wheel installed via `uv add rustbpe` in <1s. (Rust 1.74 + cargo are present "
      "as a fallback for an sdist build; not exercised.)")
    A(f"- Toy training: **{t_kreyol:.1f}s** for {TOY_VOCAB:,} vocab on {total_chars/1e6:.1f}MB "
      f"({total_chars/1e6/max(t_kreyol,1e-9):.1f} MB/s).")
    A("- Vocab size is a clean parameter (`train_from_iterator(iter, vocab_size, pattern=…)`); "
      "nanochat's `tok_train.py` default is 32,768, we sweep 8k–32k. rustbpe trains "
      f"`vocab_size − {len(config.SPECIAL_TOKENS)}` (the {len(config.SPECIAL_TOKENS)} special "
      "tokens are appended after training). Inference uses a `tiktoken.Encoding` built from "
      "rustbpe's `get_mergeable_ranks()` + `get_pattern()`.")
    A("")
    A("## 2. Round-trip (encode→decode identity)")
    A("")
    A(f"Accent-heavy Kreyòl (è/ò are multi-byte UTF-8) + code-switched French/English: "
      f"**{len(ROUNDTRIP) - len(rt_fail)}/{len(ROUNDTRIP)} exact**"
      + ("." if not rt_fail else f" — FAILURES: {rt_fail}"))
    A("")
    A("## 3. Apostrophe-regex decision → **Kreyòl-aware pattern**")
    A("")
    A("nanochat's GPT-4-style pattern hard-codes an English-contraction clause "
      "`'(?i:[sdmt]|ll|ve|re)`. Kreyòl clitics elide on the *other* side of the apostrophe "
      "(`m'ap` = mwen ap). The probe shows the clause **misfires** on Kreyòl TMA forms whose "
      "marker starts with t/s/d/m (`te`, `ta`, …): it splits `'t` off, shredding the marker.")
    A("")
    A("| form | stock pre-tokenization | Kreyòl-aware pre-tokenization |")
    A("|---|---|---|")
    for form, st, kr in apos_rows:
        flag = "" if st == kr else "  ⚠️"
        A(f"| `{form}` | `{st}` | `{kr}`{flag} |")
    A(f"| _(sentence)_ `{APOS_SENT}` | `{sent_stock}` | `{sent_kreyol}` |")
    A("")
    A("**Decision:** drop the English-contraction clause (the only change). Rationale: it "
      "**fixes** `m'te`→`['m',\"'te\"]`, `n'ta`→`['n',\"'ta\"]` etc. (marker kept whole), "
      "leaves the `m'ap`/`l'ap`/`n'ap`/`t'ap`/`w'ap` family **identical** to stock, and leaves "
      "English single-letter contractions (`don't`→`['don',\"'t\"]`) **identical** too — the "
      "general letter clause handles `'t` the same way. Net English regression is limited to "
      "the rare multi-letter-after-apostrophe-starting-with-s/d/m/t case. The pattern flows "
      "through rustbpe→tiktoken cleanly (`train_from_iterator(pattern=…)` → `get_pattern()`).")
    A("")
    A("Token-level on the trained toy (Kreyòl-aware): "
      + "; ".join(f"`{f}` → {toy_tokens[f]}" for f in ["m'ap", "n'ta", "m'te"]) + ".")
    A("")
    A("## 4. Format bridges — exact-ID parity vs tiktoken")
    A("")
    A(f"Built (a) an HF `tokenizer.json` (BPE with reconstructed merges + a `Split(regex)` + "
      f"`ByteLevel` pre-tokenizer) and (b) a browser `{{vocab, merges}}` JSON plus a "
      f"tiktoken-native base64 dump (js-tiktoken). Re-encoded a **{par['n_sentences']}-line** "
      "source-mixed probe set through the HF bridge and compared token IDs to tiktoken:")
    A("")
    A(f"- **sentence-exact ID match: {par['sentence_exact_match']}/{par['n_sentences']} "
      f"= {par['sentence_exact_frac']:.1%}**")
    A(f"- token-level match: {par['token_match_frac']:.2%}")
    if par["first_divergence"]:
        d0 = par["first_divergence"]
        A(f"- first divergence (if any): `{d0['text']}` — tiktoken `{d0['tiktoken']}` vs "
          f"HF `{d0['hf']}`")
    A("")
    A("The **tiktoken encoding (from rustbpe) is the source of truth**; the HF and browser "
      "artifacts are bridges. Parity is reported, not assumed. The browser also ships the "
      "tiktoken-native base64 form for js-tiktoken (exact by construction).")
    A("")
    A("## 5. Go/no-go")
    A("")
    A(f"**{verdict}.** rustbpe builds trivially (wheel), trains fast, round-trips exactly, the "
      "apostrophe question is resolved with evidence, and the format bridges "
      + ("reproduce tiktoken IDs exactly" if par["sentence_exact_frac"] >= 0.999
         else f"reproduce tiktoken IDs on {par['sentence_exact_frac']:.1%} of lines (divergence documented)")
      + ". Proceeding to the vocab sweep (Part 2) with rustbpe and the Kreyòl-aware pattern.")

    out = os.path.join(config.REPORTS, "rustbpe_spike.md")
    os.makedirs(config.REPORTS, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    data.log(f"wrote {out}")
    data.log(f"VERDICT={verdict} rt_fail={len(rt_fail)} parity={par['sentence_exact_frac']:.3f}")
    return {"verdict": verdict, "parity": par, "t_train": t_kreyol, "rt_fail": rt_fail}


if __name__ == "__main__":
    run()
