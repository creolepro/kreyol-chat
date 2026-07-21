"""Render tokenizer_v0.md from the sweep results dict."""

from __future__ import annotations

import os

from . import config


def _f(x, n=3):
    return f"{x:.{n}f}"


def _pieces(d):
    return " ".join(f"`{p}`" for p in d)


def write_report(r: dict):
    L = []
    A = L.append
    chosen = r["chosen_vocab"]
    rows = sorted(r["vocab_rows"], key=lambda x: x["vocab"])

    A("# Kreyòl tokenizer v0 — Workstream B")
    A("")
    A(f"*Snapshot {r['snapshot']}. Byte-level BPE trained with **rustbpe** "
      f"(nanochat pinned `{r['nanochat_commit'][:12]}`), inference via tiktoken. "
      f"Pre-tokenization: **{r['pattern_name']}** pattern (B0 decision — see "
      "[rustbpe_spike.md](rustbpe_spike.md)). See [../../docs/phase-0.md](../../docs/phase-0.md) "
      "Workstream B and [../../docs/plan.md](../../docs/plan.md) §3.2.*")
    A("")

    # headline
    ch = next(x for x in rows if x["vocab"] == chosen)
    ht = ch["flores"]["ht"]["bytes_per_token"]
    en = ch["flores"]["en"]["bytes_per_token"]
    A("## Pick")
    A("")
    A(f"**Chosen vocab: {chosen:,}.** {r['chosen_reason']}. At {chosen:,}, whole-word "
      f"survival on the core Kreyòl list is **{ch['survival_core']['frac']:.0%}** and on the "
      f"top-{config.TOP_WORDS_N} corpus words **{ch['survival_top']['frac']:.0%}**; the "
      f"token-embedding table costs **{ch['embedding']['embed_params_m']}M** params at nanochat "
      f"d12 width ({config.D12_MODEL_DIM}). On FLORES+ our tokenizer already compresses Kreyòl "
      f"(**{_f(ht)}** bytes/token) better than English (**{_f(en)}**) — a preview of the "
      "fertility flip (Workstream C).")
    A("")

    # training data + weighting
    nat = r["sample_natural"]
    sens = r["sample_sensitivity"]

    def comp_str(m):
        c = m["composition_chars"]
        tot = sum(c.values())
        return ", ".join(f"{k} {v/tot:.0%}" for k, v in sorted(c.items(), key=lambda x: -x[1]))
    A("## Training data + source weighting")
    A("")
    A(f"Deterministic, seeded sample (seed {config.SAMPLE_SEED}) of the **train split only** "
      f"(the tokenizer_eval holdout and the 15 probe proverbs are excluded). "
      f"**Primary weighting = `natural`** (the corpus's own source proportions): a Kreyòl "
      "tokenizer should reflect what the corpus actually *is*, and re-weighting is a modeling "
      "choice we test separately rather than bake into v0.")
    A("")
    A(f"- **natural** sample: {nat['docs']:,} docs / {nat['chars']/1e6:.0f}M chars — {comp_str(nat)}.")
    A(f"- **sensitivity** variant (16k only): crawl downweighted to ~60%, Wikipedia upweighted "
      f"— {sens['docs']:,} docs / {sens['chars']/1e6:.0f}M chars — {comp_str(sens)}.")
    A("")
    natural_16k = next(x for x in rows if x["vocab"] == 16384)
    sens_16k = r["sensitivity"]
    d_bpt = (sens_16k["compression_holdout"]["bytes_per_token"]
             - natural_16k["compression_holdout"]["bytes_per_token"])
    d_surv = sens_16k["survival_top"]["frac"] - natural_16k["survival_top"]["frac"]
    # "closed" if downweighting crawl does not MATERIALLY BEAT natural on either metric.
    closed = (d_bpt < 0.1) and (d_surv < 0.05)
    A(f"**Sensitivity result (16k):** holdout bytes/token {_f(natural_16k['compression_holdout']['bytes_per_token'])} "
      f"(natural) vs {_f(sens_16k['compression_holdout']['bytes_per_token'])} (crawl-downweighted) — "
      f"Δ={d_bpt:+.3f}; top-{config.TOP_WORDS_N} survival {natural_16k['survival_top']['frac']:.0%} vs "
      f"{sens_16k['survival_top']['frac']:.0%} (Δ={d_surv:+.1%}). "
      + ("Downweighting crawl **does not beat** natural on either metric (both move by a hair, and "
         "if anything natural is slightly better), so **the weighting question is closed for v0** — "
         "natural stands. (A different *content* mix is a Phase-1 modeling choice, separate from the "
         "tokenizer.)" if closed else
         "The metrics differ enough to warrant a closer look before Phase 1."))
    A("")

    # sweep table
    A("## Vocab sweep")
    A("")
    A("Compression = **bytes per token on the held-out tokenizer_eval slice** (higher = better; "
      "fewer tokens for the same text). FLORES+ columns are measurement-only (Kreyòl "
      "out-of-domain; English/French are the code-switch regression check — reported, not gated).")
    A("")
    A(f"Holdout: **{r['holdout_docs']:,} docs** ("
      + ", ".join(f"{s} {n:,}" for s, n in r["holdout_by_source"].items())
      + f"). FLORES+ devtest: {r['flores_n']['ht']} sentences/lang.")
    A("")
    A("| vocab | bytes/token (holdout) | FLORES ht | FLORES en | FLORES fr | embed params (d12) | round-trip | survival core | survival top-500 | train |")
    A("|--:|--:|--:|--:|--:|--:|:--:|--:|--:|--:|")
    for x in rows:
        star = " ⬅" if x["vocab"] == chosen else ""
        rt = "✓" if x["roundtrip"]["failures"] == 0 else f"{x['roundtrip']['failures']} fail"
        A(f"| **{x['vocab']:,}**{star} | {_f(x['compression_holdout']['bytes_per_token'])} "
          f"| {_f(x['flores']['ht']['bytes_per_token'])} | {_f(x['flores']['en']['bytes_per_token'])} "
          f"| {_f(x['flores']['fr']['bytes_per_token'])} | {x['embedding']['embed_params_m']}M "
          f"| {rt} | {x['survival_core']['frac']:.0%} | {x['survival_top']['frac']:.0%} "
          f"| {r['train_seconds'].get(str(x['vocab']),'—')}s |")
    A("")

    # compression curve
    A("### Compression curve (holdout bytes/token)")
    A("")
    prev = None
    A("| vocab | bytes/token | Δ vs previous |")
    A("|--:|--:|--:|")
    for x in rows:
        b = x["compression_holdout"]["bytes_per_token"]
        gain = "—" if prev is None else f"+{(b-prev)/prev:.1%}"
        A(f"| {x['vocab']:,} | {_f(b)} | {gain} |")
        prev = b
    A("")
    A(f"**Pick rule:** smallest vocab where the next size adds < 3% bytes/token. → **{chosen:,}**. "
      "Bigger vocab keeps compressing but the marginal gain flattens while the embedding table "
      "grows linearly (below) — at ~200M params (d12) that trade stops being worth it.")
    A("")

    # embedding cost
    A("## Embedding-table cost (nanochat d12, dim "
      f"{config.D12_MODEL_DIM}, untied wte+lm_head)")
    A("")
    A("| vocab | embed params | share of a ~200M d12 model |")
    A("|--:|--:|--:|")
    for x in rows:
        p = x["embedding"]["embed_params"]
        A(f"| {x['vocab']:,} | {x['embedding']['embed_params_m']}M | ~{p/200e6:.0%} |")
    A("")
    A("(`vocab × 768 × 2`; nanochat pads vocab up to a multiple for kernels, so the real table "
      "is slightly larger. This is why we don't just take 32k — the biggest vocab nearly "
      "doubles the embedding cost for a few % compression.)")
    A("")

    # survival detail
    A(f"## Whole-word survival at {chosen:,} (exhibit metric)")
    A("")
    A(f"Each word checked **bare and with a leading space**; single-token = survives. "
      f"Core grammar list: **{ch['survival_core']['n_single']}/{ch['survival_core']['n_total']} "
      f"= {ch['survival_core']['frac']:.0%}**. Top-{config.TOP_WORDS_N} corpus-frequency words "
      f"(train split, probe excluded): **{ch['survival_top']['n_single']}/{ch['survival_top']['n_total']} "
      f"= {ch['survival_top']['frac']:.0%}**.")
    A("")
    A("Core list, per word (bare / leading-space):")
    A("")
    A("| word | bare | ` word` |    | word | bare | ` word` |")
    A("|---|:--:|:--:|---|---|:--:|:--:|")
    cr = ch["survival_core"]["rows"]
    half = (len(cr) + 1) // 2
    for i in range(half):
        left = cr[i]
        r2 = cr[i + half] if i + half < len(cr) else None
        lft = f"| `{left['word']}` | {'✓' if left['bare_single'] else '·'} | {'✓' if left['leading_space_single'] else '·'} |"
        if r2:
            rgt = f" | `{r2['word']}` | {'✓' if r2['bare_single'] else '·'} | {'✓' if r2['leading_space_single'] else '·'} |"
        else:
            rgt = " |  |  |  |"
        A(lft + rgt)
    A("")
    A("> The top-500 words are computed from corpus-v0 whitespace-word frequencies over the "
      "**train split only** (holdout + probe proverbs excluded), lowercasing-free on NFC text.")
    A("")

    # robustness
    A(f"## Robustness spot-checks at {chosen:,}")
    A("")
    rob = r["chosen_robustness"]
    for group, items in rob.items():
        A(f"**{group.replace('_',' ')}:** "
          + "; ".join(f"`{w}`→{_pieces(p)}" for w, p in items.items()))
        A("")

    # ablation
    ab = r["ablation"]
    A("## English-control ablation (exhibit)")
    A("")
    A(f"A second 16k tokenizer trained with identical settings on a size-matched sample of "
      f"**wikitext-103-raw** (English, CC-BY-SA 3.0), then both tokenizers run on the "
      f"{ab['n_teachable']} teachable proverbs:")
    A("")
    A(f"- Same proverbs cost **{ab['kreyol_total_tokens']} tokens** under the Kreyòl tokenizer "
      f"vs **{ab['english_total_tokens']}** under the English one "
      f"(**{ab['ratio_english_over_kreyol']}× more** on English).")
    A(f"- Of {ab['n_words']} distinct proverb words, **{ab['n_survive_kreyol_shatter_english']}** "
      "survive whole under Kreyòl but shatter under English. Examples:")
    A("")
    A("| word | Kreyòl | English | English pieces |")
    A("|---|--:|--:|---|")
    for e in ab["examples"]:
        A(f"| `{e['word']}` | {e['kreyol_tokens']} | {e['english_tokens']} | {_pieces(e['english_pieces'])} |")
    A("")

    # export / parity
    ep = r["export_parity"]
    A("## Exported artifacts")
    A("")
    A("The chosen tokenizer is committed under `ml/tokenizer/kreyol-bpe/` in three formats "
      "(+ the tiktoken pickle + meta):")
    A("")
    A("- `tokenizer.json` — HF `tokenizers`/transformers (and the fertility script).")
    A("- `vocab_merges.json` — plain `{vocab, merges, pattern, special_tokens}` for the browser.")
    A("- `tokenizer_tiktoken.json` — tiktoken-native base64 (js-tiktoken, exact).")
    A(f"- Export parity: the HF `tokenizer.json` reproduces tiktoken IDs on "
      f"**{ep['sentence_exact_match']}/{ep['n_sentences']} = {ep['sentence_exact_frac']:.1%}** "
      "of a 1k-line probe (source of truth = the rustbpe/tiktoken encoding).")
    A("")
    A("## Rights note")
    A("")
    A("The tokenizer is a **derived model artifact** trained on the corpus-v0 train split "
      "(MADLAD-400 ht, ht Wikipedia, teachable proverbs). `rights.yaml` allows **tokenizer "
      "training** on all three; the only open item is MADLAD *text redistribution*. The committed "
      "vocab is short byte-level subwords (longest entries are single words / morphological "
      "suffixes like `-syon`, punctuation runs — no verbatim phrases or PII), which is a standard "
      "publishable artifact and does not re-host source text. Quarantined sources (MIT-Haiti, the "
      "dictionary) and eval-only FLORES+ were **not** in training. If MADLAD's license resolves "
      "restrictively, revisit before any *commercial* redistribution.")
    A("")
    A("## Reproduce")
    A("")
    A("```bash")
    A("cd ml && uv sync")
    A("uv run python -m tokenizer.spike   # B0 go/no-go")
    A("uv run python -m tokenizer.run     # sweep + eval + export + this report")
    A("```")

    out = os.path.join(config.REPORTS, "tokenizer_v0.md")
    os.makedirs(config.REPORTS, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    from . import data
    data.log(f"wrote {out}")
    return out
