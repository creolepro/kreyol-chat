"""Render the Workstream D outputs:

  * reports/base_model_probe.md      — scorecard + provisional pick + proverb detail
  * reports/probe_naturalness_sheet.md — blinded, shuffled, de-identified outputs
  * probe/naturalness_key.json       — the hidden de-identification key (committed
                                       separately so the sheet stays blind)

Scores are model-generated / owned text only — no FLORES source/reference text is
written here (that stays in the git-ignored raw results).
"""

from __future__ import annotations

import json
import os
import random

from . import config

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPORTS = os.path.join(REPO_ROOT, "ml", "reports")
PROBE_DIR = os.path.dirname(os.path.abspath(__file__))

# display-only approximate parameter counts
PARAMS = {"qwen3-1.7b": "1.7B", "qwen3-4b": "4B", "smollm3-3b": "3B",
          "gemma3-4b": "4B", "llama3.2-3b": "3B"}
ORDER = [c["key"] for c in config.CANDIDATES]


def _fmt(x, nd=3):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "—"


def _short_err(e):
    e = (e or "").split("\n")[0]
    return (e[:80] + "…") if len(e) > 82 else e


def write_all(cards, top, fulldev_scores, fulldev_capped, slices_meta, cost):
    _write_report(cards, top, fulldev_scores, fulldev_capped, slices_meta, cost)
    _write_naturalness(cards)


# --- main report --------------------------------------------------------------

def _write_report(cards, top, fulldev_scores, fulldev_capped, slices_meta, cost):
    ok_cards = {k: c for k, c in cards.items() if c.get("ok")}
    L = []
    L.append("# Base-model probe — Workstream D (Phase 0b)\n")
    L.append(f"_Snapshot {config.SNAPSHOT_DATE}. Precision: **unquantized bf16** "
             f"(transformers, `torch_dtype=bfloat16`). Selection corpus: FLORES+ "
             f"**dev** only (`{config.FLORES_REPO}` @ `{config.FLORES_REVISION[:12]}`); "
             f"`final_devtest` untouched. GPU: Modal {config.MODAL_GPU}._\n")

    L.append("**Question.** Which pretrained base starts from the strongest Kreyòl "
             "position for Model B's continued-pretraining? Reputation doesn't "
             "settle it (none publish HT evals); a few dollars of GPU does.\n")

    # provisional pick
    if ok_cards:
        pick = top[0] if top else None
        L.append("## Provisional pick — **pending human naturalness review**\n")
        if pick:
            pc = cards[pick]
            L.append(f"On the two **automated** measures, **`{pc['repo']}`** "
                     f"(`{pick}`) leads: authored-Kreyòl BPB **{_fmt(pc['bpb_authored'])}** "
                     f"(primary, lower is better) and eng→hat chrF2++ "
                     f"**{_fmt(pc['mt_e2h']['chrf2pp'],1)}**. This ranking is "
                     f"**provisional**: the blinded naturalness rubric "
                     f"([probe_naturalness_sheet.md](probe_naturalness_sheet.md)) is "
                     f"scored by a fluent speaker and folds into the final decision.\n")
        L.append("")

    # scorecard
    L.append("## Scorecard\n")
    L.append("BPB = bits-per-byte (primary, ↓). chrF2++ / spBLEU on the 250-item "
             "dev subset (↑). Proverbs = exact-continuation hits / near-misses out "
             "of 15. Naturalness pending (see sheet).\n")
    L.append("| Model | Params | BPB authored ↓ | BPB full ↓ | chrF2++ e→h | spBLEU e→h | chrF2++ h→e | spBLEU h→e | Proverbs hit/near | Naturalness |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for k in ORDER:
        c = cards.get(k, {})
        name = f"`{c.get('repo','?')}`" + (" _(control)_" if c.get("control") else "")
        if not c.get("ok"):
            L.append(f"| {name} | {PARAMS.get(k,'?')} | — | — | — | — | — | — | — | "
                     f"_skipped: {_short_err(c.get('error'))}_ |")
            continue
        p = c["proverbs"]
        star = " ⭐" if k in top else ""
        L.append(f"| {name}{star} | {PARAMS.get(k,'?')} | **{_fmt(c['bpb_authored'])}** | "
                 f"{_fmt(c['bpb_full'])} | {_fmt(c['mt_e2h']['chrf2pp'],1)} | "
                 f"{_fmt(c['mt_e2h']['spbleu'],1)} | {_fmt(c['mt_h2e']['chrf2pp'],1)} | "
                 f"{_fmt(c['mt_h2e']['spbleu'],1)} | {p['n_hit']}/{p['n_near']} | "
                 f"_pending_ |")
    L.append("\n⭐ = promoted to the full-dev MT stage.\n")

    # full-dev refinement
    if fulldev_scores:
        n = next(iter(fulldev_scores.values())).get("n")
        cap_note = ("  \n_Bounded to a seeded {n}-sentence sample (not the full 992) "
                    "for cost: Gemma-3's multimodal `generate` path is ~20× slower "
                    "than the plain causal LMs. Both finalists are scored on the "
                    "identical sample._").format(n=n) if fulldev_capped else ""
        L.append(f"## Full-dev refinement (top {len(fulldev_scores)}, {n} sentences)\n"
                 + cap_note + "\n")
        L.append("| Model | chrF2++ e→h | spBLEU e→h | chrF2++ h→e | spBLEU h→e |")
        L.append("|---|---|---|---|---|")
        for k in ORDER:
            if k not in fulldev_scores:
                continue
            s = fulldev_scores[k]
            L.append(f"| `{cards[k]['repo']}` | {_fmt(s['mt_e2h']['chrf2pp'],1)} | "
                     f"{_fmt(s['mt_e2h']['spbleu'],1)} | {_fmt(s['mt_h2e']['chrf2pp'],1)} | "
                     f"{_fmt(s['mt_h2e']['spbleu'],1)} |")
        L.append("")

    # findings / reading
    if ok_cards:
        L.append("## Findings\n")
        rank = sorted(ok_cards, key=lambda k: ok_cards[k]["bpb_authored"])
        win = rank[0]
        chrfs = sorted((c["mt_e2h"]["chrf2pp"] for c in ok_cards.values()), reverse=True)
        runner = chrfs[1] if len(chrfs) > 1 else 0
        margin = (f", ~{chrfs[0] / runner:.1f}× the runner-up" if runner else "")
        L.append(f"- **{cards[win]['repo']} starts from the strongest Kreyòl "
                 f"position** on both automated axes — lowest authored BPB "
                 f"({_fmt(cards[win]['bpb_authored'])}) and highest eng→hat chrF2++ "
                 f"({_fmt(cards[win]['mt_e2h']['chrf2pp'],1)}{margin}). It is also the "
                 f"only model with an exact proverb hit.")
        if "smollm3-3b" in ok_cards and rank[-1] == "smollm3-3b":
            L.append("- **SmolLM3-3B is the *worst* on BPB** despite its strong-French "
                     "reputation (the reason it was hypothesized to transfer well to a "
                     "French-lexifier creole). French coverage did **not** translate "
                     "into a Kreyòl exposure advantage here — a direct, if negative, "
                     "answer to that open question.")
        ctrl = [k for k in ok_cards if cards[k].get("control")]
        if ctrl and rank[-1] not in ctrl:
            ck = ctrl[0]
            pos = rank.index(ck) + 1
            L.append(f"- **The control ({cards[ck]['repo']}) is not the floor** — it "
                     f"ranks #{pos} of {len(rank)} on BPB. Llama 3.2 lists 8 official "
                     f"languages (HT not among them), yet it beats the Qwen3 and "
                     f"SmolLM3 bases on authored-Kreyòl BPB.")
        best_chrf = max(c["mt_e2h"]["chrf2pp"] for c in ok_cards.values())
        best_hit = max(c["proverbs"]["n_hit"] for c in ok_cards.values())
        if best_chrf < 15 and best_hit == 0:
            L.append("- **Every candidate is weak on Kreyòl in absolute terms** (best "
                     "eng→hat chrF2++ < 15, zero exact proverb recall). That is a "
                     "*finding*, not a failure — it quantifies how much the CPT stage "
                     "must add.")
        else:
            L.append("- Absolute quality is modest across the board — these are base "
                     "checkpoints doing completion-style few-shot, and greedy decoding "
                     "makes them loop (visible in the naturalness sheet). The point is "
                     "**relative** starting position for CPT, not usable output.")
        L.append("- These are the **\"before\" numbers** banked for the later "
                 "before/after adaptation comparison (Model B CPT).")
        if cards[win].get("repo", "").startswith("google/gemma"):
            L.append("- **License caveat (decisive for the final pick).** The leader "
                     "`google/gemma-3-4b-pt` is under the **Gemma Terms of Use**, not "
                     "Apache-2.0. plan.md §3.1 requires an Apache-2.0-clean base for "
                     "this community project (Gemma listed *only if its custom license "
                     "is acceptable*). So the quality ranking and the license gate are "
                     "**separate decisions**: if the Gemma terms are ruled out, the "
                     "best Apache-2.0-clean base is **`" +
                     next((cards[k]["repo"] for k in rank
                           if not cards[k]["repo"].startswith(("google/", "meta-llama/"))),
                          "the top Qwen3") +
                     "`** (next Apache-2.0 base by BPB). Llama-3.2 is also non-Apache "
                     "(Llama license) and is the control, not a candidate.\n")
        else:
            L.append("")

    # proverb detail (Station 2 preview) — owned text, safe to show
    L.append("## Proverb completion — raw outputs (Station 2 preview)\n")
    L.append("Each probe proverb is cut at its midpoint; the model completes the "
             "line (5-shot format, greedy). ✓ = exact continuation, ≈ = near-miss "
             "(chrF ≥ 50). The 15 probe proverbs are the held-out probe split — in "
             "no training or eval set.\n")
    L.append("_Note: the two format exemplars are real proverbs (`Piti piti zwazo fè "
             "nich li.` / `Men anpil, chay pa lou.`); models with no recall for the "
             "target proverb often just echo the exemplar tail (`…chay pa lou`), so a "
             "low chrF reflects echo, not only ignorance. A genuine hit (e.g. Gemma on "
             "`Dèyè mòn gen mòn.`) is therefore the signal that matters._\n")
    prov_models = [k for k in ORDER if k in ok_cards]
    if prov_models:
        any_card = ok_cards[prov_models[0]]
        for i, row in enumerate(any_card["proverbs"]["rows"]):
            L.append(f"**{row['num']}. {row['full']}**  \n_“{row['english']}”_  \n"
                     f"prompt → `{row['prompt_half']}` · gold → `{row['gold']}`\n")
            L.append("| Model | Completion | chrF | |")
            L.append("|---|---|---|---|")
            for k in prov_models:
                r = cards[k]["proverbs"]["rows"][i]
                mark = "✓" if r["hit"] else ("≈" if r["near"] else "")
                comp = (r["completion"] or "").replace("|", "\\|")[:120] or "_(empty)_"
                L.append(f"| `{k}` | {comp} | {_fmt(r['chrf'],1)} | {mark} |")
            L.append("")

    # method / provenance
    L.append("## Method & provenance\n")
    fm = slices_meta
    L.append("**BPB (bits-per-byte, primary).** `total_nll_bits / total_utf8_bytes` "
             "over the corpus **tokenizer_eval holdout** (never-trained docs, "
             "Workstream B definition). Each doc scored independently (no cross-doc "
             "context); the tokenizer's BOS id — or EOS id where no BOS exists "
             "(GPT-2 document-start convention) — is prepended as context and never "
             "scored; every real token is scored; docs longer than "
             f"{config.BPB_MAX_LEN} tokens are split into windows each re-seeded "
             "with the start token. Denominator is UTF-8 bytes, so BPB is "
             "**cross-tokenizer comparable**. Two slices:\n")
    if fm:
        full = fm.get("full", {})
        auth = fm.get("authored", {})
        L.append(f"- **authored-only** (primary signal): Wikipedia non-stub + owned "
                 f"docs — {auth.get('n_docs','?')} docs, {auth.get('bytes','?'):,} B "
                 f"(scored in full).")
        L.append(f"- **full holdout**: {full.get('n_docs','?')} of "
                 f"{full.get('n_docs_available','?')} holdout docs, seed-subsampled "
                 f"to a {config.BPB_FULL_BYTE_BUDGET:,}-byte budget "
                 f"({full.get('bytes','?'):,} B).\n")
    # signatures
    sig_card = next(iter(ok_cards.values()), None)
    if sig_card:
        L.append("**MT few-shot completion.** 5-shot, fixed template, greedy "
                 "(`do_sample=False, num_beams=1`), stop at newline. Template per "
                 "line: `English: …` / `Haitian Creole: …`. sacreBLEU signatures:")
        L.append(f"- spBLEU: `{sig_card['mt_e2h']['spbleu_sig']}`")
        L.append(f"- chrF2++: `{sig_card['mt_e2h']['chrf2pp_sig']}`\n")
    L.append(f"**External context (NOT comparable).** {config.ROBINSON_2023_NOTE}\n")

    # revisions + cost + reproduce
    L.append("## Runs, revisions, cost\n")
    L.append("| Model | Resolved revision | transformers | load s | total s |")
    L.append("|---|---|---|---|---|")
    for k in ORDER:
        c = cards.get(k, {})
        if not c.get("ok"):
            L.append(f"| `{c.get('repo','?')}` | _n/a_ | — | — | — |")
            continue
        L.append(f"| `{c['repo']}` | `{c['revision']}` | {c.get('transformers','?')} | "
                 f"{_fmt(c.get('load_s'),1)} | {_fmt(c.get('total_s'),1)} |")
    L.append(f"\n**GPU cost (estimate).** ~{cost['gpu_seconds']/60:.1f} GPU-min on "
             f"Modal {config.MODAL_GPU} ≈ **${cost['est_usd']:.2f}** at "
             f"${config.MODAL_L40S_USD_PER_HR}/GPU-h (list price; Modal's dashboard "
             f"is authoritative). BPB adds no generation cost (forward pass only).\n")
    L.append("**Reproduce.** `uv run python -m probe.run --stage all` "
             "(smoke → main → fulldev → report). Modal auth required; weights cache "
             "in a Modal volume so reruns skip re-download. Raw results (with FLORES "
             "prompts/refs) stay under the git-ignored `ml/data/probe/`.\n")

    os.makedirs(REPORTS, exist_ok=True)
    with open(os.path.join(REPORTS, "base_model_probe.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))


# --- naturalness sheet + hidden key ------------------------------------------

def _write_naturalness(cards):
    ok = [k for k in ORDER if cards.get(k, {}).get("ok")]
    if not ok:
        return
    rng = random.Random(config.NAT_SHUFFLE_SEED)
    key = {}   # prompt_id -> {label: model_key}
    S = []
    S.append("# Blinded naturalness review — base-model probe\n")
    S.append("_10 fixed Kreyòl completion prompts. Each model's greedy completion "
             "is shown de-identified (labels are shuffled independently per prompt). "
             "The key is committed separately (`probe/naturalness_key.json`) so this "
             "sheet stays blind._\n")
    S.append("**Rubric (score each labelled output 1–3):** "
             "**1 = unusable** (not Kreyòl / word-salad); "
             "**2 = degraded but Kreyòl** (recognizably Kreyòl, errors/odd); "
             "**3 = plausible Kreyòl** (a fluent speaker could have written it). "
             "Write your score next to each label.\n")
    labels = [chr(ord("A") + i) for i in range(len(ok))]
    for pi, spec in enumerate(config.NATURALNESS_PROMPTS):
        order = ok[:]
        rng.shuffle(order)
        key[spec["id"]] = {labels[j]: order[j] for j in range(len(order))}
        S.append(f"## {pi+1}. _{spec['category']}_\n")
        S.append(f"**Prompt (given to the model):** `{spec['prompt']}`\n")
        for j, mkey in enumerate(order):
            comp = cards[mkey]["naturalness_raw"][pi] if pi < len(
                cards[mkey]["naturalness_raw"]) else ""
            comp = (comp or "").replace("\n", " ").strip() or "_(empty)_"
            S.append(f"- **{labels[j]}** ⟶ {comp}  \n  _score: ____")
        S.append("")

    with open(os.path.join(REPORTS, "probe_naturalness_sheet.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(S))
    with open(os.path.join(PROBE_DIR, "naturalness_key.json"), "w",
              encoding="utf-8") as f:
        json.dump({"seed": config.NAT_SHUFFLE_SEED, "models": ok, "key": key},
                  f, ensure_ascii=False, indent=1)
