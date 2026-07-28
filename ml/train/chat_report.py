"""Workstream I — reports for Model C chat.

Generates:
  • ml/reports/modelc_chat.md          — data composition + license table, training config,
                                          evals (chat frozen prompts, BPB regression vs v1),
                                          before/after examples, conversion artifacts.
  • ml/reports/chat_naturalness_sheet.md — a BLINDED sheet of ~30 chat outputs for a second
                                          native review (same worksheet format as the kakugo sheet).

Reads the self-persisted result JSONs under data/train_work/chat/ + the v1 base BPB from
g_flagship_v1_results.json. Robust to missing inputs (writes what it has).

Run:  uv run python -m train.chat_report [report|sheet|all]
"""

from __future__ import annotations

import argparse
import json
import os
import random

from . import chat_config as CC
from . import config as F
from . import llama_config as G

REPORTS = os.path.join(F.REPO_ROOT, "reports")
V1_RESULTS = os.path.join(G.G_WORK, "g_flagship_v1_results.json")


def _load(path, default=None):
    if path and os.path.exists(path):
        try:
            return json.load(open(path, encoding="utf-8"))
        except Exception:
            return default
    return default


def _w(name):
    return os.path.join(CC.CHAT_WORK, name)


def _bpb_row(bpb, slices):
    return {s: (bpb.get(s, {}) or {}).get("bpb") for s in slices}


SLICES = ["general_holdout", "authored_eval", "authored_eval_v2",
          "translation_shaped_eval", "flores_hat"]
SLICE_LABEL = {"general_holdout": "general", "authored_eval": "authored",
               "authored_eval_v2": "authored_v2", "translation_shaped_eval": "translation",
               "flores_hat": "FLORES"}


def build_naturalness_sheet():
    ev = _load(_w("chat_eval_results.json"))
    if not ev or not ev.get("naturalness"):
        print("[chat_report] no eval results yet — skipping naturalness sheet")
        return
    rows = list(ev["naturalness"])
    random.Random(20260728).shuffle(rows)          # blind the order
    lines = ["# Blinded naturalness review — Model C chat outputs", "",
             "> **For a native Haitian-Creole reviewer.** Below are chat answers the model",
             "> produced (greedy). The prompts are shown; nothing about how they were generated",
             "> is revealed, and the order is shuffled. Judge the KREYÒL only.", "",
             "## How to score each item", "",
             "For every item: **(1) Naturalness 1–5** (1=not Kreyòl/nonsense, 3=understandable but",
             "awkward, 5=fully natural) — write it in the `____` slot; **(2)** tick the box if the",
             "answer is factually wrong, off-topic, or awkward.", "", "---", ""]
    for i, r in enumerate(rows, 1):
        lines += [f"## Item {i}", "",
                  "**Prompt (user):**", f"> {r['prompt']}", "",
                  "**Model answer:**", f"> {(r['answer'] or '(empty)').replace(chr(10), ' ')}", "",
                  "_naturalness (1–5):_ `____`  ",
                  "- [ ] **wrong / awkward** — tick if the answer is wrong, off-topic, or awkward", ""]
    path = os.path.join(REPORTS, "chat_naturalness_sheet.md")
    open(path, "w", encoding="utf-8").write("\n".join(lines))
    print(f"[chat_report] wrote {path} ({len(rows)} items)")


def build_report():
    dl = _load(_w("download_stats.json"), {})
    pilot = _load(_w("layer2_pilot_report.json"), {})
    full = _load(_w("layer2_full_report.json"), {})
    bins = _load(_w("chat_bins_manifest.json"), {}).get("bins", {})
    mid = _load(_w("chat_midtrain_results.json"), {})
    sft = _load(_w("chat_sft_results.json"), {})
    ev = _load(_w("chat_eval_results.json"), {})
    conv = _load(_w("chat_convert_results.json"), {})
    v1 = _load(V1_RESULTS, {})
    v1_bpb = (v1.get("final_full_bpb") or {})

    def g(d, *ks, default="—"):
        for k in ks:
            d = (d or {}).get(k, {}) if isinstance(d, dict) else {}
        return d if d not in ({}, None) else default

    L = ["# Model C chat — midtraining + SFT (the first conversational Kreyòl-first model)", "",
         f"*Snapshot {CC.SNAPSHOT_DATE}. Continues the Model C v1 base (123M, d12) through a "
         f"three-layer SFT stack (docs/data.md §3): midtraining (format) → SFT (voice), on Modal H100. "
         f"Nothing under `ml/data/` is committed.*", ""]

    # ---- data composition + license table ----
    L += ["## Data composition + licenses", "",
          "Real-data layers (Layer 1 midtraining, Layer 3 SFT cap) + the corpus-grounded Layer 2 "
          "(the quality core). Kept/dropped counts are from the build; every source was registered "
          "in `rights.yaml` **before** ingestion, screened for the 15 probe proverbs, and (for the "
          "templated sources) had all FLORES-derived rows dropped so our MT eval never leaks.", "",
          "| source | layer | license | kept | notes |", "|---|---|---|--:|---|"]
    k = lambda s: (dl.get(s, {}) or {}).get("kept", "—")
    L += [
        f"| kakugo-hat | 1 | Apache-2.0 | {k('kakugo')} | multi-turn; stripped English `<think>`/system, per-turn langid, dedup |",
        f"| aya_collection (hat) | 1 | Apache-2.0 | {k('aya_collection')} | templated bulk, capped + deduped (format variety) |",
        f"| xP3x (hat_Latn) | 1 | Apache-2.0 | {k('xp3x')} | **0 kept — 100% FLORES-derived → dropped (eval carve-out)** |",
        f"| translation-QA | 1 | Federal PD + CMU | {k('translation')} | EN↔HT turns from the PD glossary + CMU lexicon |",
        f"| Layer 2 (generated) | 2 | corpus-grounded | {(full.get('total_generated') if full else pilot.get('pilot_n_ok','—'))} | claude-opus-4-8 over authored VOA/wiki/legal passages |",
        f"| muri-it (hat) | 3 | Apache-2.0 | {k('muri_it')} | native-output pairs; response-masked |",
        f"| aya gold (hat) | 3 | Apache-2.0 | {k('aya_gold')} | human-written gold |",
    ]
    L += ["",
          f"**Layer 1 (midtrain)** packs to **{g(bins,'midtrain','train_tokens'):,}** tokens "
          f"across {g(bins,'midtrain','n_train_convos')} conversations (full-sequence loss). "
          f"**Layer 3 (SFT)** packs to **{g(bins,'sft','train_tokens'):,}** tokens across "
          f"{g(bins,'sft','n_train_convos')} conversations, response-masked "
          f"(loss-token fraction {g(bins,'sft','loss_token_frac')}).", ""]

    # ---- Layer-2 pilot cost ----
    if pilot:
        L += ["### Layer-2 generation (pilot-gated)", "",
              f"Pilot: **{pilot.get('pilot_n_ok')}/{pilot.get('pilot_n_requested')} clean** at "
              f"**${pilot.get('pilot_cost_usd')}** (${pilot.get('cost_per_ok_conversation')}/conv; "
              f"{pilot.get('in_tok_per_conv')} in / {pilot.get('out_tok_per_conv')} out tokens). "
              f"Projected 3–5k = ${pilot.get('projected_cost_3000')}–{pilot.get('projected_cost_5000')} "
              f"(Opus list rates; the API dashboard is the true bill), which exceeds the ≈$75 budget → "
              f"ran a budget-fit batch instead.",
              (f"Full run: **{full.get('total_generated')}** total Layer-2 conversations "
               f"(${(pilot.get('pilot_cost_usd') or 0)}+ batch)." if full else ""), ""]

    # ---- training config ----
    L += ["## Training", "",
          "| stage | resume from | steps | epochs | tok/step | peak LR | loss | final loss |",
          "|---|---|--:|--:|--:|--:|---|--:|",
          f"| midtrain | v1 base (step {CC.V1_BASE_STEP}) | {mid.get('num_iterations','—')} | "
          f"{mid.get('epochs','—')} | {CC.MIDTRAIN['total_batch_size']:,} | {CC.MIDTRAIN['peak_lr']} | "
          f"full-seq | {mid.get('final_loss','—')} |",
          f"| SFT | midtrain final | {sft.get('num_iterations','—')} | {sft.get('epochs','—')} | "
          f"{CC.SFT['total_batch_size']:,} | {CC.SFT['peak_lr']} | response-masked | {sft.get('final_loss','—')} |",
          ""]

    # ---- BPB regression ----
    if ev.get("bpb"):
        chat_bpb = _bpb_row(ev["bpb"], SLICES)
        base_bpb = _bpb_row(v1_bpb, SLICES)
        L += ["## BPB regression — did chat tuning damage the language model?", "",
              "BPB on the standing slices, chat model vs the v1 base (same slices, byte-normalized). "
              "Δ>0 means the chat model is slightly worse at raw LM on that slice — expected and small; "
              "the goal is a model that *answers*, without wrecking the language.", "",
              "| slice | v1 base | chat | Δ |", "|---|--:|--:|--:|"]
        for s in SLICES:
            b, c = base_bpb.get(s), chat_bpb.get(s)
            d = (round(c - b, 4) if (b is not None and c is not None) else "—")
            L += [f"| {SLICE_LABEL[s]} | {round(b,4) if b else '—'} | {round(c,4) if c else '—'} | {d} |"]
        L += [""]

    # ---- before/after: v1 base continuation vs chat answer ----
    base_gens = {}
    for ck in (v1.get("train", {}) or {}).get("checkpoints", []):
        if ck.get("generations"):
            for gg in ck["generations"]:
                base_gens[gg["id"]] = gg["completion"]
    if ev.get("frozen_chat"):
        L += ["## Before / after — same prompt, base continuation vs chat answer", ""]
        shown = 0
        for fc in ev["frozen_chat"]:
            if shown >= 3:
                break
            base = base_gens.get(fc["id"])
            if not base:
                continue
            L += [f"**Prompt ({fc['id']}):** {fc['prompt']}", "",
                  f"- **v1 base (continues):** {base.strip()[:220]}",
                  f"- **chat (answers):** {(fc['chat_answer'] or '(empty)').strip()[:220]}", ""]
            shown += 1

    # ---- frozen prompts in chat mode ----
    if ev.get("frozen_chat"):
        L += ["## The 10 frozen prompts, answered in chat mode", "",
              "The continuer→answerer transition (archived alongside the slider assets). Greedy.", ""]
        for fc in ev["frozen_chat"]:
            L += [f"**[{fc['id']}]** {fc['prompt']}", f"> {(fc['chat_answer'] or '(empty)').replace(chr(10),' ')}", ""]

    # ---- sampled exhibit ----
    if ev.get("exhibit_sampled"):
        L += ["## Exhibit — temperature-sampled (labeled sampled)", ""]
        for e in ev["exhibit_sampled"]:
            L += [f"**{e['prompt']}**", f"> {(e['sampled'] or '').replace(chr(10),' ')}", ""]

    # ---- conversion ----
    if conv:
        a = conv.get("artifacts", {})
        kv = conv.get("gguf_tokenizer_kv", {})
        L += ["## Deployment — full conversion chain (BOS fix + chat template embedded)", "",
              f"GGUF **f16 {round((a.get('gguf_f16',{}).get('bytes') or 0)/1e6)} MB** "
              f"(sha256 `{(a.get('gguf_f16',{}).get('sha256_16') or '')}…`) / "
              f"**Q4_K_M {round((a.get('gguf_q4',{}).get('bytes') or 0)/1e6)} MB** "
              f"(sha256 `{(a.get('gguf_q4',{}).get('sha256_16') or '')}…`) + an ONNX/transformers.js "
              f"bundle + an Ollama Modelfile with the template (Modal Volume).",
              f"GGUF tokenizer metadata carries the Part-0 fix + the chat template: "
              f"`add_bos_token={kv.get('tokenizer.ggml.add_bos_token')}`, "
              f"`chat_template={'embedded' if kv.get('has_chat_template') else 'missing'}`. "
              f"Gate 1 export Δ={g(conv,'gate1_logits','max_abs_logit_diff_fp32_roundtrip')}, "
              f"gate 5 ONNX gen_ok={g(conv,'gate5_onnx','onnx_gen_ok')}.", ""]

    L += ["## Naturalness review", "",
          "A blinded 30-output sheet ([chat_naturalness_sheet.md](chat_naturalness_sheet.md)) is built "
          "for a second native review (same worksheet format as the kakugo audit).", ""]

    path = os.path.join(REPORTS, "modelc_chat.md")
    open(path, "w", encoding="utf-8").write("\n".join(str(x) for x in L))
    print(f"[chat_report] wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["report", "sheet", "all"], default="all", nargs="?")
    args = ap.parse_args()
    os.makedirs(REPORTS, exist_ok=True)
    if args.what in ("sheet", "all"):
        build_naturalness_sheet()
    if args.what in ("report", "all"):
        build_report()


if __name__ == "__main__":
    main()
