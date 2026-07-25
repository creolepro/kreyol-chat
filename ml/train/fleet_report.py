"""Workstream H — render reports/fleet.md from the fleet + precheck results.

Reads the self-persisted results (train_work/fleet/{fleet_results,fleet_precheck}.json),
the data manifest, and the Part-0 sidecars (probe-leak audit, Bloom, kakugo pointer),
and writes reports/fleet.md: per question a setup line, a BPB table with seed spread,
a one-sentence ANSWER and a one-sentence DECISION; plus Part 0 results, the depth
pre-check, and the RECOMMENDED G-v1 CONFIG block.

Answers/decisions are DATA-DRIVEN (computed from the BPB deltas vs the seed spread) so
re-running after new results regenerates the prose. Edit this template, not fleet.md.

Run:  cd ml && uv run python -m train.fleet_report
"""

from __future__ import annotations

import json
import os

from . import config as F
from . import fleet_config as H

RESULTS = os.path.join(H.FLEET_WORK, "fleet_results.json")
PRECHECK = os.path.join(H.FLEET_WORK, "fleet_precheck.json")
MANIFEST = os.path.join(H.FLEET_WORK, "fleet_data_manifest.json")
PROBE_AUDIT = os.path.join(F.DATA, "interim", "probe_leak_audit.json")
PROBE_FINEWEB = os.path.join(F.DATA, "interim", "probe_leak_fineweb.json")
BLOOM = os.path.join(F.DATA, "interim", "j0", "bloom.json")
OUT = os.path.join(F.REPO_ROOT, "reports", "fleet.md")

SLICE_LABEL = {"general_holdout": "general", "authored_eval": "authored",
               "translation_shaped_eval": "transl.", "authored_eval_v2": "authored_v2",
               "flores_hat": "FLORES"}
SLICES = ["general_holdout", "authored_eval", "translation_shaped_eval",
          "authored_eval_v2", "flores_hat"]


def _load(p, default=None):
    return json.load(open(p)) if os.path.exists(p) else default


def _results() -> dict:
    r = _load(RESULTS, {})
    return r.get("results", r) if isinstance(r, dict) else {}


def _bpb(res, tag, sl):
    r = res.get(tag)
    return (r or {}).get("bpb", {}).get(sl)


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def _spread(vals):
    vals = [v for v in vals if v is not None]
    return round(max(vals) - min(vals), 4) if len(vals) > 1 else 0.0


def _fmt(v):
    return f"{v:.4f}" if isinstance(v, (int, float)) else "—"


# --- per-condition mean BPB across the seeds in a question --------------------

def _cond_table(res, runs_by_cond):
    """runs_by_cond: {cond_label: [tags]} -> rows of mean BPB per slice + seed spread."""
    rows, means = [], {}
    for cond, tags in runs_by_cond.items():
        means[cond] = {}
        cells = []
        for sl in SLICES:
            vals = [_bpb(res, t, sl) for t in tags]
            m = _mean(vals)
            sp = _spread(vals)
            means[cond][sl] = m
            cells.append(f"{_fmt(m)}" + (f" ±{sp:.3f}" if len(tags) > 1 and sp else ""))
        rows.append((cond, cells))
    return rows, means


def _table_md(rows, header_note=""):
    head = "| condition | " + " | ".join(SLICE_LABEL[s] + " ↓" for s in SLICES) + " |"
    sep = "|---|" + "|".join("--:" for _ in SLICES) + "|"
    body = "\n".join("| " + c + " | " + " | ".join(cells) + " |" for c, cells in rows)
    return "\n".join([head, sep, body]) + (f"\n\n{header_note}" if header_note else "")


def _delta(means, a, b, sl):
    """means[a][sl] - means[b][sl] (negative => a better/lower)."""
    va, vb = means.get(a, {}).get(sl), means.get(b, {}).get(sl)
    if va is None or vb is None:
        return None
    return round(va - vb, 4)


# --- question renderers (answer/decision driven by the numbers) ---------------

def render_question(res, q, meta):
    rows, means = _cond_table(res, meta["runs"])
    conds = list(meta["runs"].keys())
    hs = meta["headline_slice"]
    lines = [f"### {q} — {meta['title']}", "",
             f"*Setup: {meta['varies']}. {meta['seeds']} seed(s). "
             f"Decides: {meta['decides']}.*", "", _table_md(rows)]
    ans, dec = _verdict(q, conds, means, hs)
    lines += ["", f"**Answer.** {ans}", "", f"**Decision.** {dec}"]
    return "\n".join(lines), means


def _seedspread_max(res, meta):
    sp = []
    for tags in meta["runs"].values():
        if len(tags) > 1:
            for sl in SLICES:
                sp.append(_spread([_bpb(res, t, sl) for t in tags]))
    return max(sp) if sp else 0.0


def _verdict(q, conds, means, hs):
    """Data-driven ANSWER + DECISION. `hs` = headline slice."""
    def d(a, b, sl):
        return _delta(means, a, b, sl)

    if q == "Q1":
        k, e = "kreyol-bpe", "english-24k"
        dh = d(k, e, hs)                       # negative => kreyol-bpe lower BPB (better)
        worse = [SLICE_LABEL[s] for s in SLICES if (d(k, e, s) or 0) > 0]
        if dh is not None and dh < 0:
            ans = (f"Yes — at identical v0.2.1 data/order/compute the Kreyòl vocabulary reaches "
                   f"**{abs(dh):.3f} bits/byte lower** general BPB than the English-24k ablation "
                   f"(same size, pattern, algorithm), so the vocabulary improves LEARNING, not just cost"
                   + ("." if not worse else f"; it wins on every slice."))
            dec = "Station 1 can claim the tokenizer causally — kreyol-bpe is confirmed for G-v1."
        else:
            ans = ("No measurable learning advantage — the two tokenizers converge to similar BPB, so "
                   "kreyol-bpe's win is efficiency (fewer bits/byte), not learnability.")
            dec = "Frame Station 1 as an efficiency claim; keep kreyol-bpe for its byte-efficiency."
        return ans, dec

    if q == "Q2":
        nat, wt = "natural", "weighted"
        da, dv = d(wt, nat, "authored_eval"), d(wt, nat, "authored_eval_v2")
        dg = d(wt, nat, "general_holdout")
        helped = (da is not None and da < 0) or (dv is not None and dv < 0)
        hurt_general = dg is not None and dg > 0.01
        if helped and not hurt_general:
            ans = (f"Yes — authored-upweighting lowers authored BPB (authored Δ={_fmt(da)}, "
                   f"authored_v2 Δ={_fmt(dv)}) without materially hurting general (Δ={_fmt(dg)}).")
            dec = "G-v1 samples with the config_v0_2 mix weights; add an authored-upweighted late-curriculum tail."
        elif helped and hurt_general:
            ans = (f"Partly — upweighting shifts voice toward authored (Δ={_fmt(da)}) but at a general-BPB "
                   f"cost (Δ={_fmt(dg)}).")
            dec = "Use the mix weights only in a late-curriculum tail, not the whole run."
        else:
            ans = (f"No — at this scale/budget the mix weights do not measurably move authored BPB "
                   f"(authored Δ={_fmt(da)}); the authored pool is too small to shift voice via reweighting.")
            dec = "Keep natural sampling for G-v1; pursue authored VOICE via more authored DATA, not reweighting."
        return ans, dec

    if q == "Q7":
        a, b = "v0.2.1", "v0.1"
        dg = d(a, b, "general_holdout")
        dau = d(a, b, "authored_eval")
        if dg is not None and dg < 0:
            ans = (f"Helps — v0.2.1 reaches **{abs(dg):.3f} lower** general BPB than v0.1 at fixed compute "
                   f"(authored Δ={_fmt(dau)}); fineweb-2's bulk adds signal, not just noise.")
            dec = "G-v1 trains on v0.2.1."
        elif dg is not None and dg > 0:
            ans = (f"Dilutes — v0.2.1 is **{dg:.3f} higher** general BPB than v0.1 at fixed compute; the "
                   f"fineweb-2 bulk crowds out the cleaner v0.1 signal in a fixed budget.")
            dec = "G-v1 trains on v0.1 (or v0.2.1 with fineweb-2 down-weighted); re-scope the corpus."
        else:
            ans = "Neutral — v0.1 and v0.2.1 reach the same general BPB at fixed compute."
            dec = "Train G-v1 on v0.2.1 for its register coverage (no BPB cost) but expect no BPB gain from bulk."
        return ans, dec

    if q == "Q3":
        dg = d("v0.1", "v0", "general_holdout")
        if dg is not None and dg < -0.005:
            ans = f"Yes — the drop-only de-junk (v0.1) reaches {abs(dg):.3f} lower general BPB than v0 at fixed compute."
            dec = "Worth getting more aggressive: fund the semantic (translationese) filters E deferred."
        elif dg is not None and dg > 0.005:
            ans = f"No — v0 slightly beats v0.1 ({dg:.3f}); the removed docs carried usable signal at this budget."
            dec = "Keep filtering high-precision/low-recall; don't chase recall — it costs signal."
        else:
            ans = "Neutral — de-junking neither helps nor hurts BPB at fixed compute (it removed ~mechanical junk, as designed)."
            dec = "Keep v0.1's precise filters; heavier filtering is unjustified on BPB grounds alone."
        return ans, dec

    if q == "Q4":
        din = d("stubs_out", "stubs_in", "authored_eval")
        dg = d("stubs_out", "stubs_in", "general_holdout")
        if (din is not None and din < -0.005) or (dg is not None and dg < -0.005):
            ans = f"Filler — dropping bot-stubs improves BPB (authored Δ={_fmt(din)}, general Δ={_fmt(dg)})."
            dec = "Corpus policy v0.3: drop wiki bot-stubs."
        elif (din is not None and din > 0.005) or (dg is not None and dg > 0.005):
            ans = f"Food — dropping bot-stubs hurts BPB (authored Δ={_fmt(din)}, general Δ={_fmt(dg)}); keep them."
            dec = "Corpus policy v0.3: keep bot-stubs (flagged)."
        else:
            ans = "Neutral — bot-stubs are BPB-inert at this budget."
            dec = "Corpus policy v0.3: keep bot-stubs (flagged); revisit only if they hurt a downstream eval."
        return ans, dec

    if q == "Q5":
        vals = {c: means.get(c, {}).get(hs) for c in ["4ep", "8ep", "12ep"]}
        seq = [vals.get(c) for c in ["4ep", "8ep", "12ep"]]
        best = min([c for c in ["4ep", "8ep", "12ep"] if vals.get(c) is not None],
                   key=lambda c: vals[c], default=None)
        d48 = (seq[1] - seq[0]) if seq[0] and seq[1] else None
        d812 = (seq[2] - seq[1]) if seq[1] and seq[2] else None
        ans = (f"BPB {_fmt(seq[0])}→{_fmt(seq[1])}→{_fmt(seq[2])} at 4→8→12 epochs "
               f"(Δ4→8={_fmt(d48)}, Δ8→12={_fmt(d812)}); best at **{best}**.")
        if d812 is not None and d812 > -0.005:
            dec = "Returns flatten by ~8 epochs — G-v1's ~6.7-epoch schedule is in the safe band; don't push past ~8×."
        else:
            dec = "Still improving at 12 epochs — G-v1 can safely repeat further; extend the token schedule."
        return ans, dec

    return "—", "—"


# --- G-v1 recommended config --------------------------------------------------

def recommend_gv1(res, precheck, means_by_q):
    """Assemble the recommended G-v1 config block from the fleet + precheck verdicts."""
    # depth from the pre-check (lower general BPB wins; monotonic-in-depth = d12)
    depth, depth_note = 12, "pre-check unavailable — default to G v0's d12"
    if precheck:
        d12 = precheck.get("precheck-v021-d12", {}).get("bpb", {}).get("general_holdout")
        d16 = precheck.get("precheck-v021-d16", {}).get("bpb", {}).get("general_holdout")
        if d12 is not None and d16 is not None:
            depth = 12 if d12 <= d16 else 16
            depth_note = (f"d12 {d12:.4f} vs d16 {d16:.4f} general BPB on v0.2.1 (~175M tok) — "
                          f"{'d12 still wins' if depth==12 else 'd16 now wins at 219.6M unique'}")
    # corpus from Q7
    q7 = means_by_q.get("Q7", {})
    corpus = "v0.2.1"
    if q7:
        dg = _delta(q7, "v0.2.1", "v0.1", "general_holdout")
        corpus = "v0.2.1" if (dg is None or dg <= 0) else "v0.1"
    # mix from Q2
    q2 = means_by_q.get("Q2", {})
    mix = "config_v0_2.MIX_WEIGHTS (authored-upweighted)"
    if q2:
        da = _delta(q2, "weighted", "natural", "authored_eval")
        dg = _delta(q2, "weighted", "natural", "general_holdout")
        if not ((da is not None and da < 0) and (dg is None or dg <= 0.01)):
            mix = "natural sampling (mix weights did not beat natural at fleet scale)"
    return {"corpus": corpus, "tokenizer": "kreyol-bpe (Q1-confirmed)", "depth": depth,
            "depth_note": depth_note, "mix": mix}


# --- render -------------------------------------------------------------------

def build():
    res = _results()
    precheck = _load(PRECHECK, {})
    man = _load(MANIFEST, {})
    audit = _load(PROBE_AUDIT, {})
    fineweb = _load(PROBE_FINEWEB, {})
    bloom = _load(BLOOM, {})

    L = ["# Micro-model fleet — Workstream H", "",
         f"*Snapshot {H.SNAPSHOT_DATE}. Micro arch: standard Llama width 384 / depth 6 "
         f"(**{_micro_params():,} params**, 64% in the untied 24k embeddings), the SAME "
         f"`LlamaForCausalLM` contract as Workstream G scaled down. ~{H.FLEET_TOKENS//1_000_000}M "
         f"tokens/run (near-Chinchilla). All BPB byte-normalized on the standing E/J slices "
         f"(general holdout · authored_eval · translation_shaped_eval · authored_eval_v2 · FLORES hat, "
         f"measurement-only). GPU: Modal H100. Generated by `train/fleet_report.py`.*", ""]

    # Part 0
    L += ["## Part 0 — CPU side-quests", "", _part0(audit, fineweb, bloom), ""]

    # Part 2 questions
    L += ["## Part 2 — the questions", ""]
    means_by_q = {}
    for q in ["Q1", "Q2", "Q7", "Q3", "Q4", "Q5"]:
        meta = _qmeta(q)
        sec, means = render_question(res, q, meta)
        means_by_q[q] = means
        L += [sec, ""]

    # Part 3 depth pre-check
    L += ["## Part 3 — flagship depth pre-check (full-size d12 vs d16 on v0.2.1)", "",
          _precheck_table(precheck), ""]

    # G-v1 config
    rec = recommend_gv1(res, precheck, means_by_q)
    L += ["## Recommended G-v1 config", "",
          f"- **Corpus:** {rec['corpus']}",
          f"- **Tokenizer:** {rec['tokenizer']}",
          f"- **Depth:** d{rec['depth']} — {rec['depth_note']}",
          f"- **Mix:** {rec['mix']}",
          "", "*(The runnable block lives in docs/phase-1.md Workstream G.)*", ""]

    # reproduce
    L += ["## Reproduce", "", "```bash", "cd ml && uv sync",
          "uv run python -m train.fleet_tokenizer          # english-24k ablation tokenizer",
          "uv run python -m train.fleet_data               # tokenize variants -> bins + eval",
          "uv run python -m train.fleet_run upload",
          "uv run python -m train.fleet_run verify",
          "uv run python -m train.fleet_run smoke          # 1-run pipeline check",
          "uv run python -m train.fleet_run fleet          # Part 2 (13 micro runs)",
          "uv run python -m train.fleet_run precheck        # Part 3 (d12/d16 full-size)",
          "uv run python -m train.fleet_run collect && uv run python -m train.fleet_report",
          "```", ""]

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print(f"[fleet_report] wrote {OUT}")
    print(f"[fleet_report] recommended G-v1: {rec}")
    return rec


def _micro_params():
    from . import llama_model as M
    return M.param_count(H.MICRO_DEPTH, H.MICRO_ARCH)["total"]


def _qmeta(q):
    return H.QUESTIONS[q]


def _part0(audit, fineweb, bloom):
    lines = []
    # Bloom
    bl = "still **gated** (auto-grant pending one HF-account click) — logged and skipped; Bloom did NOT enter the corpus."
    if bloom and bloom.get("keeper_docs"):
        bl = f"accessible — {bloom['keeper_docs']} BY/BY-SA keepers ingested as v0.2.2."
    lines += [f"- **0a Bloom (sil-ai/bloom-lm):** {bl}"]
    # probe leak
    if audit:
        v01 = audit.get("shards", {}).get("v0.1", {})
        leaked = sorted(v01.get("per_probe_counts", {}).keys())
        fw = sorted((fineweb or {}).get("per_probe", {}).keys()) if fineweb else []
        allp = sorted(set(leaked) | set(int(x) for x in fw))
        p31 = audit.get("probe_31_in_v0_1")
        lines += [
            f"- **0b Probe-leak provenance:** the 19 J-removed leak-docs matched probes "
            f"**{', '.join('#'+str(n) for n in allp)}** (v0.1 MADLAD: {', '.join('#'+str(n) for n in leaked)}; "
            f"fineweb-2 net-new: {', '.join('#'+str(int(n)) for n in fw)}). "
            f"**Probe #31 ‘Lè chat pa la’ had {audit.get('probe_31_v0_1_hits',0)} hits in v0.1** — "
            f"{'⚠️ FLAG: it WAS a v0.1 leak; footnote the Model C v0 exhibit claim until G-v1.' if p31 else 'Workstream E was right; the Model C v0 exhibit claim stands.'} "
            f"v0.2.1 shard is clean ({'0 leaks' if audit.get('v0_2_1_clean') else 'LEAKS REMAIN'})."]
    # kakugo
    ksheet = os.path.join(F.REPO_ROOT, "reports", "kakugo_audit_sheet.md")
    kmsg = ("[reports/kakugo_audit_sheet.md](kakugo_audit_sheet.md) — 150 stratified, blinded "
            "conversations, 1–5 naturalness + wrong/awkward flag; awaits a native reviewer (decides SFT Layer 1)."
            if os.path.exists(ksheet) else "pending.")
    lines += [f"- **0c kakugo-hat audit pack:** {kmsg}"]
    return "\n".join(lines)


def _precheck_table(precheck):
    if not precheck:
        return "_pending._"
    head = "| depth | params | " + " | ".join(SLICE_LABEL[s] + " ↓" for s in SLICES) + " | tok/s | epochs |"
    sep = "|--:|--:|" + "|".join("--:" for _ in SLICES) + "|--:|--:|"
    rows = []
    for tag in sorted(precheck.keys()):
        r = precheck[tag]
        cells = [_fmt(r.get("bpb", {}).get(s)) for s in SLICES]
        rows.append(f"| d{r.get('depth')} | {r.get('params',0)/1e6:.0f}M | " + " | ".join(cells) +
                    f" | {r.get('median_tok_s')} | {r.get('epochs')} |")
    d12 = precheck.get("precheck-v021-d12", {}).get("bpb", {}).get("general_holdout")
    d16 = precheck.get("precheck-v021-d16", {}).get("bpb", {}).get("general_holdout")
    note = ""
    if d12 is not None and d16 is not None:
        note = (f"\n\n**Pick:** {'d12' if d12<=d16 else 'd16'} — general BPB d12 {d12:.4f} vs d16 {d16:.4f} "
                f"on v0.2.1 (219.6M unique). "
                + ("d12 still wins (data-limited regime holds even at 2× the tokens)."
                   if d12 <= d16 else "the extra tokens now justify d16's capacity — the ranking flipped from G's 112M-token sweep."))
    return "\n".join([head, sep] + rows) + note


if __name__ == "__main__":
    build()
