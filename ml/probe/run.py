"""Workstream D orchestrator — staged funnel over Modal, then score + report.

Stages (cost control, run in order):
  smoke   — 20 dev sentences/model, MT both directions. Catch template/decoding
            breakage before the expensive run. Loads are cheap (weights cached).
  main    — full battery on the fixed 250-item dev subset for every model that
            loads: BPB (full + authored-only holdout), MT (both dirs), proverb
            completion, naturalness completions.
  fulldev — MT both directions on the FULL dev set for the top-2 by the automated
            scorecard (authored BPB primary).
  report  — (re)build reports from saved raw results; no GPU.

Raw results (which embed FLORES prompts/refs) are written under the git-ignored
ml/data/probe/; only the reports (scores + model-generated text) are committed.

Usage:
  uv run python -m probe.run --stage all       # smoke -> main -> fulldev -> report
  uv run python -m probe.run --stage smoke
  uv run python -m probe.run --stage main
  uv run python -m probe.run --stage fulldev
  uv run python -m probe.run --report          # rebuild report from saved JSON
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import config, data, measures

RAW_DIR = os.path.join(data.TD.config.DATA, "probe")  # ml/data/probe (git-ignored)


def log(m):
    print(m, file=sys.stderr, flush=True)


def _save(name, obj):
    os.makedirs(RAW_DIR, exist_ok=True)
    with open(os.path.join(RAW_DIR, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


def _load_raw(name):
    p = os.path.join(RAW_DIR, name)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


# --- local data assembly ------------------------------------------------------

def build_local(token):
    flores = data.load_flores_dev(token)
    sel = data.select_ids(flores["ids"])
    slices = data.holdout_slices()
    proverbs = data.load_probe_proverbs()
    log(f"  dev ids: {len(flores['ids'])}; exemplars={sel['exemplars']}; "
        f"eval subset={len(sel['subset'])}; full-dev={len(sel['full_dev'])}")
    return {"flores": flores, "sel": sel, "slices": slices,
            "proverbs": proverbs,
            "prov_prompts": data.proverb_prompts(proverbs),
            "nat_prompts": data.naturalness_prompts()}


# --- Modal driver -------------------------------------------------------------

def warm(token):
    """CPU prefetch of all candidate weights -> accessibility map (skips gated
    repos before any GPU spend)."""
    from . import modal_app
    repos = [c["repo"] for c in config.CANDIDATES]
    with modal_app.app.run():
        pf = modal_app.prefetch.remote(repos, token)
    for repo, r in pf.items():
        log(f"  prefetch {repo}: " + ("cached" if r.get("ok")
                                       else f"SKIP {r.get('error')}"))
    _save("results_prefetch.json", pf)
    return {repo for repo, r in pf.items() if r.get("ok")}


def _run_specs(specs, token, accessible=None):
    """Run a list of (key, spec) sequentially on Modal; return {key: raw}.
    Repos not in `accessible` (gated) are skipped without GPU spend."""
    from . import modal_app
    out = {}
    with modal_app.app.run():
        for key, spec in specs:
            if accessible is not None and spec["repo"] not in accessible:
                out[key] = {"repo": spec["repo"], "ok": False,
                            "error": "gated/unavailable (prefetch failed)"}
                log(f"  [modal] {key}: {spec['repo']} SKIP (not accessible)")
                continue
            log(f"  [modal] {key}: {spec['repo']} ...")
            try:
                r = modal_app.run_model.remote(spec, token)
            except Exception as e:
                r = {"repo": spec["repo"], "ok": False,
                     "error": f"runtime {type(e).__name__}: {e}"}
            if r.get("ok"):
                log(f"    ok rev={r['revision'][:12]} arch={r.get('arch')} "
                    f"load={r.get('load_s')}s total={r.get('total_s')}s")
            else:
                log(f"    SKIP: {r.get('error')}")
            out[key] = r
    return out


def stage_smoke(L, token, accessible):
    specs = []
    for c in config.CANDIDATES:
        mt = data.mt_prompts(L["flores"], L["sel"]["exemplars"], L["sel"]["smoke"])
        specs.append((c["key"], {"repo": c["repo"], "tasks": {"mt": {
            "eng2hat": mt["eng2hat"]["prompts"],
            "hat2eng": mt["hat2eng"]["prompts"]}}}))
    res = _run_specs(specs, token, accessible)
    _save("results_smoke.json", res)
    _smoke_gate(res)
    return res


def _smoke_gate(res):
    loaded = {k: r for k, r in res.items() if r.get("ok")}
    if not loaded:
        raise SystemExit("smoke: NO candidate loaded — cannot proceed (check gates).")
    any_output = False
    log("  --- smoke previews (eng->hat, first 2) ---")
    for k, r in loaded.items():
        outs = r.get("mt", {}).get("eng2hat", [])
        nonempty = sum(1 for o in outs if o.strip())
        any_output = any_output or nonempty > 0
        log(f"    {k}: {nonempty}/{len(outs)} non-empty; e.g. {outs[:2]}")
    if not any_output:
        raise SystemExit("smoke: every loaded model returned empty MT — likely a "
                         "template/decoding bug in the harness. Stopping.")
    log("  smoke gate PASSED.")


def stage_main(L, token, smoke, accessible):
    ok = [c for c in config.CANDIDATES if smoke.get(c["key"], {}).get("ok")]
    specs = []
    for c in ok:
        mt = data.mt_prompts(L["flores"], L["sel"]["exemplars"], L["sel"]["subset"])
        specs.append((c["key"], {"repo": c["repo"], "tasks": {
            "bpb": {"full": L["slices"]["full"]["texts"],
                    "authored": L["slices"]["authored"]["texts"]},
            "mt": {"eng2hat": mt["eng2hat"]["prompts"],
                   "hat2eng": mt["hat2eng"]["prompts"]},
            "proverbs": L["prov_prompts"],
            "naturalness": L["nat_prompts"]}}))
    res = _run_specs(specs, token, accessible)
    # attach refs so scoring is reproducible from the raw file
    mt = data.mt_prompts(L["flores"], L["sel"]["exemplars"], L["sel"]["subset"])
    res["_refs"] = {"eng2hat": mt["eng2hat"]["refs"], "hat2eng": mt["hat2eng"]["refs"]}
    res["_slices_meta"] = {s: {k: v for k, v in L["slices"][s].items() if k != "texts"}
                           for s in L["slices"]}
    _save("results_main.json", res)
    return res


def _fulldev_ids(L, top_keys):
    """Full dev (992) when both finalists are fast; a seeded cap when a slow
    (multimodal) model is present, scored identically for both."""
    full = L["sel"]["full_dev"]
    if any(k in config.SLOW_MODELS for k in top_keys):
        return full[:config.FULLDEV_CAP_SLOW], True
    return full, False


def stage_fulldev(L, token, top_keys, accessible):
    eval_ids, capped = _fulldev_ids(L, top_keys)
    mt = data.mt_prompts(L["flores"], L["sel"]["exemplars"], eval_ids)
    specs = []
    for key in top_keys:
        c = next(c for c in config.CANDIDATES if c["key"] == key)
        specs.append((key, {"repo": c["repo"], "tasks": {"mt": {
            "eng2hat": mt["eng2hat"]["prompts"],
            "hat2eng": mt["hat2eng"]["prompts"]}}}))
    log(f"  full-dev on {len(eval_ids)} sentences" +
        (" (capped for cost — slow multimodal model in top-2)" if capped else ""))
    res = _run_specs(specs, token, accessible)
    res["_refs"] = {"eng2hat": mt["eng2hat"]["refs"], "hat2eng": mt["hat2eng"]["refs"]}
    res["_n"] = len(eval_ids)
    res["_capped"] = capped
    _save("results_fulldev.json", res)
    return res


# --- scoring / scorecard ------------------------------------------------------

def score_main(main, proverbs):
    refs = main["_refs"]
    cards = {}
    for c in config.CANDIDATES:
        r = main.get(c["key"])
        if not r or not r.get("ok"):
            cards[c["key"]] = {"ok": False, "repo": c["repo"],
                               "error": (r or {}).get("error", "not run"),
                               "control": c.get("control", False)}
            continue
        card = {"ok": True, "repo": c["repo"], "revision": r["revision"],
                "arch": r.get("arch"), "control": c.get("control", False),
                "load_s": r.get("load_s"), "total_s": r.get("total_s"),
                "transformers": r.get("transformers")}
        card["bpb_full"] = r["bpb"]["full"]["bpb"]
        card["bpb_authored"] = r["bpb"]["authored"]["bpb"]
        card["mt_e2h"] = measures.score_mt(r["mt"]["eng2hat"], refs["eng2hat"])
        card["mt_h2e"] = measures.score_mt(r["mt"]["hat2eng"], refs["hat2eng"])
        card["proverbs"] = measures.score_proverbs(proverbs, r["proverbs"])
        card["naturalness_raw"] = r["naturalness"]
        cards[c["key"]] = card
    return cards


def pick_top(cards, k):
    ok = [(key, c) for key, c in cards.items() if c.get("ok")]
    # primary = authored BPB (lower better); tie-break chrF2++ eng->hat (higher)
    ok.sort(key=lambda kc: (kc[1]["bpb_authored"], -kc[1]["mt_e2h"]["chrf2pp"]))
    return [key for key, _ in ok[:k]]


def _est_cost(*raws):
    secs = 0.0
    for raw in raws:
        if not raw:
            continue
        for k, r in raw.items():
            if isinstance(r, dict) and r.get("ok"):
                secs += r.get("total_s", 0.0)
    return secs, secs / 3600.0 * config.MODAL_L40S_USD_PER_HR


# --- orchestration ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["all", "smoke", "main", "fulldev"],
                    default="all")
    ap.add_argument("--report", action="store_true",
                    help="rebuild reports from saved raw results (no GPU)")
    args = ap.parse_args()

    token = data.load_env()
    if not token:
        raise SystemExit("HF_TOKEN not found in .env")

    from . import report as R

    if args.report:
        _rebuild_report(R)
        return

    L = build_local(token)
    accessible = warm(token)

    smoke = _load_raw("results_smoke.json")
    if args.stage in ("all", "smoke"):
        smoke = stage_smoke(L, token, accessible)
    if args.stage == "smoke":
        return

    main_res = _load_raw("results_main.json")
    if args.stage in ("all", "main"):
        main_res = stage_main(L, token, smoke or {c["key"]: {"ok": True}
                                                  for c in config.CANDIDATES},
                              accessible)
    if args.stage == "main":
        _rebuild_report(R)
        return

    cards = score_main(main_res, L["proverbs"])
    top = pick_top(cards, config.TOP_K)
    log(f"  top-{config.TOP_K} for full dev: {top}")
    fulldev = _load_raw("results_fulldev.json")
    if args.stage in ("all", "fulldev"):
        fulldev = stage_fulldev(L, token, top, accessible)

    _rebuild_report(R)


def _rebuild_report(R):
    main_res = _load_raw("results_main.json")
    if not main_res:
        raise SystemExit("no results_main.json — run --stage main first")
    fulldev = _load_raw("results_fulldev.json")
    # proverbs are needed to score; rebuild them locally (no GPU / no FLORES)
    proverbs = data.load_probe_proverbs()
    cards = score_main(main_res, proverbs)
    top = pick_top(cards, config.TOP_K)
    fulldev_scores = None
    if fulldev:
        fulldev_scores = {}
        for key, r in fulldev.items():
            if key.startswith("_") or not isinstance(r, dict) or not r.get("ok"):
                continue
            fulldev_scores[key] = {
                "mt_e2h": measures.score_mt(r["mt"]["eng2hat"], fulldev["_refs"]["eng2hat"]),
                "mt_h2e": measures.score_mt(r["mt"]["hat2eng"], fulldev["_refs"]["hat2eng"]),
                "n": fulldev.get("_n"),
            }
    secs, usd = _est_cost(_load_raw("results_smoke.json"), main_res, fulldev)
    R.write_all(cards, top, fulldev_scores, bool(fulldev and fulldev.get("_capped")),
                main_res.get("_slices_meta", {}),
                {"gpu_seconds": round(secs, 1), "est_usd": round(usd, 2)})
    log(f"  reports written. GPU ~{secs/60:.1f} min, est ${usd:.2f}")


if __name__ == "__main__":
    main()
