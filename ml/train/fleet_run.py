"""Workstream H orchestrator — prepare data → upload → verify → smoke → fleet → precheck.

Subcommands:
  prepare    local: tokenize corpus variants -> bins + eval_texts + manifest
  upload     push bins / eval / tokenizers to the Modal Volume
  verify     assert the micro param count (== the torch-free calc)
  smoke      train ONE micro run end-to-end (validates pipeline; measures tok/s) before
             committing GPU $ to the full fleet
  fleet      Part 2: train all 13 micro runs (self-persisting, resumable) -> results
  precheck   Part 3: full-size d12/d16 depth pre-check on v0.2.1
  collect    pull every self-persisted result JSON off the Volume -> local results file

Run:
  cd ml && uv run python -m train.fleet_data                 # prepare (local)
  cd ml && uv run python -m train.fleet_run upload
  cd ml && uv run python -m train.fleet_run verify
  cd ml && uv run python -m train.fleet_run smoke
  cd ml && uv run python -m train.fleet_run fleet
  cd ml && uv run python -m train.fleet_run precheck
  cd ml && uv run python -m train.fleet_run collect
"""

from __future__ import annotations

import argparse
import json
import os

import modal

from . import config as F
from . import fleet_config as H
from .fleet_app import (app, fleet_train_batch, precheck_depth, read_result,
                        verify_micro_params)

VOL = modal.Volume.from_name(F.MODAL_VOLUME, create_if_missing=True)


def _save(name, obj):
    os.makedirs(H.FLEET_WORK, exist_ok=True)
    p = os.path.join(H.FLEET_WORK, name)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f"[fleet_run] wrote {p}")


def _manifest() -> dict:
    p = os.path.join(H.FLEET_WORK, "fleet_data_manifest.json")
    assert os.path.exists(p), "run `python -m train.fleet_data` first (prepare)"
    return json.load(open(p))


def _run_list(only=None) -> list:
    """Resolve H.RUNS into concrete run dicts (Q5 steps from the manifest)."""
    man = _manifest()
    q5_steps = man.get("q5_epoch_steps", {})
    runs = []
    for tag, r in H.RUNS.items():
        if only and tag not in only:
            continue
        variant = r["variant"]
        num_iter = r["num_iterations"] or q5_steps.get(tag)
        assert num_iter, f"{tag}: no step count (Q5 needs the manifest)"
        runs.append({
            "tag": tag, "variant": variant, "bin": f"{variant}.bin",
            "tok": H.VARIANTS[variant]["tok"], "seed": H.SEEDS[r["seed"]],
            "num_iterations": int(num_iter),
            "arch": H.MICRO_ARCH, "depth": H.MICRO_DEPTH,
        })
    return runs


def do_verify():
    with modal.enable_output(), app.run():
        res = verify_micro_params.remote()
    print(json.dumps({k: v for k, v in res.items() if k != "arch"}, indent=2))
    assert res["match"], f"micro param mismatch: real={res['real']} calc={res['calc']}"
    print(f"[verify] micro params match: {res['real']:,} (d{res['depth']}) ✅")


def do_smoke():
    runs = _run_list(only=["v021_kbpe.sA"])
    # a cheap smoke: 24 steps only (validate end-to-end + measure tok/s / compile)
    runs[0] = {**runs[0], "num_iterations": 24, "tag": "smoke_v021_kbpe"}
    with modal.enable_output(), app.run():
        res = fleet_train_batch.remote(runs, force=True)
    _save("fleet_smoke.json", res)
    r = res["results"]["smoke_v021_kbpe"]
    print(f"[smoke] params={r['params']:,} tok/s={r['median_tok_s']} "
          f"wall={r['wall_seconds']}s bpb={r['bpb']}")


def do_fleet(only=None):
    runs = _run_list(only=only)
    print(f"[fleet] launching {len(runs)} runs: {[r['tag'] for r in runs]}")
    with modal.enable_output(), app.run(detach=True):
        res = fleet_train_batch.remote(runs)
    _save("fleet_results.json", res)
    for tag, r in res["results"].items():
        print(f"  {tag}: bpb={r.get('bpb')} tok/s={r.get('median_tok_s')} epochs={r.get('epochs')}")


def do_precheck():
    P = H.DEPTH_PRECHECK
    bin_name = f"{P['corpus_variant']}.bin"
    results = {}
    with modal.enable_output(), app.run(detach=True):
        for depth in P["depths"]:
            tag = P["model_tag"].format(depth=depth)
            r = precheck_depth.remote(depth, bin_name, P["num_iterations"], P["seed"], tag)
            results[tag] = r
            print(f"[precheck] d{depth}: params={r['params']:,} bpb={r['bpb']} "
                  f"tok/s={r['median_tok_s']} epochs={r['epochs']}")
    _save("fleet_precheck.json", results)


def do_collect():
    """Pull every self-persisted result off the Volume (disconnect-proof collection)."""
    tags = [f"fleet_{t}.json" for t in H.RUNS] + \
           [f"precheck_{H.DEPTH_PRECHECK['model_tag'].format(depth=d)}.json"
            for d in H.DEPTH_PRECHECK["depths"]]
    out = {"fleet": {}, "precheck": {}}
    with modal.enable_output(), app.run():
        for name in tags:
            r = read_result.remote(name)
            if not r:
                print(f"[collect] MISSING {name}")
                continue
            if name.startswith("precheck_"):
                out["precheck"][r.get("tag", name)] = r
            else:
                out["fleet"][r["tag"]] = r
    _save("fleet_results.json", {"results": out["fleet"]})
    _save("fleet_precheck.json", out["precheck"])
    print(f"[collect] {len(out['fleet'])} fleet + {len(out['precheck'])} precheck results")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("upload")
    sub.add_parser("verify")
    sub.add_parser("smoke")
    fl = sub.add_parser("fleet"); fl.add_argument("--only", type=str, default="")
    sub.add_parser("precheck")
    sub.add_parser("collect")
    args = ap.parse_args()

    if args.cmd == "upload":
        from . import fleet_data
        fleet_data.do_upload()
    elif args.cmd == "verify":
        do_verify()
    elif args.cmd == "smoke":
        do_smoke()
    elif args.cmd == "fleet":
        only = [x for x in args.only.split(",") if x.strip()] or None
        do_fleet(only)
    elif args.cmd == "precheck":
        do_precheck()
    elif args.cmd == "collect":
        do_collect()


if __name__ == "__main__":
    main()
