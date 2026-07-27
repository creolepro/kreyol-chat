"""Workstream G orchestrator — prepare data → upload → verify → {gate, sweep, flagship}.

Subcommands (each writes a results JSON under data/train_work/g/ that the reports read):

  prepare   local: tokenize corpus v0.1 → bins, build parity probe + tokenizer bundle
  upload    push bins / tokenizer / parity probe to the Modal Volume (G layout)
  verify    assert param counts (d12/d16/d20) == the torch-free calc
  gate      Part 2: train d16 throwaway → run F2 gates 1-6 → g_gates_results.json
  sweep     Part 3: train d12/d16/d20 (identical order+budget) → BPB → g_sweep_results.json
  flagship  Part 4: train chosen depth w/ per-ckpt gens+BPB → convert → g_flagship[_v1]_results.json
            (--version v1 = fleet-informed FLAGSHIP_V1: corpus v0.2.1, ~1.0B tokens, distinct tag)
  base-bpb  base-model BPB on the SAME eval slices → g_base_bpb.json (vs-scorecard table)
  scorecard Model C v1's 5-row table: re-score v0 + the 3 bases on the CURRENT (v1) eval set

Run:
  cd ml && uv run python -m train.tokenize_g [--sample] [--v1]     # (prepare, local)
  cd ml && uv run python -m train.g_run gate
  cd ml && uv run python -m train.g_run sweep
  cd ml && uv run python -m train.g_run flagship --depth 16
  cd ml && uv run python -m train.g_run flagship --depth 12 --num-iter 1907 --version v1
  cd ml && uv run python -m train.g_run base-bpb
  cd ml && uv run python -m train.g_run scorecard
"""

from __future__ import annotations

import argparse
import json
import os

import modal

from . import config as F
from . import llama_config as G
from . import prepare as Fprep
from .llama_app import (app, verify_params, train, generate, bpb,
                        convert_gates, base_bpb, read_result, exhibit_gen, gguf_meta,
                        reconvert_bos_and_check)

VOL = modal.Volume.from_name(F.MODAL_VOLUME, create_if_missing=True)


def _out(name):
    os.makedirs(G.G_WORK, exist_ok=True)
    return os.path.join(G.G_WORK, name)


def _save(name, obj):
    with open(_out(name), "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f"[g_run] wrote {_out(name)}")


# --- upload -------------------------------------------------------------------

def do_upload():
    # tokenizer bundle: reuse F's builder to get tokenizer.pkl locally, ship it + tokenizer.json
    if not os.path.exists(os.path.join(F.BUNDLE_TOKENIZER, "tokenizer.pkl")):
        print("[upload] building tokenizer bundle (prepare)…")
        Fprep.build_tokenizer_bundle()
    assert os.path.isdir(G.G_BUNDLE_DATA), "run `python -m train.tokenize_g` first"
    print("[upload] pushing bins + tokenizer + parity probe to the Volume…")
    with VOL.batch_upload(force=True) as b:
        b.put_directory(G.G_BUNDLE_DATA, "/g/data")
        b.put_file(os.path.join(F.BUNDLE_TOKENIZER, "tokenizer.pkl"), "/g/tokenizer/tokenizer.pkl")
        b.put_file(F.KREYOL_BPE_HF_JSON, "/g/tokenizer/tokenizer.json")
        b.put_file(G.CHECKPOINT_PROMPTS, "/g/checkpoint_prompts.json")   # frozen exhibit prompts
    print("[upload] done")


# --- helpers to assemble a train cfg -----------------------------------------

def _ckpt_steps_for(num_iter, ckpt_tokens=None):
    steps = G.tokens_to_steps(ckpt_tokens or G.FLAGSHIP["checkpoint_tokens"])
    return sorted(s for s in set(steps) if 0 <= s <= num_iter)


def _train_cfg(depth, tag, num_iter, save_steps=None, resume_from=-1, ckpt_evals=False):
    return {"depth": depth, "model_tag": tag, "num_iterations": num_iter,
            "save_steps": save_steps or [], "resume_from_step": resume_from,
            "ckpt_evals": ckpt_evals}


# --- verify -------------------------------------------------------------------

def do_verify():
    with modal.enable_output(), app.run():
        res = verify_params.remote()
    print(json.dumps(res, indent=2))
    _save("g_verify_params.json", res)
    for d in G.DEPTHS:
        assert res[f"d{d}"]["match"], f"param mismatch at d{d}!"
    print("[verify] param counts match the torch-free calc for all depths ✅")


# --- Part 2: gate -------------------------------------------------------------

def do_gate(skip_train=False):
    S = G.GATE
    with modal.enable_output(), app.run():
        results = {"part": "F2-gates", "gate_config": S}
        if not skip_train:
            cfg = _train_cfg(S["depth"], S["model_tag"], S["num_iterations"])
            results["train"] = train.remote(cfg)
            print(f"[gate] trained {S['model_tag']} → steps {results['train']['checkpoint_steps']}")
        step = S["num_iterations"]
        results["gates"] = convert_gates.remote(S["model_tag"], step)
    _save("g_gates_results.json", results)
    _print_gate_summary(results["gates"])


def _print_gate_summary(g):
    print("\n=== F2 GATE SUMMARY ===")
    print(f"  gate1 logits lossless: {g['gate1_logits']['pass']} "
          f"(fp32 roundtrip Δ={g['gate1_logits']['max_abs_logit_diff_fp32_roundtrip']:.2e})")
    print(f"  gate2 GGUF convert rc={g['gate2_gguf']['final_rc']} "
          f"exists={g['gate2_gguf']['gguf_exists']} registered={g['gate2_gguf']['registered_kreyol_bpe']}")
    if "gate4_parity" in g:
        p = g["gate4_parity"]["probe"]
        print(f"  gate4 parity: HF {p['hf_sentence_frac']:.3f} / llama.cpp {p['llamacpp_sentence_frac']:.3f} "
              f"(fixtures_all_match={g['gate4_parity']['fixtures_all_match']})")
    if "gate3_stock" in g:
        s = g["gate3_stock"]
        print(f"  gate3 stock ollama: rejected_unknown_pretok={s.get('ollama_rejected_unknown_pretokenizer')} "
              f"loaded={s.get('ollama_loaded_and_generated')}")
    print(f"  gate5 onnx export_ok={g['gate5_onnx'].get('export_ok')} gen_ok={g['gate5_onnx'].get('onnx_gen_ok')}")
    if "gate6_cross_runtime" in g:
        print(f"  gate6 cross-runtime: {json.dumps(g['gate6_cross_runtime'].get('pre_quant', [])[:1])}")


# --- Part 3: sweep ------------------------------------------------------------

def do_sweep(depths=None):
    S = G.SWEEP
    depths = depths or S["depths"]
    # merge into any existing results (so a single-depth re-run doesn't drop the others)
    out = _out("g_sweep_results.json")
    results = json.load(open(out)) if os.path.exists(out) else {"part": "depth-sweep", "sweep_config": S, "runs": {}}
    with modal.enable_output(), app.run(detach=True):
        for depth in depths:
            tag = S["model_tag"].format(depth=depth)
            # inline full-slice BPB (final_full_bpb) so one self-persisting container does
            # train + BPB — disconnect-proof, no separate bpb.remote().
            cfg = _train_cfg(depth, tag, S["num_iterations"])
            cfg["ckpt_evals"] = True         # loads eval_texts so final_full_bpb has data
            cfg["final_full_bpb"] = True
            cfg["save_steps"] = []           # sweep needs only the final checkpoint's BPB
            cfg["device_batch_size"] = 16    # halve activation memory (d20 headroom)
            tr = train.remote(cfg)
            bp = tr["final_full_bpb"]
            results["runs"][f"d{depth}"] = {"train": tr, "bpb": bp}
            final_loss = tr["logs"][-1]["loss"] if tr["logs"] else None
            print(f"[sweep] d{depth}: median_tok_s={tr['median_tok_s']} final_loss={final_loss} "
                  f"bpb={ {k: round(v['bpb'],4) for k,v in bp.items()} }")
            _save("g_sweep_results.json", results)   # incremental save after each depth
    _save("g_sweep_results.json", results)


# --- Part 4: flagship ---------------------------------------------------------

def do_flagship(depth, num_iter=None, version="v0"):
    """Detach-based + Volume-resumable: train (with inline final full BPB) then convert. Each
    step self-persists to the Volume, so a client disconnect (ephemeral app killed when the
    local driver dies) never loses work — re-run to resume from whatever is already on the
    Volume. detach=True keeps the app running server-side past a client disconnect.

    version="v1" uses G.FLAGSHIP_V1 (corpus v0.2.1, ~1.0B tokens, 1B checkpoint, distinct tag
    `modelc-v1-d{depth}`) and writes g_flagship_v1_results.json so v0's results are untouched."""
    cfg_fl = G.FLAGSHIP_V1 if version == "v1" else G.FLAGSHIP
    num_iter = num_iter or cfg_fl["num_iterations"]
    tag = cfg_fl["model_tag"].format(depth=depth)
    save_steps = _ckpt_steps_for(num_iter, cfg_fl["checkpoint_tokens"])
    final = save_steps[-1]
    with modal.enable_output(), app.run(detach=True):
        tr = read_result.remote(f"train_{tag}.json")
        if not tr:
            cfg = _train_cfg(depth, tag, num_iter, save_steps=save_steps, ckpt_evals=True)
            cfg["final_full_bpb"] = True
            tr = train.remote(cfg)
            print(f"[flagship] trained {tag}: epochs={tr['epochs']} ckpts={tr['checkpoint_steps']}")
        else:
            print(f"[flagship] train result already on Volume (resuming)")
        gt = read_result.remote(f"gates_{tag}.json")
        if not gt:
            gt = convert_gates.remote(tag, final)
    results = {"part": "flagship", "version": version, "corpus": cfg_fl.get("corpus"),
               "depth": depth, "tag": tag, "num_iterations": num_iter,
               "checkpoint_steps": save_steps, "train": tr,
               "final_full_bpb": (tr or {}).get("final_full_bpb"), "gates": gt}
    _save("g_flagship_results.json" if version == "v0" else "g_flagship_v1_results.json", results)
    ffb = results.get("final_full_bpb") or {}
    if ffb:
        print(f"[flagship] final full BPB: { {k: round(v['bpb'],4) for k,v in ffb.items()} }")


# --- v1 five-row scorecard: re-score v0 + bases on the CURRENT eval set --------

def do_scorecard(v0_depth=12):
    """Re-score Model C v0 + the three 3-4B bases on the CURRENT eval_texts.json (v1's: the
    v0.2.1 general_holdout + authored_eval_v2). Model C v1's own row comes from its training
    final_full_bpb (g_flagship_v1_results.json), so the 5 rows are all on IDENTICAL slices."""
    token = _hf_token()
    if not token:
        print("[scorecard] WARNING: no HF_TOKEN in .env — gated bases (gemma/llama) will 401")
    v0_tag = G.FLAGSHIP["model_tag"].format(depth=v0_depth)
    v0_step = G.FLAGSHIP["num_iterations"]          # final v0 checkpoint (step == num_iter)
    results = {"part": "v1-scorecard", "v0_tag": v0_tag, "v0_step": v0_step,
               "eval_note": "all rows re-scored on the v1 eval_texts.json "
                            "(v0.2.1 general_holdout + authored_eval_v2); frozen slices are corpus-invariant",
               "modelc_v0": None, "bases": {}}
    with modal.enable_output(), app.run():
        r0 = bpb.remote(v0_tag, v0_step)
        results["modelc_v0"] = r0["bpb"]
        print(f"[scorecard] modelc-v0: { {k: round(v['bpb'],4) for k,v in r0['bpb'].items()} }")
        for repo, rev in BASES:
            rb = base_bpb.remote(repo, rev, token=token)
            results["bases"][repo] = rb["bpb"]
            print(f"[scorecard] {repo}: { {k: round(v['bpb'],4) for k,v in rb['bpb'].items() if v.get('bpb')} }")
    _save("g_v1_scorecard.json", results)


# --- base-model BPB on the same slices ---------------------------------------

BASES = [
    ("google/gemma-3-4b-pt", "cc012e0a6d0787b4adcc0fa2c4da74402494554d"),
    ("Qwen/Qwen3-4B-Base", "906bfd4b4dc7f14ee4320094d8b41684abff8539"),
    ("meta-llama/Llama-3.2-3B", "13afe5124825b4f3751f836b40dafda64c1ed062"),
]


def _hf_token():
    """Read HF_TOKEN from the repo-root .env (same walk as the Workstream-D probe)."""
    d = F.REPO_ROOT
    for _ in range(5):
        cand = os.path.join(d, ".env")
        if os.path.exists(cand):
            for line in open(cand, encoding="utf-8"):
                if line.strip().startswith("HF_TOKEN="):
                    v = line.strip().split("=", 1)[1].strip().strip('"').strip("'")
                    if v:
                        return v
        d = os.path.dirname(d)
    return None


def do_base_bpb():
    token = _hf_token()
    if not token:
        print("[base-bpb] WARNING: no HF_TOKEN in .env — gated bases (gemma/llama) will 401")
    results = {"part": "base-bpb", "bases": {}}
    with modal.enable_output(), app.run():
        for repo, rev in BASES:
            r = base_bpb.remote(repo, rev, token=token)
            results["bases"][repo] = r["bpb"]
            summary = {k: v["bpb"] for k, v in r["bpb"].items()}
            print(f"[base-bpb] {repo}: {summary}")
    _save("g_base_bpb.json", results)


# --- v1 exhibit: the 1901 heritage-fable cold prompt --------------------------

# Opening lines of «Le Loup, la Chèvre et le Chevreau», Georges Sylvain, Cric? Crac!
# (1901) — held out of training ENTIRELY (reports/heritage_cric_crac_fable.md), so this is
# a genuine cold prompt. Pre-reform Kreyòl orthography, 1901-print OCR, verbatim.
HERITAGE_FABLE_PROMPT = ("Loup, ti Cabritt avec manman Cabritt\n"
                         "Nan toutt temps, cabritt ac loup,\n")


def do_exhibits(depth=12, version="v1"):
    """Generate the heritage-fable cold prompt (greedy + one sampled) from the final flagship
    checkpoint → g_v1_exhibits.json (rendered by the report, alongside the slider asset)."""
    cfg_fl = G.FLAGSHIP_V1 if version == "v1" else G.FLAGSHIP
    tag = cfg_fl["model_tag"].format(depth=depth)
    step = cfg_fl["num_iterations"]
    with modal.enable_output(), app.run():
        hf = exhibit_gen.remote(tag, step, HERITAGE_FABLE_PROMPT,
                                max_tokens=64, temperature=0.8, seed=20260726)
    results = {"part": "v1-exhibits", "tag": tag, "step": step, "heritage_fable": hf}
    _save("g_v1_exhibits.json", results)
    print(f"[exhibits] heritage fable greedy → {hf['greedy'][:90]!r}")
    print(f"[exhibits] heritage fable sampled → {hf['sampled'][:90]!r}")


def do_diagnose_bos(depth=12, version="v1"):
    """Gate-6 root-cause probe: does llama.cpp prepend our BOS the way native does?"""
    cfg_fl = G.FLAGSHIP_V1 if version == "v1" else G.FLAGSHIP
    tag = cfg_fl["model_tag"].format(depth=depth)
    step = cfg_fl["num_iterations"]
    with modal.enable_output(), app.run():
        res = gguf_meta.remote(tag, step)
    _save("g_v1_bos_diagnosis.json", res)
    print(f"[diagnose-bos] ids_default[:3]={ (res.get('ids_default') or [])[:3] } "
          f"prepends_bos_24567={res.get('llamacpp_prepends_bos_24567')} "
          f"prepends_a_bos={res.get('llamacpp_prepends_a_bos')}")
    print(f"[diagnose-bos] tokenizer_kv={res.get('tokenizer_kv')}")


def do_bos_fix(depth=12, version="v1"):
    """PART 0 — apply the BOS fix (add_bos_token=true in the GGUF), reconvert f16+Q4, and
    re-run the cross-runtime greedy agreement check (gate 6). CPU-only, no GPU spend."""
    cfg_fl = G.FLAGSHIP_V1 if version == "v1" else G.FLAGSHIP
    tag = cfg_fl["model_tag"].format(depth=depth)
    step = cfg_fl["num_iterations"]
    with modal.enable_output(), app.run():
        res = reconvert_bos_and_check.remote(tag, step)
    _save("g_v1_bos_fix.json", res)
    print(f"[bos-fix] add_bos_token_in_gguf={res.get('add_bos_token_in_gguf')} "
          f"prepends_bos_24567={res.get('llamacpp_now_prepends_bos_24567')} "
          f"first_token_agree={res.get('n_first_token_agree')}/{res.get('n_prompts')} "
          f"mean_lcp={res.get('mean_native_vs_llamacpp_lcp')} clean={res.get('gate6_pass_clean')}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("upload")
    sub.add_parser("verify")
    g = sub.add_parser("gate"); g.add_argument("--skip-train", action="store_true")
    sw = sub.add_parser("sweep"); sw.add_argument("--depths", type=str, default="")
    fl = sub.add_parser("flagship"); fl.add_argument("--depth", type=int, required=True)
    fl.add_argument("--num-iter", type=int, default=None)
    fl.add_argument("--version", choices=["v0", "v1"], default="v0",
                    help="v1 = fleet-informed FLAGSHIP_V1 (corpus v0.2.1, ~1.0B tokens, distinct tag)")
    sub.add_parser("base-bpb")
    sc = sub.add_parser("scorecard"); sc.add_argument("--v0-depth", type=int, default=12)
    ex = sub.add_parser("exhibits"); ex.add_argument("--depth", type=int, default=12)
    ex.add_argument("--version", choices=["v0", "v1"], default="v1")
    db = sub.add_parser("diagnose-bos"); db.add_argument("--depth", type=int, default=12)
    db.add_argument("--version", choices=["v0", "v1"], default="v1")
    bf = sub.add_parser("bos-fix"); bf.add_argument("--depth", type=int, default=12)
    bf.add_argument("--version", choices=["v0", "v1"], default="v1")
    args = ap.parse_args()

    if args.cmd == "upload":
        do_upload()
    elif args.cmd == "verify":
        do_verify()
    elif args.cmd == "gate":
        do_gate(skip_train=args.skip_train)
    elif args.cmd == "sweep":
        depths = [int(x) for x in args.depths.split(",") if x.strip()] if args.depths else None
        do_sweep(depths)
    elif args.cmd == "flagship":
        do_flagship(args.depth, args.num_iter, version=args.version)
    elif args.cmd == "base-bpb":
        do_base_bpb()
    elif args.cmd == "scorecard":
        do_scorecard(args.v0_depth)
    elif args.cmd == "exhibits":
        do_exhibits(args.depth, version=args.version)
    elif args.cmd == "diagnose-bos":
        do_diagnose_bos(args.depth, version=args.version)
    elif args.cmd == "bos-fix":
        do_bos_fix(args.depth, version=args.version)


if __name__ == "__main__":
    main()
