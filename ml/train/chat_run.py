"""Workstream I orchestrator — upload chat bins → midtrain → SFT → eval → convert.

Each stage self-persists to the Modal Volume (disconnect-proof; re-run to resume). Step
counts are derived from the packed-bin token totals × the configured epochs.

Run:
  cd ml && uv run python -m train.chat_tokenize          # build bins (local)
  cd ml && uv run python -m train.chat_run upload
  cd ml && uv run python -m train.chat_run midtrain
  cd ml && uv run python -m train.chat_run sft
  cd ml && uv run python -m train.chat_run eval
  cd ml && uv run python -m train.chat_run convert
  cd ml && uv run python -m train.chat_run all           # upload→midtrain→sft→eval→convert
"""

from __future__ import annotations

import argparse
import json
import os

import modal

from . import chat_config as CC
from . import config as F
from .chat_app import (app, chat_train, chat_eval, chat_convert,
                       chat_read_result, chat_upload_check, chat_reset, chat_regression)

VOL = modal.Volume.from_name(F.MODAL_VOLUME, create_if_missing=True)


def _save(name, obj):
    os.makedirs(CC.CHAT_WORK, exist_ok=True)
    with open(os.path.join(CC.CHAT_WORK, name), "w") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)
    print(f"[chat_run] wrote {os.path.join(CC.CHAT_WORK, name)}")


def _manifest():
    return json.load(open(os.path.join(CC.CHAT_WORK, "chat_bins_manifest.json"), encoding="utf-8"))


def _num_iter(train_tokens, epochs, total_batch):
    return max(1, round(epochs * train_tokens / total_batch))


def do_upload():
    files = ["midtrain.bin", "midtrain.mask.bin", "midtrain_val.bin", "midtrain_val.mask.bin",
             "sft.bin", "sft.mask.bin", "sft_val.bin", "sft_val.mask.bin"]
    print("[chat_run] uploading chat bins to the Volume…")
    with VOL.batch_upload(force=True) as b:
        for fn in files:
            p = os.path.join(CC.CHAT_BUNDLE, fn)
            if os.path.exists(p):
                b.put_file(p, f"/chat/data/{fn}")
    with modal.enable_output(), app.run():
        sizes = chat_upload_check.remote()
    print(f"[chat_run] on Volume: {sizes}")


def do_midtrain():
    m = _manifest()["bins"]["midtrain"]
    S = CC.MIDTRAIN
    num_iter = _num_iter(m["train_tokens"], S["epochs"], S["total_batch_size"])
    cfg = {"phase": "midtrain", "model_tag": S["model_tag"], "bin_prefix": "midtrain",
           "resume_tag": S["resume_tag"], "resume_step": S["resume_step"],
           "num_iterations": num_iter, "max_seq_len": S["max_seq_len"],
           "total_batch_size": S["total_batch_size"], "device_batch_size": S["device_batch_size"],
           "peak_lr": S["peak_lr"], "min_lr_frac": S["min_lr_frac"], "warmup_frac": S["warmup_frac"],
           "weight_decay": S["weight_decay"], "adam_beta1": S["adam_beta1"],
           "adam_beta2": S["adam_beta2"], "grad_clip": S["grad_clip"], "seed": S["seed"]}
    print(f"[chat_run] midtrain: {num_iter} steps ({S['epochs']} epochs over {m['train_tokens']:,} tok)")
    with modal.enable_output(), app.run(detach=True):
        tr = chat_read_result.remote(f"train_{S['model_tag']}.json")
        if not tr:
            tr = chat_train.remote(cfg)
    _save("chat_midtrain_results.json", tr)
    print(f"[chat_run] midtrain done: final_loss={tr.get('final_loss')} epochs={tr.get('epochs')}")


def do_sft():
    man = _manifest()["bins"]
    S = CC.SFT
    mid_iter = _num_iter(man["midtrain"]["train_tokens"], CC.MIDTRAIN["epochs"],
                         CC.MIDTRAIN["total_batch_size"])
    num_iter = _num_iter(man["sft"]["train_tokens"], S["epochs"], S["total_batch_size"])
    cfg = {"phase": "sft", "model_tag": S["model_tag"], "bin_prefix": "sft",
           "resume_tag": CC.MIDTRAIN["model_tag"], "resume_step": mid_iter,
           "num_iterations": num_iter, "max_seq_len": S["max_seq_len"],
           "total_batch_size": S["total_batch_size"], "device_batch_size": S["device_batch_size"],
           "peak_lr": S["peak_lr"], "min_lr_frac": S["min_lr_frac"], "warmup_frac": S["warmup_frac"],
           "weight_decay": S["weight_decay"], "adam_beta1": S["adam_beta1"],
           "adam_beta2": S["adam_beta2"], "grad_clip": S["grad_clip"], "seed": S["seed"]}
    print(f"[chat_run] sft: {num_iter} steps ({S['epochs']} epochs over {man['sft']['train_tokens']:,} tok), "
          f"resume midtrain@{mid_iter}")
    with modal.enable_output(), app.run(detach=True):
        tr = chat_read_result.remote(f"train_{S['model_tag']}.json")
        if not tr:
            tr = chat_train.remote(cfg)
    _save("chat_sft_results.json", tr)
    print(f"[chat_run] sft done: final_loss={tr.get('final_loss')} epochs={tr.get('epochs')}")


def _sft_final_step():
    man = _manifest()["bins"]
    return _num_iter(man["sft"]["train_tokens"], CC.SFT["epochs"], CC.SFT["total_batch_size"])


def do_eval():
    tag = CC.SFT["model_tag"]
    step = _sft_final_step()
    with modal.enable_output(), app.run():
        res = chat_eval.remote(tag, step)
    _save("chat_eval_results.json", res)
    print(f"[chat_run] eval BPB: { {k: round(v['bpb'],4) for k,v in res['bpb'].items()} }")
    for fc in res["frozen_chat"][:3]:
        print(f"  [{fc['id']}] {fc['prompt'][:40]!r} -> {fc['chat_answer'][:80]!r}")


def do_convert():
    tag = CC.SFT["model_tag"]
    step = _sft_final_step()
    with modal.enable_output(), app.run(detach=True):
        res = chat_read_result.remote(f"convert_{tag}.json")
        if not res:
            res = chat_convert.remote(tag, step)
    _save("chat_convert_results.json", res)
    a = res.get("artifacts", {})
    print(f"[chat_run] convert: f16={a.get('gguf_f16',{}).get('bytes')} "
          f"q4={a.get('gguf_q4',{}).get('bytes')} kv={res.get('gguf_tokenizer_kv')}")


def do_reset_sft():
    tag = CC.SFT["model_tag"]
    with modal.enable_output(), app.run():
        r = chat_reset.remote(tag)
    print(f"[chat_run] reset {tag}: {r}")


def do_regression(label, step=None):
    """Run ml/corpus/chat_regression_prompts.json through the SFT model at temp 0 + 0.7."""
    tag = CC.SFT["model_tag"]
    step = step if step is not None else _sft_final_step()
    prompts = json.load(open(os.path.join(F.REPO_ROOT, "corpus", "chat_regression_prompts.json"),
                             encoding="utf-8"))["prompts"]
    with modal.enable_output(), app.run():
        res = chat_regression.remote(tag, int(step), prompts)
    _save(f"chat_regression_{label}.json", res)
    for r in res["regression"][:4]:
        print(f"  [{r['id']}] t0 -> {r['temp0'][:80]!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["upload", "midtrain", "sft", "eval", "convert", "all",
                                    "reset-sft", "regression"])
    ap.add_argument("--label", type=str, default="baseline")
    ap.add_argument("--step", type=int, default=None)
    args = ap.parse_args()
    if args.cmd == "reset-sft":
        do_reset_sft()
        return
    if args.cmd == "regression":
        do_regression(args.label, args.step)
        return
    if args.cmd in ("upload", "all"):
        do_upload()
    if args.cmd in ("midtrain", "all"):
        do_midtrain()
    if args.cmd in ("sft", "all"):
        do_sft()
    if args.cmd in ("eval", "all"):
        do_eval()
    if args.cmd in ("convert", "all"):
        do_convert()


if __name__ == "__main__":
    main()
