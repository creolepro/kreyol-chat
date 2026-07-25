"""Workstream H — Modal app for the micro-model fleet.

ONE warm container trains a whole batch of micro runs sequentially, self-persisting each
run's result to the Volume as it finishes (so a mid-batch crash loses only the in-flight
run, and a re-invocation resumes by skipping completed tags). Every run:
  build a fresh micro `LlamaForCausalLM` (fleet_config.MICRO_ARCH) -> AdamW + cosine ->
  train num_iterations at 2^19 tok/step -> BPB on the 5 fixed slices with the run's OWN
  tokenizer (byte-normalized, so kreyol-bpe vs english-24k is comparable).

`fleet_train_batch` runs the micro fleet; `precheck_depth` runs the full-size (G.ARCH)
d12/d16 depth pre-check on a fleet bin. Driven by train/fleet_run.py.
"""

from __future__ import annotations

import gc
import json
import os
import pickle
import time

import modal

from . import config as F
from . import fleet_config as H
from . import llama_config as G

VOL = modal.Volume.from_name(F.MODAL_VOLUME, create_if_missing=True)
CACHE = "/cache"

# lean image — no llama.cpp / onnx (the fleet measures BPB, not conversion).
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.9.1")
    .pip_install("numpy>=1.26", "transformers>=4.55", "tiktoken>=0.11", "safetensors")
    .env({"HF_HUB_DISABLE_PROGRESS_BARS": "1", "PYTHONUNBUFFERED": "1",
          "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "8",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
)

app = modal.App(F.MODAL_APP_NAME + "-fleet")


def _enc(tok_key: str):
    import tiktoken
    d = pickle.load(open(os.path.join(H.FLEET_TOK_DIR, f"{tok_key}.pkl"), "rb"))
    ranks, pattern = d["mergeable_ranks"], d["pattern"]
    offset = len(ranks)
    special = {name: offset + i for i, name in enumerate(F.SPECIAL_TOKENS)}
    return tiktoken.Encoding(name=f"fleet-{tok_key}", pat_str=pattern,
                             mergeable_ranks=ranks, special_tokens=special)


def _build_model(depth: int, arch: dict, attn_impl: str):
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(
        vocab_size=arch["vocab_size"], hidden_size=arch["hidden_size"],
        intermediate_size=arch["intermediate_size"], num_hidden_layers=depth,
        num_attention_heads=arch["num_attention_heads"],
        num_key_value_heads=arch["num_key_value_heads"], hidden_act=arch["hidden_act"],
        max_position_embeddings=arch["max_position_embeddings"], rope_theta=arch["rope_theta"],
        rms_norm_eps=arch["rms_norm_eps"], attention_bias=arch["attention_bias"],
        mlp_bias=arch["mlp_bias"], tie_word_embeddings=arch["tie_word_embeddings"],
        bos_token_id=None, eos_token_id=None, pad_token_id=None, torch_dtype="bfloat16")
    cfg._attn_implementation = attn_impl
    return LlamaForCausalLM(cfg)


def _cleanup():
    import torch
    try:
        torch._dynamo.reset()
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _train_one(run: dict) -> dict:
    """Train one run to completion; return result (BPB on the 5 slices + curve)."""
    import numpy as np
    import torch
    import torch.nn.functional as F_

    from . import data_g
    from . import bpb_g

    _cleanup()
    T = H.TRAIN
    arch = run.get("arch", H.MICRO_ARCH)
    depth = run.get("depth", H.MICRO_DEPTH)
    seq_len = T["max_seq_len"]
    total_batch = T["total_batch_size"]
    dev_batch = run.get("device_batch_size", T["device_batch_size"])
    grad_accum = total_batch // (dev_batch * seq_len)
    num_iter = run["num_iterations"]
    seed = run["seed"]
    warmup = max(20, round(T["warmup_frac"] * num_iter))
    peak_lr = run.get("peak_lr", T["peak_lr"])

    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")

    bin_path = os.path.join(H.FLEET_DATA_DIR, run["bin"])
    train_b = data_g.Batches(bin_path, seq_len, seed)
    unique_tokens = train_b.unique_tokens()

    model = _build_model(depth, arch, T["attn_impl"]).to(device)
    n_params = int(sum(p.numel() for p in model.parameters()))
    opt = torch.optim.AdamW(model.parameters(), lr=peak_lr,
                            betas=(T["adam_beta1"], T["adam_beta2"]),
                            weight_decay=T["weight_decay"], fused=True)
    run_model = model
    if T["compile"]:
        try:
            run_model = torch.compile(model)
        except Exception as e:
            print(f"[fleet] compile failed ({e}); eager")

    logs, t_win, tok_win = [], time.time(), 0
    for step in range(num_iter):
        lr = data_g.lr_at(step, num_iter, peak_lr, T["min_lr_frac"], warmup)
        for gp in opt.param_groups:
            gp["lr"] = lr
        opt.zero_grad(set_to_none=True)
        micro_loss = 0.0
        for xb, yb in train_b.step_batch(step, total_batch, dev_batch):
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = run_model(input_ids=xb).logits
                loss = F_.cross_entropy(logits.view(-1, logits.size(-1)).float(),
                                        yb.view(-1), reduction="mean") / grad_accum
            loss.backward()
            micro_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), T["grad_clip"])
        opt.step()
        tok_win += total_batch
        if step % 20 == 0 or step == num_iter - 1:
            torch.cuda.synchronize()
            dt = time.time() - t_win
            tok_s = tok_win / dt if dt > 0 else 0
            logs.append({"step": step, "loss": round(micro_loss, 4), "lr": lr, "tok_s": int(tok_s)})
            print(f"[fleet:{run['tag']}] step {step}/{num_iter} loss {micro_loss:.4f} tok/s {int(tok_s):,}")
            t_win, tok_win = time.time(), 0

    # BPB on the 5 fixed slices with THIS run's tokenizer (byte-normalized)
    enc = _enc(run["tok"])
    encode = enc.encode_ordinary
    bos = enc.encode_single_token("<|bos|>")
    eval_texts = json.load(open(os.path.join(H.FLEET_DATA_DIR, "eval_texts.json"), encoding="utf-8"))
    model.eval()
    bpb = {}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for name, texts in eval_texts.items():
            if texts:
                bpb[name] = round(bpb_g.score_bpb(model, encode, texts, seq_len, bos)["bpb"], 4)
    steady = [l["tok_s"] for l in logs if l["step"] >= 2 and l["tok_s"] > 0]
    out = {
        "tag": run["tag"], "variant": run.get("variant"), "tokenizer": run["tok"],
        "seed": seed, "depth": depth, "params": n_params,
        "num_iterations": num_iter, "effective_tokens": num_iter * total_batch,
        "unique_train_tokens": unique_tokens,
        "epochs": round(num_iter * total_batch / max(1, unique_tokens), 3),
        "warmup_steps": warmup, "peak_lr": peak_lr, "grad_accum": grad_accum,
        "median_tok_s": int(sorted(steady)[len(steady) // 2]) if steady else None,
        "final_loss": logs[-1]["loss"] if logs else None,
        "bpb": bpb,
    }
    del model, opt, run_model
    _cleanup()
    return out


def _persist(name: str, obj: dict):
    rd = H.FLEET_RESULTS
    os.makedirs(rd, exist_ok=True)
    with open(os.path.join(rd, name), "w") as fh:
        json.dump(obj, fh)
    VOL.commit()


@app.function(image=image, gpu=F.MODAL_GPU, volumes={CACHE: VOL}, timeout=4 * 60 * 60)
def fleet_train_batch(runs: list, force: bool = False) -> dict:
    """Train a batch of micro runs in ONE container, self-persisting each. Resumable:
    a run whose result JSON already exists on the Volume is skipped unless force."""
    import torch
    results, skipped = {}, []
    print(f"[fleet] batch of {len(runs)} runs on {torch.cuda.get_device_name(0)}")
    for run in runs:
        name = f"fleet_{run['tag']}.json"
        path = os.path.join(H.FLEET_RESULTS, name)
        if os.path.exists(path) and not force:
            skipped.append(run["tag"])
            results[run["tag"]] = json.load(open(path))
            print(f"[fleet] skip {run['tag']} (already on Volume)")
            continue
        t0 = time.time()
        res = _train_one(run)
        res["wall_seconds"] = round(time.time() - t0, 1)
        _persist(name, res)
        results[run["tag"]] = res
        print(f"[fleet] DONE {run['tag']}: {res['wall_seconds']}s "
              f"params={res['params']:,} tok/s={res['median_tok_s']} bpb={res['bpb']}")
    return {"results": results, "skipped": skipped}


@app.function(image=image, gpu=F.MODAL_GPU, volumes={CACHE: VOL}, timeout=3 * 60 * 60)
def precheck_depth(depth: int, bin_name: str, num_iter: int, seed: int, tag: str,
                   force: bool = False) -> dict:
    """Full-size (G.ARCH) depth pre-check on a fleet bin — G's exact recipe/seed."""
    name = f"precheck_{tag}.json"
    path = os.path.join(H.FLEET_RESULTS, name)
    if os.path.exists(path) and not force:
        print(f"[precheck] {tag} already on Volume")
        return json.load(open(path))
    run = {"tag": tag, "arch": G.ARCH, "depth": depth, "bin": bin_name,
           "tok": "kreyol-bpe", "seed": seed, "num_iterations": num_iter,
           "device_batch_size": 16, "peak_lr": G.TRAIN["peak_lr"], "variant": "v021_kbpe"}
    # G's recipe: warmup 100 fixed, peak_lr 1.5e-3 (override the fleet proportional warmup)
    res = _train_one_gstyle(run)
    _persist(name, res)
    return res


def _train_one_gstyle(run: dict) -> dict:
    """Depth pre-check variant of _train_one using G's EXACT schedule (fixed warmup 100,
    peak_lr 1.5e-3, cosine) so it is comparable to the original depth sweep."""
    import torch
    import torch.nn.functional as F_
    from . import data_g, bpb_g

    _cleanup()
    T = G.TRAIN
    seq_len = T["max_seq_len"]
    total_batch = T["total_batch_size"]
    dev_batch = run["device_batch_size"]
    grad_accum = total_batch // (dev_batch * seq_len)
    num_iter = run["num_iterations"]
    seed = run["seed"]
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")

    train_b = data_g.Batches(os.path.join(H.FLEET_DATA_DIR, run["bin"]), seq_len, seed)
    unique_tokens = train_b.unique_tokens()
    model = _build_model(run["depth"], G.ARCH, T["attn_impl"]).to(device)
    n_params = int(sum(p.numel() for p in model.parameters()))
    opt = torch.optim.AdamW(model.parameters(), lr=T["peak_lr"],
                            betas=(T["adam_beta1"], T["adam_beta2"]),
                            weight_decay=T["weight_decay"], fused=True)
    run_model = model
    if T["compile"]:
        try:
            run_model = torch.compile(model)
        except Exception as e:
            print(f"[precheck] compile failed ({e}); eager")
    logs, t_win, tok_win = [], time.time(), 0
    for step in range(num_iter):
        lr = data_g.lr_at(step, num_iter, T["peak_lr"], T["min_lr_frac"], T["warmup_steps"])
        for gp in opt.param_groups:
            gp["lr"] = lr
        opt.zero_grad(set_to_none=True)
        micro_loss = 0.0
        for xb, yb in train_b.step_batch(step, total_batch, dev_batch):
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = run_model(input_ids=xb).logits
                loss = F_.cross_entropy(logits.view(-1, logits.size(-1)).float(),
                                        yb.view(-1), reduction="mean") / grad_accum
            loss.backward()
            micro_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), T["grad_clip"])
        opt.step()
        tok_win += total_batch
        if step % 20 == 0 or step == num_iter - 1:
            torch.cuda.synchronize()
            dt = time.time() - t_win
            tok_s = tok_win / dt if dt > 0 else 0
            logs.append({"step": step, "loss": round(micro_loss, 4), "tok_s": int(tok_s)})
            print(f"[precheck:{run['tag']}] step {step}/{num_iter} loss {micro_loss:.4f} tok/s {int(tok_s):,}")
            t_win, tok_win = time.time(), 0

    enc = _enc("kreyol-bpe")
    eval_texts = json.load(open(os.path.join(H.FLEET_DATA_DIR, "eval_texts.json"), encoding="utf-8"))
    model.eval()
    bpb = {}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for name, texts in eval_texts.items():
            if texts:
                bpb[name] = round(bpb_g.score_bpb(model, enc.encode_ordinary, texts, seq_len,
                                                  enc.encode_single_token("<|bos|>"))["bpb"], 4)
    steady = [l["tok_s"] for l in logs if l["step"] >= 2 and l["tok_s"] > 0]
    out = {"tag": run["tag"], "depth": run["depth"], "params": n_params,
           "num_iterations": num_iter, "effective_tokens": num_iter * total_batch,
           "unique_train_tokens": unique_tokens,
           "epochs": round(num_iter * total_batch / max(1, unique_tokens), 3),
           "median_tok_s": int(sorted(steady)[len(steady) // 2]) if steady else None,
           "final_loss": logs[-1]["loss"] if logs else None, "bpb": bpb}
    del model, opt, run_model
    _cleanup()
    return out


@app.function(image=image, volumes={CACHE: VOL}, timeout=300)
def read_result(name: str) -> dict | None:
    p = os.path.join(H.FLEET_RESULTS, name)
    return json.load(open(p)) if os.path.exists(p) else None


@app.function(image=image, gpu=F.MODAL_GPU, volumes={CACHE: VOL}, timeout=600)
def verify_micro_params() -> dict:
    import torch
    with torch.device("meta"):
        m = _build_model(H.MICRO_DEPTH, H.MICRO_ARCH, "sdpa")
    real = int(sum(p.numel() for p in m.parameters()))
    from . import llama_model as M
    calc = M.param_count(H.MICRO_DEPTH, H.MICRO_ARCH)["total"]
    return {"real": real, "calc": calc, "match": real == calc,
            "depth": H.MICRO_DEPTH, "arch": H.MICRO_ARCH}
