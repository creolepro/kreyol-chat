"""Modal app for Workstream G — Model C as a STANDARD HF Llama.

The architecture swap (docs/phase-1.md G, binding): we train the real
`transformers.LlamaForCausalLM`, so `save_pretrained` yields a canonical HF repo and the
F2 conversion chain (GGUF / ONNX / Ollama / browser) has zero architectural divergence —
the whole reason the Workstream-F nanochat arch couldn't convert.

Functions (all on the shared Modal Volume):
  verify_params   — build each depth, assert param counts == the torch-free calc.
  train           — the training loop: fp32 master + bf16 autocast + AdamW + cosine;
                    grad-accum; torch.compile; HF checkpoint save/resume across calls;
                    per-checkpoint frozen-prompt generations + 4-slice BPB folded in.
  generate        — greedy generation from an HF checkpoint (ad-hoc).
  bpb             — 4-slice BPB from an HF checkpoint (ad-hoc / base-model comparison).
  convert_gates   — F2 gates 1-6: native↔HF logits, GGUF, stock-vs-patched llama.cpp,
                    token-ID parity, ONNX browser export, cross-runtime greedy agreement.
  base_bpb        — BPB of a HF base model (Gemma/Qwen/Llama) on the SAME slices.

Driven by train/g_run.py.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time

import modal

from . import config as F
from . import llama_config as G

VOL = modal.Volume.from_name(F.MODAL_VOLUME, create_if_missing=True)
CACHE = "/cache"
LLAMA_CPP = "/root/llama.cpp"

# Image: torch + transformers + the conversion/export toolchain. llama.cpp is cloned,
# C++-patched (kreyol-bpe pre-tokenizer), and built at image-build time so the GGUF gate
# runs against a compiled, patched llama-cli/llama-tokenize.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential", "cmake", "curl", "libcurl4-openssl-dev")
    .pip_install("torch==2.9.1")
    .pip_install(
        "numpy>=1.26", "transformers>=4.55", "tokenizers>=0.20", "tiktoken>=0.11",
        "safetensors", "gguf", "sentencepiece", "protobuf", "huggingface_hub>=0.34",
        "accelerate>=1.0", "optimum[onnxruntime]>=1.23",
    )
    .run_commands(
        f"git clone --filter=blob:none {G.LLAMA_CPP_REPO} {LLAMA_CPP}",
        f"cd {LLAMA_CPP} && git checkout {G.LLAMA_CPP_COMMIT}",
    )
    .add_local_file(os.path.join(F.REPO_ROOT, "train", "patch_llamacpp_cpp.py"),
                    "/root/patch_llamacpp_cpp.py", copy=True)
    .run_commands(
        # patched llama.cpp (kreyol-bpe pre-tokenizer registered): build CLI + tokenize + quantize
        f"python /root/patch_llamacpp_cpp.py {LLAMA_CPP}",
        f"cd {LLAMA_CPP} && (cmake -S . -B build -DLLAMA_CURL=OFF -DGGML_NATIVE=OFF "
        f"-DCMAKE_BUILD_TYPE=Release >/tmp/cmake_cfg.log 2>&1 || (tail -40 /tmp/cmake_cfg.log; false))",
        # llama-completion = the non-interactive one-shot generator (-st single-turn); the new
        # llama-cli is an interactive chat client that re-uses -p every loop (infinite gen).
        f"cd {LLAMA_CPP} && (cmake --build build --target llama-completion llama-tokenize "
        f"llama-quantize -j 4 >/tmp/cmake_build.log 2>&1 || (tail -60 /tmp/cmake_build.log; false))",
    )
    # STOCK Ollama (bundles STOCK llama.cpp) for gate 3 — the genuine stock-runtime test.
    # install.sh downloads a zstd-compressed bundle (it got past download in attempt 1, only
    # failing to extract without zstd) and skips the absent-systemd service step. zstd + this
    # are LATE layers so the cached llama.cpp build above is not invalidated.
    .apt_install("zstd")
    .run_commands("curl -fsSL https://ollama.com/install.sh | sh")
    .env({
        "HF_HUB_DISABLE_PROGRESS_BARS": "1", "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "8",
        "OLLAMA_MODELS": "/tmp/ollama_models",
        # UTF-8 locale so accented Kreyòl prompts survive argv into llama-cli / ollama
        "LC_ALL": "C.UTF-8", "LANG": "C.UTF-8",
        # reduce allocator fragmentation (the OOM message's own suggestion)
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
)

app = modal.App(F.MODAL_APP_NAME + "-g")


def _load_enc():
    """Our tiktoken Encoding from the Volume tokenizer bundle (tokenizer.pkl, written by
    train/prepare.build_tokenizer_bundle and uploaded to G_TOKENIZER_DIR)."""
    import pickle
    with open(os.path.join(G.G_TOKENIZER_DIR, "tokenizer.pkl"), "rb") as fh:
        return pickle.load(fh)


def _tok_encode_fn():
    enc = _load_enc()
    return enc.encode_ordinary, enc.encode_single_token("<|bos|>")


# ============================ verify param counts =============================

@app.function(image=image, volumes={CACHE: VOL}, timeout=600)
def verify_params() -> dict:
    import torch
    from . import llama_model as M

    out = {}
    for depth in G.DEPTHS:
        with torch.device("meta"):
            model = M.build_model(depth)
        real = sum(p.numel() for p in model.parameters())
        calc = M.param_count(depth)["total"]
        out[f"d{depth}"] = {"real": int(real), "calc": int(calc), "match": real == calc}
    out["arch"] = G.ARCH
    return out


# ================================ training ====================================

_CKPT = lambda tag: os.path.join(G.G_CKPT_DIR, tag)


def _save_ckpt(model, opt, step, rng_state, tag):
    import torch
    d = os.path.join(_CKPT(tag), f"step_{step}")
    os.makedirs(d, exist_ok=True)
    model.save_pretrained(d, safe_serialization=True)
    torch.save({"step": step, "opt": opt.state_dict(), "rng": rng_state},
               os.path.join(d, "training_state.pt"))
    VOL.commit()
    return d


# scaledown_window=2: the container shuts down right after each call, so a reused warm
# container can't carry a prior invocation's live GPU tensors into the next depth (the depth
# sweep's train.remote() calls otherwise accumulated d12+d16+d20 → OOM at d20).
@app.function(image=image, gpu=F.MODAL_GPU, volumes={CACHE: VOL}, timeout=3 * 60 * 60,
              scaledown_window=2)
def train(cfg: dict) -> dict:
    """Train (or resume) Model C. Fresh container each call → a successful resume proves
    HF checkpoints round-trip across Modal function calls (F2 requirement inherited)."""
    import gc
    import numpy as np
    import torch
    import torch.nn.functional as F_

    from . import llama_model as M
    from . import data_g
    from . import bpb_g

    # Modal may REUSE a warm container across successive train.remote() calls (the depth
    # sweep). Clear any residual model / optimizer / compiled-graph GPU memory from a prior
    # invocation up front, or a later (bigger) depth OOMs on accumulated allocations.
    try:
        torch._dynamo.reset()
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    torch.manual_seed(G.TRAIN["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")

    depth = cfg["depth"]
    tag = cfg["model_tag"]
    T = G.TRAIN
    seq_len = T["max_seq_len"]
    total_batch = T["total_batch_size"]
    dev_batch = cfg.get("device_batch_size") or T["device_batch_size"]   # per-run memory override
    grad_accum = total_batch // (dev_batch * seq_len)
    num_iter = cfg["num_iterations"]
    save_steps = sorted(set(cfg.get("save_steps", [])) | {num_iter})
    do_ckpt_evals = cfg.get("ckpt_evals", False)

    # data
    train_b = data_g.Batches(os.path.join(G.G_DATA_DIR, "train.bin"), seq_len, T["seed"])
    val_b = data_g.Batches(os.path.join(G.G_DATA_DIR, "val.bin"), seq_len, T["seed"] + 7)
    unique_tokens = train_b.unique_tokens()
    enc = _load_enc()
    encode, bos_id = enc.encode_ordinary, enc.encode_single_token("<|bos|>")
    eval_texts = json.load(open(os.path.join(G.G_DATA_DIR, "eval_texts.json"), encoding="utf-8")) \
        if do_ckpt_evals else {}
    prompts = json.load(open(G.G_CHECKPOINT_PROMPTS, encoding="utf-8"))["prompts"] \
        if do_ckpt_evals else []

    # model + optimizer
    model = M.build_model(depth, attn_impl=T["attn_impl"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=T["peak_lr"],
                            betas=(T["adam_beta1"], T["adam_beta2"]),
                            weight_decay=T["weight_decay"], fused=True)

    start_step = 0
    resume_from = cfg.get("resume_from_step", -1)
    if resume_from >= 0:
        from transformers import LlamaForCausalLM
        d = os.path.join(_CKPT(tag), f"step_{resume_from}")
        model = LlamaForCausalLM.from_pretrained(d, torch_dtype=torch.float32).to(device)
        model.config._attn_implementation = T["attn_impl"]
        opt = torch.optim.AdamW(model.parameters(), lr=T["peak_lr"],
                                betas=(T["adam_beta1"], T["adam_beta2"]),
                                weight_decay=T["weight_decay"], fused=True)
        state = torch.load(os.path.join(d, "training_state.pt"), map_location=device)
        opt.load_state_dict(state["opt"])
        start_step = state["step"]
        print(f"[train] resumed {tag} at step {start_step}")

    run_model = model
    if T["compile"]:
        try:
            run_model = torch.compile(model)
        except Exception as e:
            print(f"[train] compile failed ({e}); eager")
            run_model = model

    gpu_name = torch.cuda.get_device_name(0)
    flops_per_tok = 6 * sum(p.numel() for p in model.parameters())  # ~6ND
    h100_bf16 = 989e12
    logs, ckpt_records, t_win, tok_win = [], [], time.time(), 0

    for step in range(start_step, num_iter + 1):
        # ---- checkpoint (save weights; optionally gens + BPB) at the schedule points
        if step in save_steps and (step != start_step or step == 0 or step == num_iter):
            model.eval()
            _save_ckpt(model, opt, step, torch.get_rng_state(), tag)
            rec = {"step": step, "tokens": step * total_batch}
            if do_ckpt_evals:
                gens = []
                for p in prompts:
                    ids = [bos_id] + encode(p["prompt"])
                    x = torch.tensor([ids], device=device)
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                        y = model.generate(x, max_new_tokens=G.GEN_MAX_TOKENS, do_sample=False,
                                           num_beams=1, use_cache=True)
                    comp = enc.decode(y[0, len(ids):].tolist())
                    gens.append({"id": p["id"], "category": p["category"],
                                 "prompt": p["prompt"], "completion": comp})
                rec["generations"] = gens
                # per-checkpoint BPB (the learning CURVE) on a fixed CAPPED subset per slice —
                # consistent across checkpoints and fast (batch-1 BPB over ~2k docs is slow, and
                # there are ~8 checkpoints). The FULL-slice BPB for the vs-bases table is a single
                # bpb.remote() at the end (do_flagship), matching base_bpb exactly.
                cap = G.BPB_CKPT_MAX_DOCS
                rec["bpb"] = {}
                rec["bpb_cap_docs"] = cap
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    for name, texts in eval_texts.items():
                        sub = texts[:cap]
                        if sub:
                            rec["bpb"][name] = bpb_g.score_bpb(model, encode, sub, seq_len, bos_id)["bpb"]
                print(f"[ckpt {step}] tokens={rec['tokens']:,} bpb={rec['bpb']}")
            ckpt_records.append(rec)
            model.train()
            if step == num_iter:
                break

        # ---- one optimizer step (grad accumulation) --------------------------
        model.train()
        lr = data_g.lr_at(step, num_iter, T["peak_lr"], T["min_lr_frac"], T["warmup_steps"])
        for gp in opt.param_groups:
            gp["lr"] = lr
        opt.zero_grad(set_to_none=True)
        micro_loss = 0.0
        for xb, yb in train_b.step_batch(step, total_batch, dev_batch):
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = run_model(input_ids=xb)
                logits = out.logits
                loss = F_.cross_entropy(logits.view(-1, logits.size(-1)).float(),
                                        yb.view(-1), reduction="mean") / grad_accum
            loss.backward()
            micro_loss += float(loss)   # each term = batch_ce/grad_accum → sum = batch mean CE
        torch.nn.utils.clip_grad_norm_(model.parameters(), T["grad_clip"])
        opt.step()
        tok_win += total_batch

        if step % 10 == 0 or step == num_iter - 1:
            torch.cuda.synchronize()
            dt = time.time() - t_win
            tok_s = tok_win / dt if dt > 0 else 0
            mfu = flops_per_tok * tok_s / h100_bf16
            # periodic quick val CE
            val_ce = None
            if step % 50 == 0:
                model.eval()
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    vx, vy = next(val_b.step_batch(step, dev_batch * seq_len, dev_batch))
                    vx, vy = vx.to(device), vy.to(device)
                    vout = model(input_ids=vx).logits
                    val_ce = float(F_.cross_entropy(vout.view(-1, vout.size(-1)).float(), vy.view(-1)))
                model.train()
            logs.append({"step": step, "loss": round(micro_loss, 4), "lr": lr,
                         "tok_s": int(tok_s), "mfu": round(mfu, 3), "val_ce": val_ce})
            print(f"[train] step {step}/{num_iter} loss {micro_loss:.4f} "
                  f"tok/s {int(tok_s):,} mfu {mfu:.2f}"
                  + (f" val_ce {val_ce:.4f}" if val_ce else ""))
            t_win, tok_win = time.time(), 0

    # optional FULL-slice BPB at the final model (uncapped) — the vs-bases number, matching
    # base_bpb. Done here (same container) so the flagship needs no extra long client call.
    final_full_bpb = None
    if cfg.get("final_full_bpb") and eval_texts:
        model.eval()
        final_full_bpb = {}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for name, texts in eval_texts.items():
                if texts:
                    final_full_bpb[name] = bpb_g.score_bpb(model, encode, texts, seq_len, bos_id)
        print(f"[train] final full BPB: { {k: round(v['bpb'],4) for k,v in final_full_bpb.items()} }")

    # throughput median excludes only the compile/warmup transient (step 0-4), not the LR
    # warmup (which is the same compute per step).
    steady = [l["tok_s"] for l in logs if l["step"] >= 5 and l["tok_s"] > 0]
    out = {
        "tag": tag, "depth": depth, "num_iterations": num_iter,
        "params": int(sum(p.numel() for p in model.parameters())),
        "unique_train_tokens": unique_tokens,
        "effective_tokens": num_iter * total_batch,
        "epochs": round(num_iter * total_batch / max(1, unique_tokens), 3),
        "grad_accum": grad_accum, "gpu": gpu_name,
        "median_tok_s": int(sorted(steady)[len(steady) // 2]) if steady else None,
        "logs": logs, "checkpoints": ckpt_records,
        "checkpoint_steps": [r["step"] for r in ckpt_records],
        "final_full_bpb": final_full_bpb,
    }
    # self-persist to the Volume so a client disconnect (ephemeral app) never loses the run.
    rd = os.path.join(G.G_DIR, "results")
    os.makedirs(rd, exist_ok=True)
    with open(os.path.join(rd, f"train_{tag}.json"), "w") as fh:
        json.dump(out, fh)
    VOL.commit()
    return out


# ============================ ad-hoc generate / bpb ===========================

@app.function(image=image, gpu=F.MODAL_GPU, volumes={CACHE: VOL}, timeout=1200)
def generate(tag: str, step: int, prompts: list, max_tokens: int) -> dict:
    import torch
    from transformers import LlamaForCausalLM
    enc = _load_enc()
    bos_id = enc.encode_single_token("<|bos|>")
    d = os.path.join(_CKPT(tag), f"step_{step}")
    model = LlamaForCausalLM.from_pretrained(d, torch_dtype=torch.bfloat16).to("cuda").eval()
    outs = []
    for p in prompts:
        ids = [bos_id] + enc.encode_ordinary(p)
        x = torch.tensor([ids], device="cuda")
        with torch.inference_mode():
            y = model.generate(x, max_new_tokens=max_tokens, do_sample=False, num_beams=1)
        outs.append({"prompt": p, "completion": enc.decode(y[0, len(ids):].tolist())})
    return {"tag": tag, "step": step, "outputs": outs}


@app.function(image=image, gpu=F.MODAL_GPU, volumes={CACHE: VOL}, timeout=1800)
def bpb(tag: str, step: int, slices: list | None = None) -> dict:
    import torch
    from transformers import LlamaForCausalLM
    from . import bpb_g
    encode, bos_id = _tok_encode_fn()
    eval_texts = json.load(open(os.path.join(G.G_DATA_DIR, "eval_texts.json"), encoding="utf-8"))
    if slices:
        eval_texts = {k: v for k, v in eval_texts.items() if k in slices}
    d = os.path.join(_CKPT(tag), f"step_{step}")
    model = LlamaForCausalLM.from_pretrained(d, torch_dtype=torch.bfloat16).to("cuda").eval()
    res = {}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for name, texts in eval_texts.items():
            if texts:
                res[name] = bpb_g.score_bpb(model, encode, texts, G.TRAIN["max_seq_len"], bos_id)
    return {"tag": tag, "step": step, "bpb": res}


@app.function(image=image, gpu=F.MODAL_GPU, volumes={CACHE: VOL}, timeout=1200)
def exhibit_gen(tag: str, step: int, prompt: str, max_tokens: int = 64,
                temperature: float = 0.8, top_p: float = 0.95, seed: int = 20260726) -> dict:
    """One cold-prompt exhibit: greedy + ONE sampled completion from an HF checkpoint. Used for
    the 1901 heritage-fable cold prompt (a text the model never trained on)."""
    import torch
    from transformers import LlamaForCausalLM
    enc = _load_enc()
    bos_id = enc.encode_single_token("<|bos|>")
    d = os.path.join(_CKPT(tag), f"step_{step}")
    model = LlamaForCausalLM.from_pretrained(d, torch_dtype=torch.bfloat16).to("cuda").eval()
    ids = [bos_id] + enc.encode_ordinary(prompt)
    x = torch.tensor([ids], device="cuda")
    with torch.inference_mode():
        yg = model.generate(x, max_new_tokens=max_tokens, do_sample=False, num_beams=1)
        torch.manual_seed(seed)
        ys = model.generate(x, max_new_tokens=max_tokens, do_sample=True,
                            temperature=temperature, top_p=top_p, num_beams=1)
    return {"tag": tag, "step": step, "prompt": prompt, "max_tokens": max_tokens,
            "temperature": temperature, "top_p": top_p, "seed": seed,
            "greedy": enc.decode(yg[0, len(ids):].tolist()),
            "sampled": enc.decode(ys[0, len(ids):].tolist())}


# ============================ F2 conversion gates =============================

@app.function(image=image, gpu=F.MODAL_GPU, volumes={CACHE: VOL}, timeout=3600)
def convert_gates(tag: str, step: int, do_quant: bool = True, do_ollama: bool = True) -> dict:
    """Run F2 gates 1-6 on the (tag, step) HF checkpoint. Model C is a real HF Llama, so the
    export is architecturally lossless — these gates prove the deployment chain that the
    Workstream-F nanochat arch could not clear."""
    import torch
    from transformers import LlamaForCausalLM
    from . import gates

    enc = _load_enc()
    bos_id = enc.encode_single_token("<|bos|>")
    ck = os.path.join(G.G_CKPT_DIR, tag, f"step_{step}")
    hf_dir = "/tmp/hf_export"
    onnx_dir = "/tmp/onnx_export"
    art = os.path.join(G.G_ARTIFACT_DIR, tag)
    os.makedirs(art, exist_ok=True)
    tok_json = os.path.join(G.G_TOKENIZER_DIR, "tokenizer.json")

    probe = json.load(open(G.PARITY_PROBE, encoding="utf-8"))
    all_prompts = [p["prompt"] for p in json.load(open(G.G_CHECKPOINT_PROMPTS, encoding="utf-8"))["prompts"]]
    prompts = all_prompts[:5]                                   # gates 1/2/4/5/6 use the first 5

    # export the trained model as a clean HF repo (weights + our tokenizer.json)
    model = LlamaForCausalLM.from_pretrained(ck, torch_dtype=torch.float32).to("cuda").eval()
    gates.export_hf(model, tok_json, hf_dir)

    res = {"tag": tag, "step": step, "llama_cpp_commit": G.LLAMA_CPP_COMMIT}

    # gate 1 — native vs exported logits
    res["gate1_logits"] = gates.gate1_logits(hf_dir, "Dèyè mòn gen mòn. Mwen renmen peyi mwen.")

    # native greedy completions (feed gate 6 + a Kreyòl-emits sanity check)
    native = {}
    for p in prompts:
        ids = [bos_id] + enc.encode_ordinary(p)
        x = torch.tensor([ids], device="cuda")
        with torch.inference_mode():
            y = model.generate(x, max_new_tokens=32, do_sample=False, num_beams=1)
        native[p] = enc.decode(y[0, len(ids):].tolist())
    res["native_greedy"] = native

    # gate 2 — GGUF convert + patched-llama.cpp generation
    gguf = os.path.join(art, f"modelc-{tag}-step{step}-f16.gguf")
    res["gate2_gguf"] = gates.convert_gguf(hf_dir, gguf)
    if res["gate2_gguf"]["gguf_exists"]:
        res["gate2_gguf"]["llama_cpp_generation"] = gates.llama_generate(gguf, prompts[0], n=48)

    # gate 4 — three-way token-ID parity
    if res["gate2_gguf"]["gguf_exists"]:
        res["gate4_parity"] = gates.gate4_parity(enc, tok_json, gguf,
                                                 probe["probe_lines"], probe["fixtures"])

    # gate 3 — stock Ollama + GPT-2-fallback clitic damage
    if res["gate2_gguf"]["gguf_exists"] and do_ollama:
        res["gate3_stock"] = gates.gate3_stock(gguf, enc, hf_dir, probe["fixtures"])

    # gate 5 — ONNX (transformers.js path)
    res["gate5_onnx"] = gates.gate5_onnx(hf_dir, onnx_dir, enc, bos_id, prompts[0], n=32)

    # gate 6 — cross-runtime greedy + Q4
    gguf_q4 = None
    if res["gate2_gguf"]["gguf_exists"] and do_quant:
        gguf_q4 = os.path.join(art, f"modelc-{tag}-step{step}-Q4_K_M.gguf")
        res["gate6_quantize"] = gates.quantize_q4(gguf, gguf_q4)
    if res["gate2_gguf"]["gguf_exists"]:
        res["gate6_cross_runtime"] = gates.gate6_cross_runtime(
            gguf, native, res.get("gate5_onnx", {}).get("onnx_completion"),
            prompts, enc, bos_id, gguf_q4)

    # gate 6b — first-position logit margin vs fp32↔f16 delta on ALL 10 frozen prompts
    # (closes the v0 flat-distribution story: demonstrated, not asserted)
    res["gate6_first_pos_margin"] = gates.gate6_first_pos_margin(
        hf_dir, gguf if res["gate2_gguf"]["gguf_exists"] else None, all_prompts, enc, bos_id)
    print(f"[gate6b] {res['gate6_first_pos_margin']['verdict']}")

    # persist artifacts (GGUF/ONNX) to the Volume; hashes recorded for the report
    import hashlib
    def _sha(path):
        if not os.path.exists(path):
            return None
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()[:16]
    if os.path.isdir(onnx_dir):
        shutil.copytree(onnx_dir, os.path.join(art, "onnx"), dirs_exist_ok=True)
    res["artifacts"] = {
        "gguf_f16": {"path": gguf, "sha256_16": _sha(gguf),
                     "bytes": os.path.getsize(gguf) if os.path.exists(gguf) else 0},
        "gguf_q4": {"path": gguf_q4, "sha256_16": _sha(gguf_q4) if gguf_q4 else None,
                    "bytes": os.path.getsize(gguf_q4) if gguf_q4 and os.path.exists(gguf_q4) else 0},
        "onnx_dir": os.path.join(art, "onnx"),
    }
    # self-persist so a client disconnect never loses the gate results
    rd = os.path.join(G.G_DIR, "results")
    os.makedirs(rd, exist_ok=True)
    with open(os.path.join(rd, f"gates_{tag}.json"), "w") as fh:
        json.dump(res, fh)
    VOL.commit()
    return res


@app.function(image=image, volumes={CACHE: VOL}, timeout=300)
def read_result(name: str) -> dict | None:
    """Read a self-persisted result JSON from the Volume (client-side collection after a
    detached run whose client disconnected). Returns None if absent."""
    p = os.path.join(G.G_DIR, "results", name)
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)


@app.function(image=image, volumes={CACHE: VOL}, timeout=90 * 60)
def reconvert_bos_and_check(tag: str, step: int, do_quant: bool = True) -> dict:
    """PART 0 — the BOS fix, CPU-only. Re-export the (tag, step) checkpoint with
    `add_bos_token=true` baked into tokenizer_config.json (so the converter writes
    `tokenizer.ggml.add_bos_token` into the GGUF), reconvert f16 (+ Q4), and re-run the
    cross-runtime greedy agreement check (gate 6). Expected: llama.cpp now prepends our
    <|bos|> (24567) the way native does, first greedy tokens agree, and the LCP jumps from
    ~0 to clean. Everything CPU (fp32 native + `-ngl 0` llama.cpp) — no GPU spend."""
    import re as _re
    import torch
    from transformers import LlamaForCausalLM
    from . import gates

    enc = _load_enc()
    bos_id = enc.encode_single_token("<|bos|>")
    ck = os.path.join(G.G_CKPT_DIR, tag, f"step_{step}")
    hf_dir = "/tmp/hf_export_bosfix"
    art = os.path.join(G.G_ARTIFACT_DIR, tag)
    os.makedirs(art, exist_ok=True)
    tok_json = os.path.join(G.G_TOKENIZER_DIR, "tokenizer.json")
    all_prompts = [p["prompt"] for p in json.load(open(G.G_CHECKPOINT_PROMPTS, encoding="utf-8"))["prompts"]]

    # export (CPU, fp32) WITH the add_bos_token fix
    model = LlamaForCausalLM.from_pretrained(ck, torch_dtype=torch.float32).eval()
    gates.export_hf(model, tok_json, hf_dir)

    # native greedy (fp32, CPU) — the reference the runtimes must match
    native = {}
    for p in all_prompts:
        ids = [bos_id] + enc.encode_ordinary(p)
        x = torch.tensor([ids])
        with torch.inference_mode():
            y = model.generate(x, max_new_tokens=32, do_sample=False, num_beams=1)
        native[p] = enc.decode(y[0, len(ids):].tolist())

    # reconvert f16 (overwrites the v1 artifact so downstream picks up the fix) + Q4
    gguf = os.path.join(art, f"modelc-{tag}-step{step}-f16.gguf")
    conv = gates.convert_gguf(hf_dir, gguf)
    gguf_q4 = os.path.join(art, f"modelc-{tag}-step{step}-Q4_K_M.gguf")
    q4 = gates.quantize_q4(gguf, gguf_q4) if (do_quant and conv["gguf_exists"]) else None

    # confirm the metadata now carries add_bos_token=true + which BOS
    kv = {}
    try:
        from gguf import GGUFReader
        rd = GGUFReader(gguf)
        for name, field in rd.fields.items():
            if name.startswith("tokenizer.ggml.") and any(t in name for t in ("bos", "eos", "add_", "pre")):
                try:
                    kv[name] = field.contents()
                except Exception:
                    kv[name] = None
    except Exception as e:
        kv["_error"] = f"{type(e).__name__}: {e}"[:200]

    # does llama.cpp now prepend our BOS? (default vs --no-bos tokenize)
    def _ids(text, no_bos):
        cmd = [gates.LTOK, "-m", gguf, "-p", text, "--ids", "-ngl", "0"] + (["--no-bos"] if no_bos else [])
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        except Exception:
            return None
        m = _re.search(r"\[([\d,\s]*)\]", r.stdout)
        return [int(x) for x in m.group(1).split(",") if x.strip()] if m else None
    probe_txt = "Bonjou! Kijan ou ye jodi a?"
    ids_default = _ids(probe_txt, no_bos=False)
    ids_no_bos = _ids(probe_txt, no_bos=True)

    # cross-runtime greedy (llama.cpp f16, CPU) vs native — LCP + first-token agreement
    rows = []
    for p in all_prompts:
        lc = gates.llama_generate(gguf, p, n=32)
        cont = lc["text"].split(p, 1)[-1] if p in lc["text"] else lc["text"]
        nat = native.get(p, "")
        nat_first = enc.decode(enc.encode_ordinary(gates._norm_txt(nat))[:1]) if nat.strip() else ""
        lc_first = enc.decode(enc.encode_ordinary(gates._norm_txt(cont))[:1]) if cont.strip() else ""
        rows.append({"prompt": p,
                     "native_vs_llamacpp_lcp": gates._lcp_ratio(nat, cont),
                     "first_token_agree": gates._norm_txt(nat)[:1] == gates._norm_txt(cont)[:1]
                                          and bool(gates._norm_txt(nat)),
                     "native_preview": gates._norm_txt(nat)[:120],
                     "llamacpp_preview": gates._norm_txt(cont)[:120]})
    mean_lcp = round(sum(r["native_vs_llamacpp_lcp"] for r in rows) / max(1, len(rows)), 3)
    n_first_agree = sum(1 for r in rows if r["first_token_agree"])
    prepends_bos = bool(ids_default) and bool(ids_no_bos) and len(ids_default) == len(ids_no_bos) + 1 \
        and ids_default[0] == 24567
    clean = prepends_bos and n_first_agree >= max(1, (len(rows) * 8) // 10)

    import hashlib
    def _sha(path):
        if not path or not os.path.exists(path):
            return None
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()[:16]

    res = {
        "tag": tag, "step": step, "part": "part0-bos-fix",
        "convert_rc": conv["final_rc"], "gguf_exists": conv["gguf_exists"],
        "tokenizer_kv": kv,
        "add_bos_token_in_gguf": kv.get("tokenizer.ggml.add_bos_token"),
        "ids_default_first3": (ids_default or [])[:3], "ids_no_bos_first3": (ids_no_bos or [])[:3],
        "llamacpp_now_prepends_bos_24567": prepends_bos,
        "n_first_token_agree": n_first_agree, "n_prompts": len(rows),
        "mean_native_vs_llamacpp_lcp": mean_lcp,
        "gate6_pass_clean": clean,
        "verdict": (f"gate 6 CLEAN — add_bos_token now in GGUF; llama.cpp prepends <|bos|> 24567, "
                    f"first greedy token agrees with native on {n_first_agree}/{len(rows)} prompts, "
                    f"LCP {mean_lcp} (was 0.002). BOS/prompt-boundary mismatch RESOLVED."
                    if clean else
                    f"gate 6 NOT clean: prepends_bos={prepends_bos}, first-token agree "
                    f"{n_first_agree}/{len(rows)}, LCP {mean_lcp} — inspect rows."),
        "rows": rows,
        "artifacts": {
            "gguf_f16": {"path": gguf, "sha256_16": _sha(gguf),
                         "bytes": os.path.getsize(gguf) if os.path.exists(gguf) else 0},
            "gguf_q4": {"path": gguf_q4, "sha256_16": _sha(gguf_q4),
                        "bytes": os.path.getsize(gguf_q4) if os.path.exists(gguf_q4) else 0,
                        "rc": (q4 or {}).get("rc")},
        },
    }
    rd_dir = os.path.join(G.G_DIR, "results")
    os.makedirs(rd_dir, exist_ok=True)
    with open(os.path.join(rd_dir, f"bosfix_{tag}.json"), "w") as fh:
        json.dump(res, fh)
    VOL.commit()
    print(f"[reconvert_bos] {res['verdict']}")
    return res


@app.function(image=image, volumes={CACHE: VOL}, timeout=600)
def gguf_meta(tag: str, step: int) -> dict:
    """Diagnose the native↔llama.cpp first-token divergence (gate 6): does llama.cpp's default
    generation path prepend our bos_id 24567 the way the native HF path always does? CPU-only:
    read the GGUF BOS metadata (best-effort) + probe llama-tokenize default vs --no-bos."""
    import re
    import subprocess
    from . import gates
    gguf = os.path.join(G.G_ARTIFACT_DIR, tag, f"modelc-{tag}-step{step}-f16.gguf")
    out = {"tag": tag, "step": step, "gguf_exists": os.path.exists(gguf), "native_bos_id": 24567}
    if not out["gguf_exists"]:
        return out
    try:                                            # best-effort GGUF KV read
        from gguf import GGUFReader
        rd = GGUFReader(gguf)
        kv = {}
        for name, field in rd.fields.items():
            if name.startswith("tokenizer.ggml.") and any(t in name for t in ("bos", "eos", "add_", "pre")):
                try:
                    kv[name] = field.contents()
                except Exception:
                    kv[name] = None
        out["tokenizer_kv"] = kv
    except Exception as e:
        out["tokenizer_kv_error"] = f"{type(e).__name__}: {e}"[:200]

    def _ids(text, no_bos):                          # llama-tokenize (CPU, -ngl 0)
        cmd = [gates.LTOK, "-m", gguf, "-p", text, "--ids", "-ngl", "0"] + (["--no-bos"] if no_bos else [])
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        except Exception:
            return None
        m = re.search(r"\[([\d,\s]*)\]", r.stdout)
        return [int(x) for x in m.group(1).split(",") if x.strip()] if m else None
    txt = "Bonjou! Kijan ou ye jodi a?"
    out["ids_default"] = _ids(txt, no_bos=False)
    out["ids_no_bos"] = _ids(txt, no_bos=True)
    dflt = out["ids_default"] or []
    nob = out["ids_no_bos"] or []
    out["llamacpp_prepends_a_bos"] = bool(dflt) and bool(nob) and len(dflt) == len(nob) + 1
    out["llamacpp_prepends_bos_24567"] = bool(dflt) and dflt[0] == 24567
    return out


# ==================== base-model BPB on the SAME slices =======================

@app.function(image=image, gpu=F.MODAL_GPU, volumes={CACHE: VOL}, timeout=3600)
def base_bpb(repo: str, revision: str, token: str | None = None, slices: list | None = None) -> dict:
    """BPB of a HF base model (each tokenizes the shared texts itself → byte-comparable).
    `token` (threaded from the local repo-root .env, like the Workstream-D probe) unlocks the
    gated bases (gemma / llama)."""
    import math
    import torch
    import torch.nn.functional as F_
    from transformers import AutoModelForCausalLM, AutoTokenizer

    eval_texts = json.load(open(os.path.join(G.G_DATA_DIR, "eval_texts.json"), encoding="utf-8"))
    if slices:
        eval_texts = {k: v for k, v in eval_texts.items() if k in slices}
    tok = AutoTokenizer.from_pretrained(repo, revision=revision, token=token)
    model = AutoModelForCausalLM.from_pretrained(repo, revision=revision, token=token,
                                                 torch_dtype=torch.bfloat16).to("cuda").eval()
    start_id = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    max_len = G.TRAIN["max_seq_len"]
    res = {}
    with torch.inference_mode():
        for name, texts in eval_texts.items():
            tb, bytes_, chunks = 0.0, 0, 0
            for text in texts:
                bytes_ += len(text.encode("utf-8"))
                ids = tok.encode(text, add_special_tokens=False)
                for s in range(0, len(ids), max_len):
                    w = ids[s:s + max_len]
                    if not w:
                        continue
                    inp = torch.tensor([[start_id] + w], device="cuda")
                    lg = model(inp).logits[0, :-1].float()
                    tb += float(F_.cross_entropy(lg, inp[0, 1:], reduction="sum")) / math.log(2)
                    chunks += 1
            res[name] = {"bpb": tb / bytes_ if bytes_ else None, "bytes": bytes_, "n_docs": len(texts)}
    return {"repo": repo, "revision": revision, "bpb": res}
