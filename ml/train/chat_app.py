"""Workstream I — Modal functions for Model C chat: midtrain → SFT → eval → convert.

Reuses the Workstream-G image/app/Volume (no rebuild). The chat model continues from the
Model C v1 base checkpoint: midtraining (full-sequence loss over the Layer-1 conversation
bins) then SFT (response-masked loss over the Layer-3 bins). Eval = the 10 frozen prompts
answered in CHAT mode (the continuer→answerer transition), BPB regression on the standing
slices, a blinded naturalness set, and temperature-sampled exhibit outputs. Convert = the
full chain with the Part-0 BOS fix AND the nanochat chat template embedded.

Driven by train/chat_run.py.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time

from . import chat_config as CC
from . import config as F
from . import llama_config as G
from .llama_app import app, image, VOL, CACHE, _load_enc, _CKPT, _save_ckpt


def _sp(enc):
    return {k: enc.encode_single_token(v) for k, v in {
        "bos": CC.BOS, "user_start": CC.USER_START, "user_end": CC.USER_END,
        "assistant_start": CC.ASSISTANT_START, "assistant_end": CC.ASSISTANT_END}.items()}


def _lr_at(step, num_iter, peak, min_frac, warmup):
    import math
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    if step >= num_iter:
        return peak * min_frac
    prog = (step - warmup) / max(1, num_iter - warmup)
    return peak * (min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * prog)))


# ============================ chat training ===================================

@app.function(image=image, gpu=F.MODAL_GPU, volumes={CACHE: VOL}, timeout=3 * 60 * 60,
              scaledown_window=2)
def chat_train(cfg: dict) -> dict:
    """Resume from a base checkpoint and train on the chat bins with a per-token loss mask.
    phase='midtrain' → full loss (Layer 1); phase='sft' → response-masked loss (Layer 3)."""
    import gc
    import numpy as np
    import torch
    import torch.nn.functional as F_
    from transformers import LlamaForCausalLM
    from . import chat_tokenize as CT

    try:
        torch._dynamo.reset()
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

    torch.manual_seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")

    tag = cfg["model_tag"]
    prefix = cfg["bin_prefix"]
    seq_len = cfg["max_seq_len"]
    total_batch = cfg["total_batch_size"]
    dev_batch = cfg["device_batch_size"]
    grad_accum = total_batch // (dev_batch * seq_len)
    num_iter = cfg["num_iterations"]

    data_dir = CC.CHAT_DATA_DIR
    train_b = CT.ChatBatches(os.path.join(data_dir, f"{prefix}.bin"),
                             os.path.join(data_dir, f"{prefix}.mask.bin"), seq_len, cfg["seed"])
    val_b = CT.ChatBatches(os.path.join(data_dir, f"{prefix}_val.bin"),
                           os.path.join(data_dir, f"{prefix}_val.mask.bin"), seq_len, cfg["seed"] + 7)

    # RESUMABLE: prefer the latest intermediate checkpoint for THIS tag (so an interrupted
    # run continues), else start from the base (v1 for midtrain, midtrained for sft).
    save_every = cfg.get("save_every", 150)
    start_step = 0
    ck_dir = _CKPT(tag)
    existing = []
    if os.path.isdir(ck_dir):
        for d in os.listdir(ck_dir):
            if d.startswith("step_") and os.path.exists(os.path.join(ck_dir, d, "training_state.pt")):
                s = int(d.split("_")[1])
                if 0 < s < num_iter:
                    existing.append(s)
    resume_step = max(existing) if existing else None
    if resume_step is not None:
        src = os.path.join(ck_dir, f"step_{resume_step}")
        print(f"[chat_train:{prefix}] resuming from {tag} step {resume_step}")
    else:
        src = os.path.join(_CKPT(cfg["resume_tag"]), f"step_{cfg['resume_step']}")
    model = LlamaForCausalLM.from_pretrained(src, torch_dtype=torch.float32).to(device)
    model.config._attn_implementation = "sdpa"
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["peak_lr"],
                            betas=(cfg["adam_beta1"], cfg["adam_beta2"]),
                            weight_decay=cfg["weight_decay"], fused=True)
    if resume_step is not None:
        state = torch.load(os.path.join(src, "training_state.pt"), map_location=device)
        opt.load_state_dict(state["opt"])
        start_step = state["step"]
    run_model = model
    try:
        run_model = torch.compile(model)
    except Exception as e:
        print(f"[chat_train] compile failed ({e}); eager")

    warmup = max(2, int(num_iter * cfg["warmup_frac"]))
    logs = []
    t_win, tok_win = time.time(), 0
    for step in range(start_step, num_iter + 1):
        # periodic resumable checkpoint (survives an external cancellation)
        if step > start_step and step < num_iter and step % save_every == 0:
            model.eval()
            _save_ckpt(model, opt, step, torch.get_rng_state(), tag)
            model.train()
        if step == num_iter:
            model.eval()
            _save_ckpt(model, opt, step, torch.get_rng_state(), tag)
            break
        model.train()
        lr = _lr_at(step, num_iter, cfg["peak_lr"], cfg["min_lr_frac"], warmup)
        for gp in opt.param_groups:
            gp["lr"] = lr
        opt.zero_grad(set_to_none=True)
        micro_loss = 0.0
        for xb, yb, ym in train_b.step_batch(step, total_batch, dev_batch):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            ym = ym.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = run_model(input_ids=xb).logits
                ce = F_.cross_entropy(logits.view(-1, logits.size(-1)).float(),
                                      yb.view(-1), reduction="none")
                m = ym.view(-1).float()
                loss = (ce * m).sum() / m.sum().clamp(min=1.0) / grad_accum
            loss.backward()
            micro_loss += float(loss)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()
        tok_win += total_batch
        if step % 10 == 0 or step == num_iter - 1:
            torch.cuda.synchronize()
            dt = time.time() - t_win
            tok_s = tok_win / dt if dt > 0 else 0
            val_ce = None
            if step % 25 == 0:
                model.eval()
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    vx, vy, vm = next(val_b.step_batch(step, dev_batch * seq_len, dev_batch))
                    vx, vy, vm = vx.to(device), vy.to(device), vm.to(device)
                    vl = run_model(input_ids=vx).logits
                    vce = F_.cross_entropy(vl.view(-1, vl.size(-1)).float(), vy.view(-1), reduction="none")
                    val_ce = float((vce * vm.view(-1).float()).sum() / vm.sum().clamp(min=1))
                model.train()
            logs.append({"step": step, "loss": round(micro_loss, 4), "lr": lr,
                         "tok_s": int(tok_s), "val_ce": val_ce})
            print(f"[chat_train:{prefix}] step {step}/{num_iter} loss {micro_loss:.4f} "
                  f"tok/s {int(tok_s):,}" + (f" val_ce {val_ce:.4f}" if val_ce else ""))
            t_win, tok_win = time.time(), 0

    steady = [l["tok_s"] for l in logs if l["step"] >= 3 and l["tok_s"] > 0]
    out = {"tag": tag, "phase": cfg.get("phase"), "num_iterations": num_iter,
           "unique_tokens": train_b.unique_tokens(), "effective_tokens": num_iter * total_batch,
           "epochs": round(num_iter * total_batch / max(1, train_b.unique_tokens()), 3),
           "grad_accum": grad_accum, "median_tok_s": int(sorted(steady)[len(steady) // 2]) if steady else None,
           "final_loss": logs[-1]["loss"] if logs else None, "logs": logs}
    rd = os.path.join(CC.CHAT_DIR, "results")
    os.makedirs(rd, exist_ok=True)
    with open(os.path.join(rd, f"train_{tag}.json"), "w") as fh:
        json.dump(out, fh)
    VOL.commit()
    return out


# ============================ chat-mode generation ============================

def _chat_generate(model, enc, sp, prompt, max_tokens, device, do_sample=False,
                   temperature=0.7, top_p=0.95, seed=None):
    import torch
    ids = [sp["bos"], sp["user_start"]] + enc.encode_ordinary(prompt) + [sp["user_end"], sp["assistant_start"]]
    x = torch.tensor([ids], device=device)
    # a repetition penalty is essential for a 123M model — greedy without it degenerates into
    # loops. Matches the deployment default (llama.cpp/Ollama repeat_penalty).
    kw = dict(max_new_tokens=max_tokens, num_beams=1, eos_token_id=sp["assistant_end"],
              pad_token_id=sp["assistant_end"], repetition_penalty=1.3, no_repeat_ngram_size=3)
    if seed is not None:
        torch.manual_seed(seed)
    with torch.inference_mode():
        if do_sample:
            y = model.generate(x, do_sample=True, temperature=temperature, top_p=top_p, **kw)
        else:
            y = model.generate(x, do_sample=False, **kw)
    gen = y[0, len(ids):].tolist()
    if sp["assistant_end"] in gen:
        gen = gen[:gen.index(sp["assistant_end"])]
    return enc.decode(gen).strip()


@app.function(image=image, gpu=F.MODAL_GPU, volumes={CACHE: VOL}, timeout=60 * 60)
def chat_eval(tag: str, step: int) -> dict:
    """Chat-mode eval: (a) 10 frozen prompts answered as chat (greedy); (b) BPB regression on
    the standing slices; (c) 30 blinded naturalness outputs; (d) temperature-sampled exhibit."""
    import torch
    from transformers import LlamaForCausalLM
    from . import bpb_g

    enc = _load_enc()
    sp = _sp(enc)
    encode = enc.encode_ordinary
    d = os.path.join(_CKPT(tag), f"step_{step}")
    model = LlamaForCausalLM.from_pretrained(d, torch_dtype=torch.bfloat16).to("cuda").eval()
    device = torch.device("cuda")

    frozen = json.load(open(G.G_CHECKPOINT_PROMPTS, encoding="utf-8"))["prompts"]
    frozen_chat = []
    for p in frozen:
        ans = _chat_generate(model, enc, sp, p["prompt"], CC.CHAT_GEN_MAX_TOKENS, device)
        frozen_chat.append({"id": p["id"], "category": p["category"], "prompt": p["prompt"],
                            "chat_answer": ans})
        print(f"[chat_eval] {p['id']}: {ans[:80]!r}")

    # naturalness sheet: frozen 10 + 20 authored prompts, greedy
    nat = []
    nat_prompts = [p["prompt"] for p in frozen] + CC.NATURALNESS_PROMPTS
    for pr in nat_prompts[:CC.NATURALNESS_SHEET_N]:
        nat.append({"prompt": pr, "answer": _chat_generate(model, enc, sp, pr, CC.CHAT_GEN_MAX_TOKENS, device)})

    # temperature-sampled exhibit (labeled sampled)
    exhibit = []
    for pr in CC.CHAT_EXHIBIT_PROMPTS:
        exhibit.append({"prompt": pr, "sampled": _chat_generate(
            model, enc, sp, pr, CC.CHAT_GEN_MAX_TOKENS, device, do_sample=True,
            temperature=CC.CHAT_EXHIBIT_TEMP, top_p=CC.CHAT_EXHIBIT_TOP_P, seed=20260727)})

    # BPB regression on the standing slices (did chat tuning damage the LM?)
    eval_texts = json.load(open(os.path.join(G.G_DATA_DIR, "eval_texts.json"), encoding="utf-8"))
    bpb = {}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for name, texts in eval_texts.items():
            if texts:
                bpb[name] = bpb_g.score_bpb(model, encode, texts, CC.SFT["max_seq_len"], sp["bos"])
    print(f"[chat_eval] BPB: { {k: round(v['bpb'],4) for k,v in bpb.items()} }")

    res = {"tag": tag, "step": step, "frozen_chat": frozen_chat, "naturalness": nat,
           "exhibit_sampled": exhibit, "bpb": bpb}
    rd = os.path.join(CC.CHAT_DIR, "results")
    os.makedirs(rd, exist_ok=True)
    with open(os.path.join(rd, f"eval_{tag}.json"), "w") as fh:
        json.dump(res, fh, ensure_ascii=False)
    VOL.commit()
    return res


# ============================ chat conversion (Part 5) ========================

@app.function(image=image, gpu=F.MODAL_GPU, volumes={CACHE: VOL}, timeout=90 * 60)
def chat_convert(tag: str, step: int, do_quant: bool = True) -> dict:
    """Convert the chat model through the full chain WITH the BOS fix AND the chat template
    embedded (tokenizer.chat_template in the GGUF). Emits GGUF f16+Q4, ONNX, an Ollama
    Modelfile, and a gate-6 cross-runtime chat-mode agreement check."""
    import torch
    from transformers import LlamaForCausalLM
    from . import gates

    enc = _load_enc()
    sp = _sp(enc)
    ck = os.path.join(_CKPT(tag), f"step_{step}")
    hf_dir = "/tmp/hf_chat_export"
    onnx_dir = "/tmp/onnx_chat_export"
    art = os.path.join(CC.CHAT_ARTIFACT_DIR, tag)
    os.makedirs(art, exist_ok=True)
    tok_json = os.path.join(G.G_TOKENIZER_DIR, "tokenizer.json")

    model = LlamaForCausalLM.from_pretrained(ck, torch_dtype=torch.float32).to("cuda").eval()
    # export WITH the chat template (→ tokenizer.chat_template in the GGUF) + BOS fix
    gates.export_hf(model, tok_json, hf_dir, chat_template=CC.CHAT_TEMPLATE)

    res = {"tag": tag, "step": step, "llama_cpp_commit": G.LLAMA_CPP_COMMIT}
    res["gate1_logits"] = gates.gate1_logits(hf_dir, "Bonjou! Kijan ou ye?")

    # native chat-mode greedy answers to the frozen prompts (feed gate 6)
    frozen = [p["prompt"] for p in json.load(open(G.G_CHECKPOINT_PROMPTS, encoding="utf-8"))["prompts"]]
    native = {p: _chat_generate(model, enc, sp, p, 64, torch.device("cuda")) for p in frozen[:5]}
    res["native_chat"] = native

    gguf = os.path.join(art, f"modelc-chat-{tag}-step{step}-f16.gguf")
    res["gate2_gguf"] = gates.convert_gguf(hf_dir, gguf)
    gguf_q4 = None
    if res["gate2_gguf"]["gguf_exists"] and do_quant:
        gguf_q4 = os.path.join(art, f"modelc-chat-{tag}-step{step}-Q4_K_M.gguf")
        res["gate6_quantize"] = gates.quantize_q4(gguf, gguf_q4)

    # confirm add_bos_token + chat_template landed in the GGUF metadata
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
            if name == "tokenizer.chat_template":
                kv["has_chat_template"] = True
    except Exception as e:
        kv["_error"] = f"{type(e).__name__}: {e}"[:200]
    res["gguf_tokenizer_kv"] = kv

    # gate 4 token-ID parity (carried; chat specials don't change content tokenization)
    probe = json.load(open(G.PARITY_PROBE, encoding="utf-8"))
    if res["gate2_gguf"]["gguf_exists"]:
        res["gate4_parity"] = gates.gate4_parity(enc, tok_json, gguf, probe["probe_lines"][:200], probe["fixtures"])

    # gate 5 ONNX (transformers.js path)
    res["gate5_onnx"] = gates.gate5_onnx(hf_dir, onnx_dir, enc, sp["bos"], "Bonjou! Kijan ou ye?", n=24)

    # an Ollama Modelfile with the template (Part 5 deliverable)
    modelfile = (f"FROM ./{os.path.basename(gguf)}\n"
                 f'TEMPLATE """{{{{ if .System }}}}<|user_start|>{{{{ .System }}}}<|user_end|>{{{{ end }}}}'
                 f'<|user_start|>{{{{ .Prompt }}}}<|user_end|><|assistant_start|>{{{{ .Response }}}}<|assistant_end|>"""\n'
                 f'PARAMETER stop "<|assistant_end|>"\n'
                 f'PARAMETER stop "<|user_start|>"\n'
                 f'PARAMETER repeat_penalty 1.3\n'
                 f'PARAMETER temperature 0.7\n')
    with open(os.path.join(art, "Modelfile"), "w") as fh:
        fh.write(modelfile)
    res["modelfile"] = modelfile

    def _sha(p):
        if not p or not os.path.exists(p):
            return None
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for c in iter(lambda: fh.read(1 << 20), b""):
                h.update(c)
        return h.hexdigest()[:16]
    if os.path.isdir(onnx_dir):
        shutil.copytree(onnx_dir, os.path.join(art, "onnx"), dirs_exist_ok=True)
    res["artifacts"] = {
        "gguf_f16": {"path": gguf, "sha256_16": _sha(gguf),
                     "bytes": os.path.getsize(gguf) if os.path.exists(gguf) else 0},
        "gguf_q4": {"path": gguf_q4, "sha256_16": _sha(gguf_q4),
                    "bytes": os.path.getsize(gguf_q4) if gguf_q4 and os.path.exists(gguf_q4) else 0},
        "onnx_dir": os.path.join(art, "onnx"), "modelfile": os.path.join(art, "Modelfile"),
    }
    rd = os.path.join(CC.CHAT_DIR, "results")
    os.makedirs(rd, exist_ok=True)
    with open(os.path.join(rd, f"convert_{tag}.json"), "w") as fh:
        json.dump(res, fh, ensure_ascii=False)
    VOL.commit()
    return res


@app.function(image=image, gpu=F.MODAL_GPU, volumes={CACHE: VOL}, timeout=60 * 60)
def chat_regression(tag: str, step: int, prompts: list) -> dict:
    """Run the regression prompt list (Part 0 instrument) through a checkpoint at temp 0 and a
    single temp-0.7 sample — the before/after comparison for the v1.1 informal patch."""
    import torch
    from transformers import LlamaForCausalLM
    enc = _load_enc()
    sp = _sp(enc)
    d = os.path.join(_CKPT(tag), f"step_{step}")
    model = LlamaForCausalLM.from_pretrained(d, torch_dtype=torch.bfloat16).to("cuda").eval()
    dev = torch.device("cuda")
    out = []
    for p in prompts:
        t0 = _chat_generate(model, enc, sp, p["prompt"], CC.CHAT_GEN_MAX_TOKENS, dev, do_sample=False)
        t07 = _chat_generate(model, enc, sp, p["prompt"], CC.CHAT_GEN_MAX_TOKENS, dev, do_sample=True,
                             temperature=0.7, top_p=0.95, seed=20260728)
        out.append({"id": p["id"], "category": p["category"], "prompt": p["prompt"],
                    "temp0": t0, "temp0_7": t07})
        print(f"[chat_regression] {p['id']}: t0={t0[:70]!r}")
    return {"tag": tag, "step": step, "regression": out}


@app.function(image=image, volumes={CACHE: VOL}, timeout=300)
def chat_read_result(name: str) -> dict | None:
    p = os.path.join(CC.CHAT_DIR, "results", name)
    return json.load(open(p)) if os.path.exists(p) else None


@app.function(image=image, volumes={CACHE: VOL}, timeout=300)
def chat_reset(tag: str) -> dict:
    """Remove a stage's cached results + checkpoints so it re-runs cleanly (used when the
    SFT data changes and we re-SFT on the SAME midtrain checkpoint)."""
    import shutil
    removed = []
    rd = os.path.join(CC.CHAT_DIR, "results")
    for f in [f"train_{tag}.json", f"eval_{tag}.json", f"convert_{tag}.json"]:
        p = os.path.join(rd, f)
        if os.path.exists(p):
            os.remove(p); removed.append(f)
    ck = _CKPT(tag)
    if os.path.isdir(ck):
        shutil.rmtree(ck); removed.append(f"ckpt:{tag}")
    VOL.commit()
    return {"removed": removed}


@app.function(image=image, volumes={CACHE: VOL}, timeout=600)
def chat_upload_check() -> dict:
    """Confirm the chat bins landed on the Volume."""
    d = CC.CHAT_DATA_DIR
    return {p: (os.path.getsize(os.path.join(d, p)) if os.path.exists(os.path.join(d, p)) else None)
            for p in ["midtrain.bin", "midtrain.mask.bin", "sft.bin", "sft.mask.bin"]}
