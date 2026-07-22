"""Modal app for Workstream F — nanochat training + conversion proof.

Image bakes nanochat at the pinned commit (with the --save-steps patch applied) and
its deps; the Modal Volume holds nanochat's base dir (tokenizer bundle, corpus v0.1
parquet shards, checkpoints). Functions:

  setup         — finalize the tokenizer bundle on the Volume (token_bytes.pt) and
                  VERIFY vocab-size plumbing + kernel padding at 24,576 / d12.
  train         — run scripts.base_train with our args; parse loss / tok-s / bpb.
  generate      — greedy Kreyòl generation from a checkpoint (in-framework proof).
  convert_probe — attempt nanochat-checkpoint → HF → GGUF; report the precise break.

Driven by train/run.py. No secrets needed (corpus is local; no gated models).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time

import modal

from . import config as T

NC = "/root/nanochat"
LLAMA_CPP = "/root/llama.cpp"
VOL = modal.Volume.from_name(T.MODAL_VOLUME, create_if_missing=True)
CACHE = "/cache"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential", "cmake", "curl")
    # PyPI torch 2.9.1 ships bundled CUDA and runs on H100; nanochat pins cu128 only
    # for its Hopper FA3 kernel — if that kernel doesn't load we fall back to SDPA
    # (we train with window-pattern L, which SDPA handles), so this is robust.
    .pip_install("torch==2.9.1")
    .pip_install(
        "numpy>=1.26", "pyarrow>=21", "tiktoken>=0.11", "rustbpe>=0.1",
        "wandb>=0.21", "filelock>=3.19", "psutil>=7", "requests", "kernels>=0.11.7",
        "huggingface_hub>=0.34", "jinja2", "pyyaml",
        # conversion-probe deps
        "transformers>=4.55", "gguf", "safetensors", "sentencepiece", "protobuf",
    )
    .run_commands(
        f"git clone {T.NANOCHAT_REPO} {NC}",
        f"cd {NC} && git fetch --depth 1 origin {T.NANOCHAT_COMMIT} && "
        f"git checkout {T.NANOCHAT_COMMIT}",
    )
    .add_local_file(os.path.join(T.REPO_ROOT, "train", "apply_savesteps.py"),
                    "/root/apply_savesteps.py", copy=True)
    .run_commands(f"python /root/apply_savesteps.py {NC}/scripts/base_train.py")
    .env({
        "NANOCHAT_BASE_DIR": T.MODAL_BASE_DIR,
        "WANDB_MODE": "disabled",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "8",
    })
)

app = modal.App(T.MODAL_APP_NAME)


def _nc_import():
    if NC not in sys.path:
        sys.path.insert(0, NC)


# --- setup: finalize tokenizer bundle + verify vocab plumbing -----------------

@app.function(image=image, volumes={CACHE: VOL}, timeout=600)
def setup() -> dict:
    """Convert token_bytes.json -> token_bytes.pt on the Volume, then load the
    tokenizer + build a d12 model on meta device to VERIFY the 24,576 vocab plumbs
    through nanochat and that pad_vocab_size_to=64 is a no-op (no kernel padding)."""
    import io
    import contextlib
    import torch

    tok_dir = os.path.join(T.MODAL_BASE_DIR, "tokenizer")
    with open(os.path.join(tok_dir, "token_bytes.json")) as f:
        tb = json.load(f)
    torch.save(torch.tensor(tb, dtype=torch.int32),
               os.path.join(tok_dir, "token_bytes.pt"))
    VOL.commit()

    _nc_import()
    from nanochat.tokenizer import get_tokenizer, get_token_bytes
    from nanochat.gpt import GPT, GPTConfig

    tk = get_tokenizer()
    vocab_size = tk.get_vocab_size()
    token_bytes = get_token_bytes(device="cpu")
    # Build d12 on meta device and capture whether vocab was padded.
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cfg = GPTConfig(sequence_len=2048, vocab_size=vocab_size, n_layer=12,
                        n_head=6, n_kv_head=6, n_embd=768, window_pattern="L")
        with torch.device("meta"):
            model = GPT(cfg)
    padded = model.transformer.wte.weight.shape[0]
    counts = model.num_scaling_params()
    return {
        "vocab_size": vocab_size,
        "token_bytes_len": int(token_bytes.numel()),
        "bos_id": tk.get_bos_token_id(),
        "padded_vocab_size": int(padded),
        "padding_applied": bool(padded != vocab_size),
        "padding_log": buf.getvalue().strip(),
        "param_total": int(counts["total"]),
        "param_wte": int(counts["wte"]),
        "param_lm_head": int(counts["lm_head"]),
        "param_value_embeds": int(counts["value_embeds"]),
    }


# --- train: run base_train, parse metrics -------------------------------------

_STEP_RE = re.compile(r"step (\d+)/(\d+).*?loss:\s*([\d.]+).*?tok/sec:\s*([\d,]+).*?bf16_mfu:\s*([\d.]+)")
_BPB_RE = re.compile(r"Validation bpb:\s*([\d.]+)")
_VOCAB_RE = re.compile(r"Vocab size:\s*([\d,]+)")
_PAD_RE = re.compile(r"Padding vocab_size from (\d+) to (\d+)")
_GPU_RE = re.compile(r"GPU:\s*(.+?)\s*\|")


def _run_base_train(cmd_args: list) -> dict:
    env = dict(os.environ)
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, "-m", "scripts.base_train", *cmd_args],
        cwd=NC, env=env, capture_output=True, text=True,
    )
    wall = time.time() - t0
    out = proc.stdout + "\n" + proc.stderr
    steps = [{"step": int(m.group(1)), "num_iterations": int(m.group(2)),
              "loss": float(m.group(3)), "tok_s": int(m.group(4).replace(",", "")),
              "mfu": float(m.group(5))} for m in _STEP_RE.finditer(out)]
    bpb = [float(m.group(1)) for m in _BPB_RE.finditer(out)]
    fa3 = "Using Flash Attention 3" in out
    gpu = (_GPU_RE.search(out).group(1) if _GPU_RE.search(out) else None)
    vocab = (_VOCAB_RE.search(out).group(1) if _VOCAB_RE.search(out) else None)
    return {
        "ok": proc.returncode == 0, "returncode": proc.returncode, "wall_s": round(wall, 1),
        "steps": steps, "val_bpb": bpb, "fa3": fa3, "gpu": gpu, "vocab_log": vocab,
        "padding_detected": bool(_PAD_RE.search(out)),
        "log_tail": "\n".join(out.splitlines()[-40:]),
    }


@app.function(image=image, gpu=T.MODAL_GPU, volumes={CACHE: VOL}, timeout=T.MODAL_TIMEOUT_S)
def train(cfg: dict) -> dict:
    """One base_train invocation (a fresh container each call — so a successful
    resume here proves checkpoint round-trips across Modal function calls)."""
    args = [
        "--depth", str(cfg["depth"]),
        "--max-seq-len", str(cfg["max_seq_len"]),
        "--device-batch-size", str(cfg["device_batch_size"]),
        "--total-batch-size", str(cfg["total_batch_size"]),
        "--window-pattern", cfg["window_pattern"],
        "--warmup-steps", str(cfg["warmup_steps"]),
        "--num-iterations", str(cfg["num_iterations"]),
        "--eval-every", str(cfg["eval_every"]),
        "--eval-tokens", str(cfg["eval_tokens"]),
        "--core-metric-every", str(cfg["core_metric_every"]),
        "--sample-every", str(cfg["sample_every"]),
        "--model-tag", cfg["model_tag"],
        "--run", "dummy",
    ]
    if cfg.get("save_steps"):
        args += ["--save-steps", cfg["save_steps"]]
    if cfg.get("resume_from", -1) >= 0:
        args += ["--resume-from-step", str(cfg["resume_from"])]
    res = _run_base_train(args)
    VOL.commit()
    # list checkpoints present after this call
    ckpt_dir = os.path.join(T.MODAL_BASE_DIR, "base_checkpoints", cfg["model_tag"])
    res["checkpoints"] = sorted(
        int(m.group(1)) for f in (os.listdir(ckpt_dir) if os.path.isdir(ckpt_dir) else [])
        for m in [re.match(r"model_(\d+)\.pt$", f)] if m
    )
    return res


# --- generate: greedy Kreyòl from a checkpoint (in-framework proof) -----------

@app.function(image=image, gpu=T.MODAL_GPU, volumes={CACHE: VOL}, timeout=600)
def generate(model_tag: str, step: int, prompts: list, max_tokens: int) -> dict:
    import torch
    _nc_import()
    from nanochat.checkpoint_manager import load_model

    device = torch.device("cuda")
    model, tok, meta = load_model("base", device, "eval", model_tag=model_tag, step=step)
    outs = []
    for p in prompts:
        ids = tok.encode(p, prepend="<|bos|>")
        gen = list(model.generate(ids, max_tokens=max_tokens, temperature=0.0))
        outs.append({"prompt": p, "completion": tok.decode(gen)})
    return {"model_tag": model_tag, "step": step, "outputs": outs,
            "model_config": meta.get("model_config")}


# --- convert_probe: nanochat checkpoint -> HF -> GGUF, report the break --------

# nanochat GPT param groups that HAVE a stock-Llama home vs those that do NOT.
_LLAMA_MAPPABLE_PREFIXES = ("transformer.wte", "lm_head", "transformer.h.")  # attn q/k/v/o
_NANOCHAT_ONLY = ("value_embeds", "smear_gate", "smear_lambda", "backout_lambda",
                  "resid_lambdas", "x0_lambdas", "ve_gate")


@app.function(image=image, volumes={CACHE: VOL}, timeout=1800)
def convert_probe(model_tag: str, step: int) -> dict:
    """Attempt the deployment conversion chain and report exactly where/why it breaks.
    The finding we expect: the custom 24k VOCAB plumbs through fine; the blocker is the
    ARCHITECTURE (nanochat's speedrun features have no llama.cpp/Llama graph)."""
    import torch
    _nc_import()
    from nanochat.checkpoint_manager import load_checkpoint

    ckpt_dir = os.path.join(T.MODAL_BASE_DIR, "base_checkpoints", model_tag)
    model_data, _, meta = load_checkpoint(ckpt_dir, step, torch.device("cpu"))
    sd = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}

    # 1) classify every checkpoint tensor
    mappable, nanochat_only, other = [], [], []
    mass_only = 0
    for k, v in sd.items():
        n = v.numel() if hasattr(v, "numel") else 0
        if any(tok in k for tok in _NANOCHAT_ONLY):
            nanochat_only.append(k); mass_only += n
        elif k.startswith(_LLAMA_MAPPABLE_PREFIXES):
            mappable.append(k)
        else:
            other.append(k)
    total_params = sum(v.numel() for v in sd.values() if hasattr(v, "numel"))

    # 2) structural forward-graph divergences from stock Llama (from nanochat/gpt.py)
    divergences = [
        "RMSNorm has NO learnable weight (F.rms_norm) — Llama/GGUF require a norm weight tensor per norm",
        "MLP is 2-matrix c_fc->relu()^2->c_proj — Llama is 3-matrix SwiGLU (gate_proj/up_proj/down_proj, SiLU); no gate_proj exists",
        "logit softcap (15*tanh(logits/15)) — not in the generic Llama graph",
        "Q,K scaled by 1.2 after QK-norm — not representable in llama.cpp Llama attention",
        "ResFormer value embeddings (per-layer value_embeds + input-dependent ve_gate) — no Llama analogue",
        "smear gate (mix previous token embedding) — no Llama analogue",
        "backout (subtract mid-layer residual before final norm) — no Llama analogue",
        "per-layer resid_lambdas / x0_lambdas residual scaling + x0 blending — no Llama analogue",
        "norm applied AFTER token embedding — not in the Llama graph",
    ]

    # 3) actually invoke convert_hf_to_gguf.py on a best-effort HF export to capture a
    #    concrete failure. We write a Llama-shaped config + the tensors that DO map,
    #    which is deliberately incomplete (no gate_proj, no norm weights) — the converter
    #    error is the evidence.
    convert_attempt = _attempt_gguf(sd, meta)

    return {
        "model_tag": model_tag, "step": step,
        "total_params": int(total_params),
        "n_tensors": len(sd),
        "mappable_tensors": len(mappable),
        "nanochat_only_tensors": nanochat_only,
        "nanochat_only_param_mass": int(mass_only),
        "nanochat_only_mass_pct": round(100 * mass_only / max(1, total_params), 2),
        "other_tensors": other,
        "forward_graph_divergences": divergences,
        "convert_hf_to_gguf_attempt": convert_attempt,
        "vocab_note": "the 24,576 custom vocab itself plumbs through cleanly (setup verified); the break is architectural, not vocabulary",
    }


def _attempt_gguf(sd: dict, meta: dict) -> dict:
    """Clone llama.cpp, write a minimal best-effort HF Llama dir, run convert_hf_to_gguf.py,
    capture the outcome. Any error is the finding (reported precisely)."""
    import torch

    out = {}
    # clone llama.cpp (record the actual HEAD; we don't pin a guessed SHA)
    if not os.path.isdir(LLAMA_CPP):
        r = subprocess.run(["git", "clone", "--depth", "1",
                            "https://github.com/ggml-org/llama.cpp.git", LLAMA_CPP],
                           capture_output=True, text=True)
        if r.returncode != 0:
            return {"stage": "clone_llama_cpp", "ok": False, "error": r.stderr[-800:]}
    head = subprocess.run(["git", "-C", LLAMA_CPP, "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    out["llama_cpp_commit"] = head
    convert_py = os.path.join(LLAMA_CPP, "convert_hf_to_gguf.py")
    out["convert_script_exists"] = os.path.exists(convert_py)

    # best-effort HF Llama export (KNOWN-incomplete): embeddings + attention + final norm.
    cfg = meta["model_config"]
    hf_dir = "/tmp/hf_export"
    os.makedirs(hf_dir, exist_ok=True)
    n_embd, n_layer, n_head = cfg["n_embd"], cfg["n_layer"], cfg["n_head"]
    config = {
        "architectures": ["LlamaForCausalLM"], "model_type": "llama",
        "hidden_size": n_embd, "intermediate_size": 4 * n_embd,
        "num_hidden_layers": n_layer, "num_attention_heads": n_head,
        "num_key_value_heads": cfg["n_kv_head"], "vocab_size": cfg["vocab_size"],
        "max_position_embeddings": cfg["sequence_len"], "rms_norm_eps": 1e-5,
        "rope_theta": 100000.0, "tie_word_embeddings": False, "torch_dtype": "float32",
    }
    with open(os.path.join(hf_dir, "config.json"), "w") as f:
        json.dump(config, f)
    # include our HF tokenizer.json (uploaded to the Volume) so the converter gets
    # PAST vocab loading and reaches the ARCHITECTURE — proving the break is the arch,
    # not the custom vocab.
    import shutil
    vol_tok_json = os.path.join(T.MODAL_BASE_DIR, "tokenizer", "tokenizer.json")
    out["tokenizer_json_present"] = os.path.exists(vol_tok_json)
    if out["tokenizer_json_present"]:
        shutil.copy(vol_tok_json, os.path.join(hf_dir, "tokenizer.json"))
    # map the tensors that have a Llama name; deliberately omit what has none
    from safetensors.torch import save_file
    tensors, mapped = {}, 0
    tensors["model.embed_tokens.weight"] = sd["transformer.wte.weight"].float()[:cfg["vocab_size"]].contiguous()
    tensors["lm_head.weight"] = sd["lm_head.weight"].float()[:cfg["vocab_size"]].contiguous()
    for i in range(n_layer):
        pre = f"transformer.h.{i}.attn."
        m = {f"model.layers.{i}.self_attn.q_proj.weight": pre + "c_q.weight",
             f"model.layers.{i}.self_attn.k_proj.weight": pre + "c_k.weight",
             f"model.layers.{i}.self_attn.v_proj.weight": pre + "c_v.weight",
             f"model.layers.{i}.self_attn.o_proj.weight": pre + "c_proj.weight",
             f"model.layers.{i}.mlp.up_proj.weight": f"transformer.h.{i}.mlp.c_fc.weight",
             f"model.layers.{i}.mlp.down_proj.weight": f"transformer.h.{i}.mlp.c_proj.weight"}
        for dst, srck in m.items():
            if srck in sd:
                tensors[dst] = sd[srck].float().contiguous(); mapped += 1
    save_file(tensors, os.path.join(hf_dir, "model.safetensors"))
    out["hf_tensors_written"] = len(tensors)
    out["hf_omitted_no_llama_name"] = ["mlp.gate_proj (SwiGLU gate — nanochat has none)",
                                       "input_layernorm/post_attention_layernorm weights (nanochat norms are weightless)",
                                       "all nanochat-only tensors (value_embeds/smear/backout/lambdas)"]

    # run the converter and capture the concrete outcome
    gguf_path = "/tmp/kreyol.gguf"
    r = subprocess.run(
        [sys.executable, convert_py, hf_dir, "--outfile", gguf_path, "--outtype", "f16"],
        capture_output=True, text=True, timeout=600,
    )
    out["convert_returncode"] = r.returncode
    out["convert_ok"] = r.returncode == 0 and os.path.exists(gguf_path)
    tail = (r.stdout + "\n" + r.stderr).splitlines()
    out["convert_output_tail"] = "\n".join(tail[-30:])
    return out
