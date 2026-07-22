"""Modal GPU harness for the base-model probe.

One remote function loads a candidate base checkpoint (unquantized bf16, via
transformers) from a cached Modal volume and runs the whole per-model battery:
bits-per-byte forward scoring + few-shot MT / proverb / naturalness greedy
generation. All measurement data is passed in from the local orchestrator; this
file only does GPU inference and returns raw results.

Run indirectly through `probe/run.py`. `hf_token` is passed as a call argument
(used only to fetch gated weights) and is never logged.
"""

from __future__ import annotations

import math
import time

import modal

from . import config

VOL = modal.Volume.from_name(config.MODAL_VOLUME, create_if_missing=True)
CACHE = "/cache"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.55.0",
        "accelerate>=1.2,<2",
        "huggingface_hub>=0.34,<1",
        "sentencepiece>=0.2",
        "protobuf>=4",
    )
    .env({"HF_HOME": f"{CACHE}/hf", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
)

app = modal.App(config.MODAL_APP_NAME)


# --- model loading (defensive; returns (model, tok, resolved_sha, arch)) -------

def _resolve_sha(repo, token):
    from huggingface_hub import HfApi
    return HfApi().model_info(repo, token=token).sha


def _from_pretrained(repo, common):
    """AutoModelForCausalLM, else the multimodal *-ForConditionalGeneration class
    (gemma-3): the full model handles text-only .generate() and returns vocab
    logits from a text forward pass, so we use it directly."""
    from transformers import AutoModelForCausalLM
    try:
        m = AutoModelForCausalLM.from_pretrained(repo, **common)
    except Exception as e_auto:
        try:
            from transformers import AutoModelForImageTextToText
            m = AutoModelForImageTextToText.from_pretrained(repo, **common)
        except Exception:
            raise e_auto
    arch = m.config.architectures[0] if m.config.architectures else "?"
    return m, arch


def _load(repo, token):
    """Load an unquantized bf16 causal LM + tokenizer at the resolved main SHA.
    Prefer SDPA attention (Gemma-3's default eager path is ~15x slower); fall
    back to the default impl if SDPA is unsupported. Raises on gated access."""
    import torch
    from transformers import AutoTokenizer

    sha = _resolve_sha(repo, token)
    tok = AutoTokenizer.from_pretrained(repo, revision=sha, token=token)
    base = dict(revision=sha, token=token, torch_dtype=torch.bfloat16,
                device_map="cuda", low_cpu_mem_usage=True)
    try:
        model, arch = _from_pretrained(repo, {**base, "attn_implementation": "sdpa"})
        attn = "sdpa"
    except Exception:
        model, arch = _from_pretrained(repo, base)   # default (often eager)
        attn = getattr(model.config, "_attn_implementation", "default")
    model.eval()
    return model, tok, sha, arch, attn


# --- measurement kernels ------------------------------------------------------

def _score_bpb(model, tok, texts, max_len):
    """Bits-per-byte over independent docs. See config.py for the exact policy:
    per-doc, start token prepended (BOS or EOS), start token never scored, long
    docs split into `max_len` windows, denominator = doc UTF-8 bytes."""
    import torch
    import torch.nn.functional as F

    device = next(model.parameters()).device
    start_id = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    total_bits, total_bytes, n_chunks = 0.0, 0, 0
    with torch.inference_mode():
        for text in texts:
            total_bytes += len(text.encode("utf-8"))
            ids = tok.encode(text, add_special_tokens=False)
            for s in range(0, len(ids), max_len):
                window = ids[s:s + max_len]
                if not window:
                    continue
                inp = torch.tensor([[start_id] + window], device=device)
                logits = model(inp).logits[0, :-1].float()      # predicts inp[1:]
                tgt = inp[0, 1:]
                nll_nats = F.cross_entropy(logits, tgt, reduction="sum")
                total_bits += float(nll_nats) / math.log(2)
                n_chunks += 1
    return {"nll_bits": total_bits, "bytes": total_bytes,
            "n_chunks": n_chunks, "n_docs": len(texts),
            "bpb": total_bits / total_bytes if total_bytes else None}


def _generate(model, tok, prompts, max_new, stop, batch_size):
    """Greedy batched completion; each output truncated at the first `stop`."""
    import torch

    if not prompts:
        return []
    device = next(model.parameters()).device
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    outs = []
    with torch.inference_mode():
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            enc = tok(batch, return_tensors="pt", padding=True,
                      add_special_tokens=True).to(device)
            plen = enc["input_ids"].shape[1]
            gen = model.generate(
                **enc, max_new_tokens=max_new, do_sample=False, num_beams=1,
                use_cache=True, pad_token_id=tok.pad_token_id,
                stop_strings=[stop], tokenizer=tok)
            for j in range(len(batch)):
                text = tok.decode(gen[j, plen:], skip_special_tokens=True)
                if stop and stop in text:
                    text = text.split(stop)[0]
                outs.append(text.strip())
    return outs


# --- remote entrypoint --------------------------------------------------------

@app.function(image=image, volumes={CACHE: VOL}, timeout=config.MODAL_TIMEOUT_S)
def prefetch(repos: list, hf_token: str) -> dict:
    """CPU-only: warm the weights cache and report gate accessibility BEFORE any
    GPU spend. Gated repos (Llama-3.2, maybe Gemma) surface their 401/403 here."""
    from huggingface_hub import HfApi, snapshot_download

    out = {}
    for repo in repos:
        try:
            sha = HfApi().model_info(repo, token=hf_token).sha
            snapshot_download(repo, revision=sha, token=hf_token,
                              ignore_patterns=["original/*", "*.pth", "*.gguf"])
            out[repo] = {"ok": True, "sha": sha}
        except Exception as e:
            out[repo] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    VOL.commit()
    return out


@app.function(image=image, gpu=config.MODAL_GPU, volumes={CACHE: VOL},
              timeout=config.MODAL_TIMEOUT_S)
def run_model(spec: dict, hf_token: str) -> dict:
    """Run the requested tasks for one candidate. `spec` keys:
        repo, tasks{bpb{full,authored}, mt{eng2hat,hat2eng}, proverbs, naturalness},
        gen{mt_max_new, prov_max_new, nat_max_new, batch}.
    Any task key may be absent. Returns raw results (no scoring here)."""
    import transformers

    repo = spec["repo"]
    t0 = time.time()
    try:
        model, tok, sha, arch, attn = _load(repo, hf_token)
    except Exception as e:
        return {"repo": repo, "ok": False, "error": f"{type(e).__name__}: {e}",
                "transformers": transformers.__version__}
    load_s = time.time() - t0
    VOL.commit()  # persist any freshly downloaded weights

    tasks = spec.get("tasks", {})
    gen = spec.get("gen", {})
    bs = gen.get("batch", config.GEN_BATCH_SIZE)
    out = {"repo": repo, "ok": True, "revision": sha, "arch": arch, "attn": attn,
           "dtype": "bfloat16", "transformers": transformers.__version__,
           "load_s": round(load_s, 1)}

    if "bpb" in tasks:
        out["bpb"] = {}
        for slice_name, texts in tasks["bpb"].items():
            ts = time.time()
            out["bpb"][slice_name] = _score_bpb(model, tok, texts, config.BPB_MAX_LEN)
            out["bpb"][slice_name]["seconds"] = round(time.time() - ts, 1)

    if "mt" in tasks:
        out["mt"] = {}
        for direction, prompts in tasks["mt"].items():
            out["mt"][direction] = _generate(
                model, tok, prompts, gen.get("mt_max_new", config.MT_MAX_NEW),
                config.MT_STOP, bs)

    if "proverbs" in tasks:
        out["proverbs"] = _generate(
            model, tok, tasks["proverbs"],
            gen.get("prov_max_new", config.PROVERB_MAX_NEW), config.PROVERB_STOP, bs)

    if "naturalness" in tasks:
        out["naturalness"] = _generate(
            model, tok, tasks["naturalness"],
            gen.get("nat_max_new", config.NAT_MAX_NEW), config.NAT_STOP, bs)

    out["total_s"] = round(time.time() - t0, 1)
    return out
