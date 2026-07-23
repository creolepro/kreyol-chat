"""Model C architecture — a STANDARD HF Llama, built from train/llama_config.ARCH.

`llama_config(depth)` returns an HF `LlamaConfig`; `build_model(depth)` instantiates a
fresh `LlamaForCausalLM`. Because this IS the HF class (not a nanochat model mapped to
Llama after the fact), `save_pretrained` produces a canonical HF repo that converts to
GGUF / ONNX with zero architectural divergence — the entire point of the Workstream-G
architecture swap (docs/phase-1.md F3 finding).

`param_count(depth)` is a pure-arithmetic count (no torch) so the intended shapes can be
checked locally; the Modal `setup` fn cross-checks it against the real instantiated model.
"""

from __future__ import annotations

from . import llama_config as G


def param_count(depth: int, arch: dict | None = None) -> dict:
    """Exact standard-Llama parameter count (torch-free), broken down by group."""
    a = arch or G.ARCH
    h = a["hidden_size"]
    inter = a["intermediate_size"]
    n_head = a["num_attention_heads"]
    n_kv = a["num_key_value_heads"]
    head_dim = h // n_head
    v = a["vocab_size"]

    embed = v * h
    lm_head = 0 if a["tie_word_embeddings"] else v * h

    q = h * (n_head * head_dim)
    k = h * (n_kv * head_dim)
    vp = h * (n_kv * head_dim)
    o = (n_head * head_dim) * h
    attn = q + k + vp + o
    # SwiGLU: gate_proj + up_proj (h->inter) + down_proj (inter->h)
    mlp = 2 * (h * inter) + (inter * h)
    norms = 2 * h                       # input_layernorm + post_attention_layernorm (learned)
    per_layer = attn + mlp + norms
    final_norm = h

    total = embed + lm_head + depth * per_layer + final_norm
    return {
        "depth": depth,
        "total": total,
        "embed_tokens": embed,
        "lm_head": lm_head,
        "per_layer": per_layer,
        "attn_per_layer": attn,
        "mlp_per_layer": mlp,
        "non_embedding": total - embed - lm_head,
    }


def llama_config(depth: int):
    """HF LlamaConfig for Model C at the given depth."""
    from transformers import LlamaConfig

    a = G.ARCH
    return LlamaConfig(
        vocab_size=a["vocab_size"],
        hidden_size=a["hidden_size"],
        intermediate_size=a["intermediate_size"],
        num_hidden_layers=depth,
        num_attention_heads=a["num_attention_heads"],
        num_key_value_heads=a["num_key_value_heads"],
        hidden_act=a["hidden_act"],
        max_position_embeddings=a["max_position_embeddings"],
        rope_theta=a["rope_theta"],
        rms_norm_eps=a["rms_norm_eps"],
        attention_bias=a["attention_bias"],
        mlp_bias=a["mlp_bias"],
        tie_word_embeddings=a["tie_word_embeddings"],
        bos_token_id=None,              # our BOS is a special id in the tokenizer, set at data time
        eos_token_id=None,
        pad_token_id=None,
        torch_dtype="bfloat16",
    )


def build_model(depth: int, attn_impl: str = "sdpa"):
    """Instantiate a fresh LlamaForCausalLM (fp32 master weights; the training loop
    uses bf16 autocast — the stable nanoGPT recipe, and these models are tiny)."""
    from transformers import LlamaForCausalLM

    cfg = llama_config(depth)
    cfg._attn_implementation = attn_impl
    return LlamaForCausalLM(cfg)


if __name__ == "__main__":
    # torch-free sanity: print the intended counts for the decision-block check
    for d in G.DEPTHS:
        c = param_count(d)
        print(f"d{d}: total={c['total']:,}  (non-embed {c['non_embedding']:,}, "
              f"per-layer {c['per_layer']:,})")
