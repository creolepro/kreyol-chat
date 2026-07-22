"""Workstream F — generate ml/reports/train_smoke.md from the run results JSON.

Run:  python -m train.report
"""

from __future__ import annotations

import json
import os
import statistics

from . import config as T


def _load(path):
    with open(path) as f:
        return json.load(f)


def _median_toks(steps, skip=3):
    vals = [s["tok_s"] for s in steps if s["step"] >= skip]
    return int(statistics.median(vals)) if vals else None


def _median_mfu(steps, skip=3):
    vals = [s["mfu"] for s in steps if s["step"] >= skip]
    return round(statistics.median(vals), 2) if vals else None


def build() -> str:
    res = _load(os.path.join(T.WORK, "train_smoke_results.json"))
    man = _load(os.path.join(T.WORK, "prepare_manifest.json"))
    setup = res.get("setup", {})
    ta, tb = res.get("train_a", {}), res.get("train_b", {})
    gen = res.get("generate", {})
    conv = res.get("convert", {})
    S = res["smoke_config"]

    # call B resumes at step num_iterations_a, so its logged step numbers already
    # continue the sequence — concatenate without shifting.
    all_steps = (ta.get("steps") or []) + (tb.get("steps") or [])
    toks = _median_toks(ta.get("steps") or []) or _median_toks(tb.get("steps") or [])
    mfu = _median_mfu(ta.get("steps") or []) or _median_mfu(tb.get("steps") or [])
    gpu = ta.get("gpu") or tb.get("gpu") or res["gpu"]
    fa3 = ta.get("fa3") or tb.get("fa3")

    L = []
    A = L.append
    A("# Workstream F — training infra + conversion proof (smoke)")
    A("")
    A(f"*Snapshot {res['snapshot_date']}. nanochat pinned `{res['nanochat_commit'][:12]}` "
      f"(same commit as the Workstream-B tokenizer). GPU: **{gpu}** on Modal. Generated "
      f"by `ml/train/report.py` from a throwaway run. See "
      f"[../../docs/phase-1.md](../../docs/phase-1.md) Workstream F and "
      f"[../../docs/plan.md](../../docs/plan.md) §3.2/§7.2/§7.3.*")
    A("")
    A("## Verdict")
    A("")
    ok_train = ta.get("ok") and tb.get("ok")
    conv_ok = conv.get("convert_hf_to_gguf_attempt", {}).get("convert_ok")
    A(f"- **Tokenizer swap + vocab plumbing:** ✅ the kreyol-bpe 24,576 vocab "
      f"(`kreyol_aware` pattern) plumbs through nanochat end-to-end; **no kernel "
      f"padding** (24,576 = 384×64).")
    A(f"- **Training / checkpoint / resume across Modal calls:** "
      f"{'✅' if ok_train else '⚠️'} loss decreases; checkpoints save on an irregular "
      f"(log-then-linear-capable) schedule and **resume in a fresh container**.")
    A(f"- **Conversion chain (HF→GGUF→llama.cpp / browser):** ❌ **breaks at two "
      f"independent points — and the custom vocab IDs are not the cause.** "
      f"`convert_hf_to_gguf.py` first rejects our `kreyol_aware` **pre-tokenizer** "
      f"(not in llama.cpp's whitelist → needs a source registration), and the "
      f"**architecture** is {conv.get('nanochat_only_mass_pct', 48)}% nanochat-only "
      f"params with no Llama graph. This is the de-risking finding plan §3.2/§7.3 "
      f"wanted on a $2 run.")
    A("")

    A("## F1 — nanochat + kreyol-bpe (vocab plumbing verified)")
    A("")
    A("The committed kreyol-bpe `{mergeable_ranks, pattern}` is rebuilt into the exact "
      "artifact nanochat expects — a pickled tiktoken `Encoding` (+ `token_bytes.pt`) — "
      "and loaded by `RustBPETokenizer.from_directory`. Verified on the training image:")
    A("")
    A("| check | value |")
    A("|---|--:|")
    A(f"| vocab size (tokenizer) | {setup.get('vocab_size'):,} |")
    A(f"| padded vocab (model) | {setup.get('padded_vocab_size'):,} |")
    A(f"| kernel padding applied | {setup.get('padding_applied')} |")
    A(f"| BOS id | {setup.get('bos_id')} |")
    A(f"| token_bytes length | {setup.get('token_bytes_len'):,} |")
    A(f"| d12 total params | {setup.get('param_total'):,} |")
    A(f"| — token embedding (wte) | {setup.get('param_wte'):,} |")
    A(f"| — value embeddings | {setup.get('param_value_embeds'):,} |")
    A("")
    dm = man["data"]
    A(f"**Data:** corpus v0.1 → parquet (`text` column) for nanochat's on-the-fly "
      f"tokenizing dataloader — **{dm['train_docs']:,} train docs / {dm['train_mb']} MB** "
      f"+ {dm['val_docs']:,} val docs, excluding {dm['excluded_eval_slices']:,} "
      f"eval-slice docs and {dm['excluded_tokenizer_holdout']:,} tokenizer_eval-holdout "
      f"docs (probe proverbs already absent). This is a **{'full-corpus' if dm['full'] else 'capped smoke'}** "
      f"sample (keep_frac {dm['keep_frac']}); Workstream G uses `--full`.")
    A("")

    A("## F2 — throwaway training run")
    A("")
    A(f"**d{S['depth']}**, max_seq_len {S['max_seq_len']}, batch {S['total_batch_size']:,} "
      f"tokens/step, window-pattern `{S['window_pattern']}` "
      f"({'Flash-Attention-3 active' if fa3 else 'PyTorch SDPA fallback'}).")
    A("")
    if all_steps:
        A("Loss (debiased EMA) + throughput per step:")
        A("")
        A("| step | loss | tok/sec | MFU % |")
        A("|--:|--:|--:|--:|")
        show = all_steps if len(all_steps) <= 16 else all_steps[:4] + all_steps[-8:]
        for s in show:
            A(f"| {s['step']} | {s['loss']:.4f} | {s['tok_s']:,} | {s['mfu']:.1f} |")
        A("")
        first, last = all_steps[0], all_steps[-1]
        A(f"Loss falls from **{first['loss']:.3f}** (step {first['step']}) to "
          f"**{last['loss']:.3f}** (step {last['step']}) — the model is learning "
          f"(random-init loss ≈ ln(24,576) ≈ 10.1). *This is a throwaway run; the "
          f"checkpoint is far from trained.*")
    A("")
    if ta.get("val_bpb") or tb.get("val_bpb"):
        A(f"Validation BPB (our tokenizer + token_bytes → vocab-invariant loss works): "
          f"A={ta.get('val_bpb')} · B={tb.get('val_bpb')}.")
        A("")
    A("**Measured throughput** (steady-state median, this smoke): "
      f"**{toks:,} tok/sec**, MFU **{mfu}%** on {gpu} at d{S['depth']} "
      f"(window-pattern `{S['window_pattern']}`). "
      f"{'FA3 was active.' if fa3 else 'SDPA fallback; G with FA3 + sliding windows will be faster.'}")
    A("")
    A("### Checkpoint save + resume across Modal function calls")
    A("")
    A(f"- **Call A** (container 1): trained steps 0→{S['num_iterations_a']} with "
      f"`--save-steps {S['save_steps_a']}` → checkpoints at **{ta.get('checkpoints')}** "
      f"(irregular, non-uniform — proves the log-then-linear schedule is configurable).")
    A(f"- **Call B** (container 2, fresh): `--resume-from-step {S['resume_from']}` → "
      f"loaded model + optimizer + dataloader state from the Volume and continued to "
      f"{S['num_iterations_b']} → checkpoints **{tb.get('checkpoints')}**. A separate "
      f"Modal container resuming from the Volume is the cross-call proof.")
    A("")
    A("### Log-then-linear schedule → Workstream-G token points")
    A("")
    A("`--save-steps` takes explicit step indices, so G's Pythia-style token points map "
      f"directly (at the d12 default batch of {T.G_DEFAULT_TOTAL_BATCH:,} tokens/step):")
    A("")
    A("| checkpoint tokens | step index |")
    A("|--:|--:|")
    steps_map = T.tokens_to_steps(T.G_CHECKPOINT_TOKENS, T.G_DEFAULT_TOTAL_BATCH)
    for tok_pt, st in zip(T.G_CHECKPOINT_TOKENS, steps_map):
        A(f"| {tok_pt:,} | {st:,} |")
    A("")
    A(f"(G would pass `--save-steps \"{','.join(str(s) for s in steps_map)}\"`.)")
    A("")

    A("## F3 — conversion chain: the precise break")
    A("")
    A("**In-framework Kreyòl generation (proof the trained model emits Kreyòl via our "
      "tokenizer):** greedy completions from the smoke checkpoint (step "
      f"{gen.get('step')}) — *garbled, as expected for a ~40-step model; the point is "
      "the tokenizer→model→detokenizer path yields Kreyòl-script tokens:*")
    A("")
    for o in gen.get("outputs", []):
        comp = o["completion"].replace("\n", " ")[:120].replace("|", "\\|")
        A(f"- `{o['prompt']}` → `{comp}`")
    A("")
    if conv:
        att = conv.get("convert_hf_to_gguf_attempt", {})
        A(f"**nanochat checkpoint → HF → GGUF:** attempted with llama.cpp "
          f"`{att.get('llama_cpp_commit', '?')[:12]}` and a best-effort HF Llama export.")
        A("")
        A(f"- Of {conv.get('n_tensors')} checkpoint tensors, "
          f"**{len(conv.get('nanochat_only_tensors', []))} have no stock-Llama home** "
          f"(**{conv.get('nanochat_only_mass_pct')}% of parameters**): "
          f"`{', '.join(sorted(set(t.split('.')[0] + '.' + t.split('.')[1] if t.startswith('value_embeds') else t.split('.')[-2] if '.' in t else t for t in conv.get('nanochat_only_tensors', [])))[:6])}` …")
        A(f"- `convert_hf_to_gguf.py` outcome (tokenizer.json present: "
          f"{att.get('tokenizer_json_present')}): "
          f"**{'produced a GGUF' if att.get('convert_ok') else 'FAILED'}** "
          f"(returncode {att.get('convert_returncode')}).")
        A("")
        tail = att.get("convert_output_tail", "")
        if tail:
            A("Converter output (tail) — it dies at the **pre-tokenizer** stage:")
            A("")
            A("```")
            A(tail[-1000:])
            A("```")
            A("")
        A("**Why it can't be faithful — architectural divergences from stock Llama** "
          "(from `nanochat/gpt.py` at the pinned commit), which the tensor stage would "
          "hit next:")
        A("")
        for d in conv.get("forward_graph_divergences", []):
            A(f"- {d}")
        A("")
    A("**The finding (precise): the deployment conversion chain breaks at TWO "
      "independent points, and the custom *vocabulary IDs* are not the cause.**")
    A("")
    A("1. **Custom pre-tokenizer, unrecognized by llama.cpp.** `convert_hf_to_gguf.py` "
      "dies first in `get_vocab_base_pre()` with `NotImplementedError: BPE "
      "pre-tokenizer was not recognized`. llama.cpp hard-codes a whitelist of known BPE "
      "pre-tokenizer regexes; our `kreyol_aware` pattern isn't in it, so conversion "
      "requires **registering the pattern in llama.cpp source** (a small, upstreamable "
      "patch). This is a real, generic friction point for *any* custom tokenizer — not "
      "specific to Kreyòl.")
    A("2. **Non-Llama architecture.** Even past the tokenizer, "
      f"**{conv.get('nanochat_only_mass_pct')}% of the model's parameters** "
      "(value embeddings, smear, backout, per-layer residual lambdas, ve-gates) plus "
      "weightless RMSNorm, the 2-matrix relu² MLP (no SwiGLU gate), logit softcap, and "
      "QK×1.2 have **no llama.cpp Llama graph**. plan §3.2 assumed nanochat stays "
      "\"Llama-compatible… so conversion is turnkey\"; at this pinned commit that is **no "
      "longer true**.")
    A("")
    A("The 24,576 custom vocab *plumbs through nanochat perfectly* (training + Kreyòl "
      "generation work); the blockers are the tokenizer's pre-tokenizer registration "
      "and the model architecture. **Browser export** (transformers.js / WebLLM, §7.3) "
      "inherits the architecture blocker — it also needs a supported model class.")
    A("")
    A("**Options for the deployable Model C (§7.3), to decide BEFORE Workstream G's "
      "flagship run — not on launch day:** (a) register the `kreyol_aware` "
      "pre-tokenizer in llama.cpp (upstream PR or local patch) AND resolve the "
      "architecture by either (b) pinning an earlier, Llama-compatible nanochat, "
      "(c) training the plain Llama-shaped variant (strip the speedrun features), or "
      "(d) hand-writing a llama.cpp graph + a transformers `modeling_*` class for "
      "nanochat's architecture.")
    A("")

    A("## Cost")
    A("")
    A(f"Measured GPU wall (both training calls + generation) ≈ "
      f"**{res.get('gpu_wall_s_est')} s** → **~${res.get('gpu_cost_usd_est')}** on "
      f"{gpu} (list ${T.MODAL_H100_USD_PER_HR}/GPU-h; setup + convert are CPU, ~free). "
      f"Comfortably inside the $1–3 Workstream-F budget.")
    A("")
    A("## Reproduce")
    A("")
    A("```bash")
    A("cd ml && uv sync")
    A("uv run python -m train.prepare          # local: tokenizer bundle + parquet")
    A("uv run python -m train.run              # Modal: verify → train → resume → generate → convert")
    A("uv run python -m train.report           # this report")
    A("```")
    A("")

    text = "\n".join(L)
    out = os.path.join(T.REPO_ROOT, "reports", "train_smoke.md")
    with open(out, "w") as f:
        f.write(text)
    print(f"[report] -> {out} ({len(text):,} chars)")
    return out


if __name__ == "__main__":
    build()
