"""Workstream F orchestrator — prepare → upload → verify → train → generate → convert.

Runs the whole throwaway pipeline on Modal and writes the results JSON the report
reads. Two `train` calls (A then B-resume) run in SEPARATE Modal containers, so a
successful resume proves checkpoints round-trip across function calls.

Run:
  cd ml && uv run modal run -m train.run              # smoke (default)
  cd ml && uv run python -m train.run                 # same, via app.run()
  cd ml && uv run python -m train.run --full          # whole corpus (Workstream G data)
"""

from __future__ import annotations

import argparse
import json
import os

import modal

from . import config as T
from . import prepare
from .modal_app import app, setup, train, generate, convert_probe

VOL = modal.Volume.from_name(T.MODAL_VOLUME, create_if_missing=True)


def _ensure_bundle(full: bool):
    manifest = os.path.join(T.WORK, "prepare_manifest.json")
    if not os.path.exists(manifest):
        print("[run] building local bundle (prepare)…")
        prepare.build_tokenizer_bundle()
        prepare.build_parquet(full)


def _upload_bundle():
    print("[run] uploading tokenizer bundle + parquet to the Volume…")
    with VOL.batch_upload(force=True) as batch:
        batch.put_directory(T.BUNDLE_TOKENIZER, "/nanochat/tokenizer")
        batch.put_directory(T.BUNDLE_DATA, "/nanochat/base_data_climbmix")
        batch.put_file(T.KREYOL_BPE_HF_JSON, "/nanochat/tokenizer/tokenizer.json")


def _upload_tokenizer_json():
    with VOL.batch_upload(force=True) as batch:
        batch.put_file(T.KREYOL_BPE_HF_JSON, "/nanochat/tokenizer/tokenizer.json")


def convert_only():
    """Re-run only the (CPU, ~free) conversion probe on the existing checkpoint and
    merge the result into the results JSON — used after fixing the HF export."""
    out = os.path.join(T.WORK, "train_smoke_results.json")
    res = json.load(open(out))
    _upload_tokenizer_json()
    step = max(res["train_b"]["checkpoints"])
    with modal.enable_output(), app.run():
        res["convert"] = convert_probe.remote(T.SMOKE["model_tag"], step)
    att = res["convert"]["convert_hf_to_gguf_attempt"]
    print(f"[convert] tokenizer_json_present={att.get('tokenizer_json_present')} "
          f"convert_ok={att.get('convert_ok')} rc={att.get('convert_returncode')}")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[run] updated {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--skip-convert", action="store_true")
    ap.add_argument("--convert-only", action="store_true",
                    help="re-run only the CPU conversion probe on the existing checkpoint")
    args = ap.parse_args()

    if args.convert_only:
        convert_only()
        return

    _ensure_bundle(args.full)
    _upload_bundle()

    S = T.SMOKE
    results = {"snapshot_date": T.SNAPSHOT_DATE, "nanochat_commit": T.NANOCHAT_COMMIT,
               "gpu": T.MODAL_GPU, "smoke_config": S}
    with modal.enable_output(), app.run():
        results["setup"] = setup.remote()
        print(f"[setup] {json.dumps(results['setup'], indent=2)}")

        # call A: 0 -> num_iterations_a, with irregular save-steps
        cfgA = _train_cfg(S, S["num_iterations_a"], save_steps=S["save_steps_a"], resume_from=-1)
        results["train_a"] = train.remote(cfgA)
        print(f"[train A] ok={results['train_a']['ok']} "
              f"checkpoints={results['train_a']['checkpoints']} "
              f"wall={results['train_a']['wall_s']}s")

        # call B (fresh container): resume the step-N checkpoint, continue
        cfgB = _train_cfg(S, S["num_iterations_b"], save_steps="", resume_from=S["resume_from"])
        results["train_b"] = train.remote(cfgB)
        print(f"[train B/resume] ok={results['train_b']['ok']} "
              f"checkpoints={results['train_b']['checkpoints']} "
              f"wall={results['train_b']['wall_s']}s")

        last_step = max(results["train_b"]["checkpoints"] or [S["num_iterations_b"]])
        results["generate"] = generate.remote(S["model_tag"], last_step,
                                              T.GEN_PROMPTS, T.GEN_MAX_TOKENS)
        for o in results["generate"]["outputs"]:
            print(f"[gen] {o['prompt']!r} -> {o['completion']!r}")

        if not args.skip_convert:
            results["convert"] = convert_probe.remote(S["model_tag"], last_step)
            c = results["convert"]
            print(f"[convert] nanochat-only tensors: {len(c['nanochat_only_tensors'])} "
                  f"({c['nanochat_only_mass_pct']}% of params); "
                  f"convert_hf_to_gguf ok={c['convert_hf_to_gguf_attempt'].get('convert_ok')}")

    # cost estimate (GPU wall only; setup/convert CPU are ~free)
    gpu_wall = results["train_a"]["wall_s"] + results["train_b"]["wall_s"]
    if results.get("generate"):
        gpu_wall += 60  # generation container overhead estimate
    results["gpu_wall_s_est"] = round(gpu_wall, 1)
    results["gpu_cost_usd_est"] = round(gpu_wall / 3600 * T.MODAL_H100_USD_PER_HR, 2)

    os.makedirs(T.WORK, exist_ok=True)
    out = os.path.join(T.WORK, "train_smoke_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[run] results -> {out}  (est GPU ${results['gpu_cost_usd_est']})")


def _train_cfg(S, num_iterations, save_steps, resume_from):
    return {
        "depth": S["depth"], "max_seq_len": S["max_seq_len"],
        "device_batch_size": S["device_batch_size"], "total_batch_size": S["total_batch_size"],
        "window_pattern": S["window_pattern"], "warmup_steps": S["warmup_steps"],
        "num_iterations": num_iterations,
        "eval_every": S["eval_every"], "eval_tokens": S["eval_tokens"],
        "core_metric_every": S["core_metric_every"], "sample_every": S["sample_every"],
        "model_tag": S["model_tag"], "save_steps": save_steps, "resume_from": resume_from,
    }


if __name__ == "__main__":
    main()
