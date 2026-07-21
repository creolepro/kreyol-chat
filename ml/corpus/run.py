"""Workstream A orchestrator — ingest → normalize → filter → dedup → audit → report.

Each stage is also runnable on its own (`python -m corpus.<stage> [--sample]`);
this just chains them. Run the 1% sample end-to-end first (docs/phase-0.md),
then the full build.

Run:
  cd ml && uv run python -m corpus.run --sample   # 1% smoke test
  cd ml && uv run python -m corpus.run            # full build
"""

from __future__ import annotations

import argparse
import os

from . import audit, common, config, dedup, filter as filt, ingest, normalize, report


def load_env():
    """Read repo-root .env (HF_TOKEN, etc.) into os.environ if present."""
    root = os.path.dirname(config.REPO_ROOT)  # repo root (parent of ml/)
    path = os.path.join(root, ".env")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def run(sample: bool):
    tag = common.run_tag(sample)
    common.log(f"=== corpus v0 build: {tag} (sample={sample}) ===")
    ingest.ingest_madlad(sample)
    ingest.ingest_htwiki(sample)
    ingest.ingest_proverbs(sample)
    for source in config.PIPELINE_SOURCES:
        normalize.normalize_stage(sample, source)
    for source in config.PIPELINE_SOURCES:
        filt.filter_stage(sample, source)
    dedup.dedup_stage(sample)
    audit.audit_stage(sample)
    _out, totals = report.build_report(sample)
    common.log(f"=== done: {totals[0]:,} docs, {totals[4]:,} o200k tokens ===")
    return totals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true", help="run the whole pipeline on a 1%% sample")
    args = ap.parse_args()
    load_env()
    run(args.sample)


if __name__ == "__main__":
    main()
