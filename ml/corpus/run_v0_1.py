"""Phase-1 Workstream E orchestrator — junk filter → eval slices → report.

Layers the v0.1 stage on top of the frozen corpus-v0 shard. Requires corpus v0 to
have been built first (`python -m corpus.run`). Each stage is also runnable on its
own (`python -m corpus.junk`, `-m corpus.evalslices`, `-m corpus.report_v0_1`).

Run:
  cd ml && uv run python -m corpus.run_v0_1 --sample   # 1% end-to-end
  cd ml && uv run python -m corpus.run_v0_1            # full v0.1 build
"""

from __future__ import annotations

import argparse

from . import common, evalslices, junk, report_v0_1
from .run import load_env


def run(sample: bool):
    tag = common.run_tag(sample)
    common.log(f"=== corpus v0.1 build (Workstream E): {tag} ===")
    junk.junk_stage(sample)
    evalslices.build_slices(sample)
    report_v0_1.build(sample)
    common.log("=== v0.1 done ===")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    args = ap.parse_args()
    load_env()
    run(args.sample)


if __name__ == "__main__":
    main()
