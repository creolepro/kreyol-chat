"""Apply a minimal, auditable patch to nanochat's scripts/base_train.py.

Adds a `--save-steps` argument (explicit comma-separated step indices) and saves a
checkpoint whenever the step is in that set — the ONLY change needed to realize the
log-then-linear (Pythia-style) checkpoint schedule at Workstream-G's token points,
which don't fall on a single fixed interval. nanochat's built-in `--save-every` only
does uniform intervals.

Applied during the Modal image build after cloning nanochat at the pinned commit.
Uses exact string replacement (each asserted to match once) rather than line-number
patching, so it is robust to reformatting. Run: python apply_savesteps.py <path>
"""

import sys

REPLACEMENTS = [
    # 1) new CLI argument, right after --save-every
    (
        'parser.add_argument("--save-every", type=int, default=-1, help="save checkpoints every N steps (-1 = only at end)")',
        'parser.add_argument("--save-every", type=int, default=-1, help="save checkpoints every N steps (-1 = only at end)")\n'
        'parser.add_argument("--save-steps", type=str, default="", help="kreyol-chat: explicit comma-separated step indices to ALSO checkpoint at (log-then-linear schedule)")',
    ),
    # 2) parse the explicit step set right after args are parsed
    (
        "args = parser.parse_args()\nuser_config = vars(args).copy()  # for logging",
        "args = parser.parse_args()\n"
        "user_config = vars(args).copy()  # for logging\n"
        "KREYOL_SAVE_STEPS = {int(x) for x in args.save_steps.split(',') if x.strip() != ''}  # kreyol-chat",
    ),
    # 3) also save when step is an explicit save-step
    (
        "    if last_step or (step > 0 and step != args.resume_from_step and args.save_every > 0 and step % args.save_every == 0):",
        "    if last_step or (step in KREYOL_SAVE_STEPS and step != args.resume_from_step) or (step > 0 and step != args.resume_from_step and args.save_every > 0 and step % args.save_every == 0):",
    ),
]


def main(path):
    with open(path, encoding="utf-8") as f:
        src = f.read()
    for old, new in REPLACEMENTS:
        n = src.count(old)
        assert n == 1, f"expected exactly 1 match, got {n} for:\n{old[:80]}..."
        src = src.replace(old, new)
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    print(f"[apply_savesteps] patched {path} ({len(REPLACEMENTS)} edits)")


if __name__ == "__main__":
    main(sys.argv[1])
