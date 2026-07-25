#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
kakugo_audit.py — Workstream H Part 0c

Build a BLINDED native-speaker review sheet for the `Kreyol/kakugo-hat`
Hugging Face dataset (Apache-2.0, Haitian-Creole conversations). A native
speaker uses the sheet to judge conversation naturalness; their verdict
decides whether kakugo-hat can serve as "SFT Layer 1" for Workstream I.

This script is the single source of truth for the report: it loads the
dataset, computes stratification keys, draws a STRATIFIED sample of exactly
150 conversations with a fixed seed, and writes:

  - ml/reports/kakugo_audit_sheet.md          (blinded, committable report)
  - ml/data/interim/kakugo_audit_key.json     (git-ignored answer key)

Re-running with the same seed reproduces both files byte-for-byte.

------------------------------------------------------------------------------
ACTUAL SCHEMA FOUND (train split, 41,264 rows)
------------------------------------------------------------------------------
Columns:
  generation_method : str   4 values -> translated (14310),
                            context (10000), scenario (8666), topic (8288)
  prompt_type       : str   non-null ONLY for generation_method=="context"
                            (answer a question about / classify / translate /
                             improve / summarize)
  scenario          : str   free-text, non-null ONLY for method=="scenario"
  topic             : str   free-text, non-null ONLY for method=="topic"
  system            : str   TWO values, perfectly correlated with origin (see
                            below) — a "think first" prompt for all synthetic
                            rows, "Be concise" for all translated rows.
  messages          : list[{role, content}]  roles alternate user/assistant,
                            first turn always "user". No "system" role inside.

Turn-count (len(messages)) distribution:
  2 msgs: 40676   4: 104   6: 188   8: 148   10: 67   12: 51   14: 30

------------------------------------------------------------------------------
STRATIFICATION ACTUALLY USED (given the real schema)
------------------------------------------------------------------------------
(a) turn_bucket   : short (<=2 msgs) / medium (3-4) / long (>=5)  [DERIVED from
                    len(messages)]. In this corpus only even lengths occur, so
                    medium == 4-msg convos and long == 6..14-msg convos.
(b) topic_cluster : coarse, DETERMINISTIC keyword buckets on the first user
                    turn — greeting / translation / task / question / other.
                    RATIONALE: the native topic/scenario/prompt_type fields are
                    method-specific (each is populated for only one
                    generation_method), so NONE of them covers all rows. A
                    uniform, blind-safe cluster must be derived from turn text.
                    The native fields are still recorded in the answer key.
(c) origin        : translated vs synthetic — TAGGED via generation_method
                    (translated -> "translated"; context/scenario/topic ->
                    "synthetic"). Origin IS available, so it is stratified.

Structural correlation observed and honoured by the sampler:
  * Only the SHORT bucket contains synthetic rows; medium and long are 100%
    translated. So origin is stratified WITHIN the short bucket (both origins
    are well represented there), while medium/long carry only translated.

Allocation (coverage-oriented, NOT proportional): 120 short / 12 medium /
18 long. Multi-turn is ~1.4% of the corpus; a proportional draw would yield
~2 multi-turn convos, too few to judge. We intentionally over-sample multi-turn
so the reviewer sees enough of it. Within each turn bucket the target is
apportioned across (origin, topic_cluster) cells by the largest-remainder
(Hamilton) method, so the sample is reproducible and covers every present cell.

------------------------------------------------------------------------------
BLINDING (why the sheet stays honest)
------------------------------------------------------------------------------
The sheet shows ONLY the user/assistant turn text. Everything that could reveal
origin/topic is withheld:
  * the `system` field is NOT shown (it is a 1:1 origin tell);
  * `<think>...</think>` reasoning blocks are STRIPPED from assistant turns
    (they exist ONLY in synthetic rows and are English meta-reasoning, not
    Kreyol — showing them would break the blind and is irrelevant to Kreyol
    naturalness);
  * no generation_method / prompt_type / scenario / topic / turn-count / row-id
    is printed;
  * the 150 items are shuffled so their order encodes no stratum grouping.
The recoverable mapping lives in the git-ignored answer key.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
DATASET = "Kreyol/kakugo-hat"
SEED = 20260725
N_ITEMS = 150
TURN_TARGETS = {"short": 120, "medium": 12, "long": 18}  # sums to 150
MAX_TURN_CHARS = 1200  # truncate any single runaway turn to keep sheet usable

REPO_ROOT = Path("/Volumes/CaseSensitive/kreyol-chat")
ML_ROOT = REPO_ROOT / "ml"
SHEET_PATH = ML_ROOT / "reports" / "kakugo_audit_sheet.md"
KEY_PATH = ML_ROOT / "data" / "interim" / "kakugo_audit_key.json"
ENV_PATH = REPO_ROOT / ".env"

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
THINK_OPEN_RE = re.compile(r"<think>.*$", re.DOTALL | re.IGNORECASE)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def load_hf_token() -> str | None:
    """Read HF_TOKEN from the process env or the repo-root .env."""
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def turn_bucket(n_msgs: int) -> str:
    if n_msgs <= 2:
        return "short"
    if n_msgs <= 4:
        return "medium"
    return "long"


def origin_of(generation_method: str) -> str:
    return "translated" if generation_method == "translated" else "synthetic"


# Coarse, deterministic keyword clusters on the FIRST USER turn.
_GREET = ("bonjou", "bonswa", "bon swa", "alo", "salut", "hello", "sak pase",
          "sa k ap fèt", "sak ap fèt", "koman ou ye", "kijan ou ye",
          "ki jan ou ye", "kouman ou ye")
_TRANS = ("tradui", "traduksyon", "traduction", "translate", "an kreyòl",
          "an kreyol", "an angle", "an anglè", "an fransè", "an franse",
          "vin di an", "vle di", "in english", "into haitian", "an espay")
_TASK = ("ekri ", "fè yon", "fè pou", "bay yon", "bay 3", "bay kèk", "bay egzanp",
         "ban m", "ban mwen", "kreye", "prepare", "rezime", "rezò", "klase",
         "klasifye", "amelyore", "esplike", "eksplike", "montre", "kalkile",
         "jenere", "devlope", "tanpri", "souple", "silvouplè", "write ",
         "create ", "make ", "generate ", "list ", "draft ", "summarize",
         "explain ", "convert ", "design ", "build ", "plan ", "compose ")
_QWORD = ("kisa", "kijan", "poukisa", "ki jan", "konbyen", "kilès", "ki lès",
          "eske", "èske", "ki moun", "kimoun", "ki kote", "kote", "kilè",
          "ki sa", "ki risk", "how ", "what ", "why ", "which ", "who ")


def topic_cluster(first_user_text: str) -> str:
    """Deterministic coarse cluster of the opening user turn."""
    t = (first_user_text or "").lower().strip()
    if not t:
        return "other"
    if any(k in t for k in _TRANS):
        return "translation"
    # pure social opener: short AND greeting token present
    if len(t) < 100 and any(k in t for k in _GREET):
        return "greeting"
    if any(k in t for k in _TASK):
        return "task"
    if "?" in t or any(t.startswith(k) or (" " + k) in t for k in _QWORD):
        return "question"
    return "other"


def apportion(target: int, pops: dict) -> dict:
    """
    Largest-remainder (Hamilton) apportionment of `target` across cells with
    populations `pops` (key -> population), capped at each cell's population.
    Deterministic (ties broken by sorted key). Sum of result == target,
    assuming sum(pops) >= target.
    """
    keys = sorted(pops)
    total_pop = sum(pops[k] for k in keys)
    if total_pop <= 0:
        return {k: 0 for k in keys}
    quotas = {k: target * pops[k] / total_pop for k in keys}
    alloc = {k: min(int(quotas[k]), pops[k]) for k in keys}
    remaining = target - sum(alloc.values())
    # distribute leftover by largest fractional remainder, respecting caps
    order = sorted(keys, key=lambda k: (-(quotas[k] - int(quotas[k])), k))
    i = 0
    guard = 0
    while remaining > 0 and guard < 10000:
        k = order[i % len(order)]
        if alloc[k] < pops[k]:
            alloc[k] += 1
            remaining -= 1
        i += 1
        guard += 1
    return alloc


def strip_think(content: str) -> str:
    """Remove <think>...</think> reasoning blocks from an assistant turn."""
    out = THINK_RE.sub("", content or "")
    out = THINK_OPEN_RE.sub("", out)  # tolerate an unclosed <think>
    return out.strip()


def blockquote(text: str, cap: int = MAX_TURN_CHARS) -> str:
    """Truncate + blockquote turn text so embedded markdown can't break the
    sheet layout and long turns stay readable."""
    text = text if text is not None else ""
    if len(text) > cap:
        text = text[:cap].rstrip() + " …[truncated]"
    if text == "":
        text = "(empty turn)"
    return "\n".join("> " + line for line in text.split("\n"))


# ----------------------------------------------------------------------------
# Main build
# ----------------------------------------------------------------------------
def main() -> None:
    import random

    from datasets import load_dataset

    token = load_hf_token()
    ds = load_dataset(DATASET, token=token)["train"]
    n_rows = ds.num_rows

    gm = ds["generation_method"]
    msgs_all = ds["messages"]

    # ---- compute stratification keys for every row -------------------------
    rows = []  # per-row metadata
    cells = defaultdict(list)  # (turn_bucket, origin, topic_cluster) -> [idx]
    for i in range(n_rows):
        msgs = msgs_all[i]
        tb = turn_bucket(len(msgs))
        org = origin_of(gm[i])
        first_user = next((m["content"] for m in msgs if m["role"] == "user"), "")
        tc = topic_cluster(first_user)
        rows.append({"tb": tb, "origin": org, "tc": tc, "gm": gm[i]})
        cells[(tb, org, tc)].append(i)

    # ---- allocate the 150 across turn buckets, then (origin, cluster) cells -
    alloc = {}  # cell -> count
    for tb, tb_target in TURN_TARGETS.items():
        sub_pops = {
            key: len(idxs) for key, idxs in cells.items() if key[0] == tb
        }
        sub_alloc = apportion(tb_target, sub_pops)
        alloc.update(sub_alloc)

    assert sum(alloc.values()) == N_ITEMS, sum(alloc.values())

    # ---- draw the sample (seeded, deterministic over sorted cell order) ----
    rng = random.Random(SEED)
    selected = []  # source indices
    for key in sorted(alloc):
        k = alloc[key]
        if k <= 0:
            continue
        pool = sorted(cells[key])  # sort for determinism
        selected.extend(rng.sample(pool, k))
    assert len(selected) == N_ITEMS, len(selected)

    # ---- shuffle into display order (blind the stratum grouping) -----------
    shuffle_rng = random.Random(SEED + 1)
    shuffle_rng.shuffle(selected)

    # ---- build the sheet + key --------------------------------------------
    sheet_lines = []
    sheet_lines.append(build_header())

    key_items = []
    for pos, src in enumerate(selected, start=1):
        msgs = msgs_all[src]
        meta = rows[src]
        sheet_lines.append(f"## Item {pos}\n")
        for m in msgs:
            role = m["role"]
            label = "**User:**" if role == "user" else "**Assistant:**"
            content = m["content"]
            if role == "assistant":
                content = strip_think(content)
            sheet_lines.append(label)
            sheet_lines.append(blockquote(content))
            sheet_lines.append("")  # blank line between turns
        sheet_lines.append("_naturalness (1–5):_ `____`  ")
        sheet_lines.append("- [ ] **wrong / awkward** — tick if any turn is "
                           "factually wrong, mistranslated, or awkward")
        sheet_lines.append("")

        ex = ds[int(src)]
        key_items.append({
            "item": pos,
            "source_index": int(src),
            "generation_method": ex["generation_method"],
            "origin": meta["origin"],
            "turn_bucket": meta["tb"],
            "num_messages": len(msgs),
            "topic_cluster": meta["tc"],
            "prompt_type": ex["prompt_type"],
            "scenario": ex["scenario"],
            "topic": ex["topic"],
            "system_variant": ("think" if ex["generation_method"] != "translated"
                               else "concise"),
        })

    SHEET_PATH.parent.mkdir(parents=True, exist_ok=True)
    SHEET_PATH.write_text("\n".join(sheet_lines).rstrip() + "\n", encoding="utf-8")

    # ---- stratification counts (for report + key) --------------------------
    strat = {
        "by_turn_bucket": dict(Counter(it["turn_bucket"] for it in key_items)),
        "by_origin": dict(Counter(it["origin"] for it in key_items)),
        "by_topic_cluster": dict(Counter(it["topic_cluster"] for it in key_items)),
        "by_generation_method": dict(Counter(it["generation_method"] for it in key_items)),
        "by_turn_bucket_x_origin": {
            f"{tb}|{org}": c for (tb, org), c in sorted(
                Counter((it["turn_bucket"], it["origin"]) for it in key_items).items())
        },
    }

    KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
    key_doc = {
        "dataset": DATASET,
        "dataset_num_rows": n_rows,
        "seed": SEED,
        "n_items": N_ITEMS,
        "turn_targets": TURN_TARGETS,
        "max_turn_chars": MAX_TURN_CHARS,
        "note": ("Blind sheet withholds the `system` field and strips "
                 "<think>...</think> from assistant turns; both are 1:1 origin "
                 "tells. Multi-turn convos are 100% translated in this corpus."),
        "stratification": strat,
        "items": key_items,  # item -> source_index de-shuffle map + buckets
    }
    KEY_PATH.write_text(json.dumps(key_doc, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    # ---- console report ----------------------------------------------------
    print(f"dataset            : {DATASET}  (rows={n_rows})")
    print(f"seed               : {SEED}")
    print(f"sheet              : {SHEET_PATH}")
    print(f"answer key         : {KEY_PATH}")
    print(f"items written      : {len(key_items)}")
    print("stratification counts:")
    for k, v in strat.items():
        print(f"  {k}: {v}")


def build_header() -> str:
    return (
"""# Blinded naturalness review — kakugo-hat conversations

_150 Haitian-Creole conversations drawn from a candidate SFT dataset, shown
de-identified. Each item is a full conversation (all turns, role-labelled).
No source information is shown — dataset origin, topic, generation method, and
turn counts are all hidden, and the items are shuffled, so this sheet stays
blind. The recoverable mapping (the answer key) is stored separately and is not
part of this file._

_Two things are deliberately withheld from every item so the sheet cannot leak
origin and stays focused on Kreyòl: the hidden setup/system instruction, and
any internal `<think>…</think>` reasoning inside an assistant turn (these were
in English and are removed). You are judging the visible Kreyòl conversation
only. Very long single turns are truncated with `…[truncated]`._

## How to score each item

For every numbered item, do TWO things:

1. **Naturalness — score 1 to 5.** Judge whether the Kreyòl reads like language
   a fluent Haitian-Creole speaker would actually produce across the WHOLE
   conversation (both the user's request and the assistant's reply).
   - **1 = not Kreyòl / unusable** — word-salad, wrong language, or nonsense.
   - **2 = barely Kreyòl** — recognizably Kreyòl but heavily broken.
   - **3 = understandable but awkward** — you get the meaning, but there are
     clear errors, calques, or stiff/unnatural phrasing.
   - **4 = mostly natural** — reads well, only minor slips.
   - **5 = fully natural** — a fluent speaker could have written it.

   Write your number in the `____` slot after each item.

2. **Wrong / awkward flag — tick the checkbox** if ANY turn in the conversation
   is factually wrong, mistranslated, or awkward — even one bad turn. This is
   independent of the 1–5 score: an item can read fluently (high naturalness)
   yet still contain a factual error or a mistranslation worth flagging, so tick
   the box in that case too. Leave it unticked only if every turn is clean.

Take the items in order. If you are unsure, leave a short note next to the item.

---
""")


if __name__ == "__main__":
    main()
