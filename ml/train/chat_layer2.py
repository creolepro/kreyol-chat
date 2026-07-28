"""Workstream I Part 2 — Layer 2: corpus-grounded generation (the quality core).

Two generation modes over AUTHORED corpus v0.2.1 passages (VOA journalism, storybooks,
non-stub Wikipedia, Konstitisyon), teacher = claude-opus-4-8 (independently audited strong
at Kreyòl — never substitute a weaker model for the Kreyòl text):

  • reverse-instruction (MURI): write the natural Kreyòl instruction/question whose answer
    IS the passage → a single-turn {user, assistant} where the assistant answer stays the
    native corpus passage (lightly framed).
  • doc→dialogue (SEA-LION): outline → draft → self-critique → a short native Kreyòl
    multi-turn dialogue grounded in the passage, with a teacher self-score (1–5).

PILOT-GATED: `pilot` generates 100 conversations, measures token usage → cost, and REPORTS
the projected full-run cost. Only after that (and staying within the ≈$75 API budget) does
`full` run 3–5k. Seeds are drawn from the TRAINING docs (eval slices + tokenizer holdout
excluded) so grounding adds no eval leakage. Nothing here is committed (ml/data/, git-ignored).

Run:  uv run python -m train.chat_layer2 pilot            # 100-conv pilot + cost projection
      uv run python -m train.chat_layer2 full --target 4000 --budget 75
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request

from . import chat_config as C
from . import config as F

sys.path.insert(0, F.REPO_ROOT)

# --- teacher (claude-opus-4-8) + pricing --------------------------------------
TEACHER_MODEL = "claude-opus-4-8"
API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"
# List prices for the Opus tier ($/MTok); the API dashboard is the source of truth. The
# pilot MEASURES actual token usage, so the $ figure is tokens×rate, not a guess.
PRICE_IN_PER_MTOK = 15.0
PRICE_OUT_PER_MTOK = 75.0

# passage sizing + selection
PASSAGE_MIN_CHARS = 320
PASSAGE_MAX_CHARS = 1200
# register weights for seed selection (VOA/storybooks/legal are rare + high-value → take all;
# cap Wikipedia so the authored voice isn't 97% encyclopedic)
AUTHORED_SOURCES = {"voa_nouvel", "storybooks_haiti", "konstitisyon_1987", "ht_wikipedia"}
WIKI_MAX_FRAC = 0.55             # ≤55% of Layer-2 seeds from Wikipedia
MODE_SPLIT = 0.5                 # fraction reverse-instruction (rest doc→dialogue)
MAX_TOKENS_OUT = 1400
WORKERS = 6


def _log(m):
    print(f"[layer2] {m}", flush=True)


def _api_key():
    d = F.REPO_ROOT
    for _ in range(5):
        cand = os.path.join(d, ".env")
        if os.path.exists(cand):
            for line in open(cand, encoding="utf-8"):
                if line.strip().startswith("ANTHROPIC_API_KEY="):
                    return line.strip().split("=", 1)[1].strip().strip('"').strip("'")
        d = os.path.dirname(d)
    return os.environ.get("ANTHROPIC_API_KEY")


# ============================ passage selection ===============================

def _paragraphs(text: str):
    parts = re.split(r"\n\s*\n", text)
    if len(parts) == 1:
        parts = [p for p in text.split("\n") if p.strip()]
    return [p.strip() for p in parts if p.strip()]


def _chunk(text: str):
    """Greedy paragraph accumulation into PASSAGE_MIN..MAX-char passages."""
    out, buf = [], ""
    for p in _paragraphs(text):
        if len(buf) + len(p) + 1 <= PASSAGE_MAX_CHARS:
            buf = (buf + "\n" + p).strip()
        else:
            if len(buf) >= PASSAGE_MIN_CHARS:
                out.append(buf)
            buf = p[:PASSAGE_MAX_CHARS]
        if len(buf) >= PASSAGE_MAX_CHARS:
            out.append(buf[:PASSAGE_MAX_CHARS])
            buf = ""
    if len(buf) >= PASSAGE_MIN_CHARS:
        out.append(buf)
    return out


def _excluded_ids():
    """Eval-slice + tokenizer-holdout doc_ids — the SAME exclusion the base model used, so
    Layer-2 grounding never turns a held-out passage into a training example."""
    from . import tokenize_g as T
    ex = T._excluded_slice_ids(include_v2=True)
    return ex


def _in_holdout(doc_id: str) -> bool:
    from . import tokenize_g as T
    return T._in_tokenizer_holdout(doc_id)


def select_passages(target_n: int, seed: int) -> list:
    """Build a register-balanced pool of authored passages with provenance."""
    from . import chat_data as D            # reuse probe screen + norm
    corpus = F.CORPUS_V0_2_1.format(tag="full")
    exclude = _excluded_ids()
    rng = random.Random(seed)
    by_src = {s: [] for s in AUTHORED_SOURCES}
    for line in open(corpus, encoding="utf-8"):
        d = json.loads(line)
        src = d.get("acquisition", {}).get("source")
        if src not in AUTHORED_SOURCES:
            continue
        if src == "ht_wikipedia" and d.get("wiki_bot_stub"):
            continue
        did = d.get("acquisition", {}).get("doc_id")
        if did in exclude or _in_holdout(did):
            continue
        reg = d.get("register")
        for i, ch in enumerate(_chunk(d["text"])):
            if D._has_probe(ch):
                continue
            by_src[src].append({"passage": ch, "doc_id": did, "source": src, "register": reg,
                                "pid": hashlib.sha256(f"{did}:{i}".encode()).hexdigest()[:16]})
    for s in by_src:
        rng.shuffle(by_src[s])
    # take all of the rare authored sources first; fill the rest from Wikipedia (capped)
    rare = [p for s in ("voa_nouvel", "storybooks_haiti", "konstitisyon_1987") for p in by_src[s]]
    wiki = by_src["ht_wikipedia"]
    rng.shuffle(rare)
    wiki_cap = int(target_n * WIKI_MAX_FRAC)
    chosen = rare[: target_n - min(len(wiki), wiki_cap)] + wiki[:wiki_cap]
    if len(chosen) < target_n:                      # backfill from whatever remains
        extra = rare[len(chosen):] + wiki[wiki_cap:]
        chosen += extra[: target_n - len(chosen)]
    rng.shuffle(chosen)
    chosen = chosen[:target_n]
    for i, p in enumerate(chosen):
        p["mode"] = "reverse_instruction" if (i / max(1, len(chosen))) < MODE_SPLIT else "doc_dialogue"
    _log(f"selected {len(chosen)} passages "
         f"(rare={sum(1 for p in chosen if p['source']!='ht_wikipedia')}, "
         f"wiki={sum(1 for p in chosen if p['source']=='ht_wikipedia')})")
    return chosen


# ============================ teacher generation ==============================

_SYS = (
    "Ou se yon ekspè nan lang kreyòl ayisyen. Ou ap ede kreye done antrènman pou yon ti "
    "modèl chat ki pale kreyòl. Tout tèks ou pwodui DWE an kreyòl ayisyen natirèl epi kòrèk "
    "(sof si yon tradiksyon mande espesifikman). Reponn SÈLMAN ak yon objè JSON valab, san "
    "okenn lòt tèks."
)

_REVERSE_PROMPT = (
    "Men yon pasaj otantik an kreyòl ki soti nan yon koperasyon tèks kreyòl:\n\n"
    "<<<PASAJ>>>\n{passage}\n<<<FEN PASAJ>>>\n\n"
    "Ekri YON sèl enstriksyon oswa kesyon natirèl an kreyòl ke yon moun ta poze, epi ki "
    "repons li se pasaj sa a (oswa yon vèsyon byen ekri pasaj sa a). Enstriksyon an dwe "
    "otonòm (pa refere a «pasaj la»). Bay tou yon bon repons an kreyòl ki baze SÈLMAN "
    "sou enfòmasyon nan pasaj la.\n\n"
    "Retounen JSON konsa: {{\"instruction\": \"...\", \"answer\": \"...\", \"self_score\": 1-5}}"
)

_DIALOGUE_PROMPT = (
    "Men yon pasaj otantik an kreyòl:\n\n<<<PASAJ>>>\n{passage}\n<<<FEN PASAJ>>>\n\n"
    "Fè travay sa a nan tèt ou (pa montre etap yo): (1) trase yon plan pou yon ti "
    "konvèsasyon 2 a 3 vire ki chita sou enfòmasyon nan pasaj la; (2) ekri yon bouyon; "
    "(3) kritike bouyon an — èske kreyòl la natirèl? èske li fidèl ak pasaj la? — epi korije. "
    "Konvèsasyon final la dwe an kreyòl natirèl ant yon itilizatè ak yon asistan, epi rete "
    "fidèl ak pasaj la (pa envante fè). ENPÒTAN: ni itilizatè a ni asistan an pa DWE janm "
    "refere a «pasaj la», «tèks la», oswa «dokiman an» — konvèsasyon an dwe kanpe pou kont li, "
    "kòm si se yon vrè chat san okenn pasaj devan je moun nan.\n\n"
    "Retounen JSON konsa: {{\"conversation\": [{{\"role\": \"user\", \"content\": \"...\"}}, "
    "{{\"role\": \"assistant\", \"content\": \"...\"}}], \"self_score\": 1-5, \"critique\": \"...\"}}"
)


def _post(payload: dict, key: str, timeout=120):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(API_URL, data=data, method="POST", headers={
        "content-type": "application/json", "x-api-key": key,
        "anthropic-version": API_VERSION})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _call_teacher(prompt: str, key: str, max_tokens=MAX_TOKENS_OUT, retries=4):
    payload = {"model": TEACHER_MODEL, "max_tokens": max_tokens, "system": _SYS,
               "messages": [{"role": "user", "content": prompt}]}
    for attempt in range(retries):
        try:
            resp = _post(payload, key)
            text = "".join(b.get("text", "") for b in resp.get("content", []))
            usage = resp.get("usage", {})
            return {"text": text, "in_tok": usage.get("input_tokens", 0),
                    "out_tok": usage.get("output_tokens", 0)}
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "ignore")[:200]
            if e.code in (429, 500, 503, 529) and attempt < retries - 1:
                time.sleep(2 ** attempt * 3)
                continue
            return {"error": f"HTTP {e.code}: {body}"}
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt * 3)
                continue
            return {"error": f"{type(e).__name__}: {e}"[:200]}


def _parse_json(text: str):
    """Extract the first JSON object from the teacher output (tolerant of stray prose/fences)."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(json)?|```$", "", text, flags=re.MULTILINE).strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def generate_one(passage: dict, key: str) -> dict:
    from . import chat_data as D
    mode = passage["mode"]
    prompt = (_REVERSE_PROMPT if mode == "reverse_instruction" else _DIALOGUE_PROMPT).format(
        passage=passage["passage"])
    r = _call_teacher(prompt, key)
    if "error" in r:
        return {"pid": passage["pid"], "error": r["error"], "in_tok": 0, "out_tok": 0}
    obj = _parse_json(r["text"])
    if not obj:
        return {"pid": passage["pid"], "error": "unparseable", "in_tok": r["in_tok"],
                "out_tok": r["out_tok"]}
    if mode == "reverse_instruction":
        u, a = str(obj.get("instruction", "")).strip(), str(obj.get("answer", "")).strip()
        if not u or not a:
            return {"pid": passage["pid"], "error": "empty", "in_tok": r["in_tok"], "out_tok": r["out_tok"]}
        msgs = [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    else:
        conv = obj.get("conversation") or []
        msgs = [{"role": ("assistant" if m.get("role") == "assistant" else "user"),
                 "content": str(m.get("content", "")).strip()} for m in conv if str(m.get("content", "")).strip()]
        if len(msgs) < 2 or msgs[0]["role"] != "user" or not any(m["role"] == "assistant" for m in msgs):
            return {"pid": passage["pid"], "error": "bad_conversation", "in_tok": r["in_tok"], "out_tok": r["out_tok"]}
    # standing rule: no probe proverb may enter training, even generated
    if any(D._has_probe(m["content"]) for m in msgs):
        return {"pid": passage["pid"], "error": "probe_leak", "in_tok": r["in_tok"], "out_tok": r["out_tok"]}
    return {"pid": passage["pid"], "in_tok": r["in_tok"], "out_tok": r["out_tok"],
            "record": {"messages": msgs, "source": f"layer2:{mode}", "layer": 2,
                       "meta": {"mode": mode, "seed_source": passage["source"],
                                "seed_register": passage["register"], "seed_doc_id": passage["doc_id"],
                                "self_score": obj.get("self_score"), "pid": passage["pid"]}}}


# ============================ runners =========================================

def _cost(in_tok, out_tok):
    return in_tok / 1e6 * PRICE_IN_PER_MTOK + out_tok / 1e6 * PRICE_OUT_PER_MTOK


def _run_batch(passages, key, out_path, budget=None, done_pids=None):
    """Threaded generation with incremental append (resumable) + running cost/budget guard."""
    done_pids = done_pids or set()
    todo = [p for p in passages if p["pid"] not in done_pids]
    agg = {"n_ok": 0, "n_err": 0, "in_tok": 0, "out_tok": 0, "errors": {}}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fh = open(out_path, "a", encoding="utf-8")
    stop = False
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(generate_one, p, key): p for p in todo}
        try:
            for fut in cf.as_completed(futs):
                res = fut.result()
                agg["in_tok"] += res.get("in_tok", 0)
                agg["out_tok"] += res.get("out_tok", 0)
                if "record" in res:
                    fh.write(json.dumps(res["record"], ensure_ascii=False) + "\n"); fh.flush()
                    agg["n_ok"] += 1
                else:
                    agg["n_err"] += 1
                    agg["errors"][res.get("error", "?")[:40]] = agg["errors"].get(res.get("error", "?")[:40], 0) + 1
                if (agg["n_ok"] + agg["n_err"]) % 25 == 0:
                    _log(f"{agg['n_ok']} ok / {agg['n_err']} err | "
                         f"cost so far ${_cost(agg['in_tok'], agg['out_tok']):.2f}")
                if budget is not None and _cost(agg["in_tok"], agg["out_tok"]) >= budget:
                    _log(f"BUDGET ${budget} reached — stopping"); stop = True
                    break
        finally:
            if stop:
                for f2 in futs:
                    f2.cancel()
            fh.close()
    agg["cost_usd"] = round(_cost(agg["in_tok"], agg["out_tok"]), 2)
    return agg


def run_pilot(n=100, seed=C.DL_SEED) -> dict:
    key = _api_key()
    assert key, "no ANTHROPIC_API_KEY in .env"
    if os.path.exists(C.LAYER2_PILOT):
        os.remove(C.LAYER2_PILOT)
    passages = select_passages(n, seed)
    t0 = time.time()
    agg = _run_batch(passages, key, C.LAYER2_PILOT)
    dt = time.time() - t0
    per = agg["n_ok"] or 1
    proj = {
        "pilot_n_requested": n, "pilot_n_ok": agg["n_ok"], "pilot_n_err": agg["n_err"],
        "pilot_errors": agg["errors"], "pilot_in_tok": agg["in_tok"], "pilot_out_tok": agg["out_tok"],
        "pilot_cost_usd": agg["cost_usd"], "pilot_seconds": round(dt, 1),
        "price_in_per_mtok": PRICE_IN_PER_MTOK, "price_out_per_mtok": PRICE_OUT_PER_MTOK,
        "cost_per_ok_conversation": round(agg["cost_usd"] / per, 4),
        "in_tok_per_conv": round(agg["in_tok"] / per), "out_tok_per_conv": round(agg["out_tok"] / per),
        "projected_cost_3000": round(agg["cost_usd"] / per * 3000, 2),
        "projected_cost_4000": round(agg["cost_usd"] / per * 4000, 2),
        "projected_cost_5000": round(agg["cost_usd"] / per * 5000, 2),
        "ok_rate": round(agg["n_ok"] / max(1, n), 3),
    }
    os.makedirs(C.CHAT_WORK, exist_ok=True)
    with open(os.path.join(C.CHAT_WORK, "layer2_pilot_report.json"), "w") as fh:
        json.dump(proj, fh, indent=2, ensure_ascii=False)
    _log("PILOT COST REPORT:\n" + json.dumps(proj, indent=2))
    return proj


def run_full(target=4000, budget=75.0, seed=C.DL_SEED + 7) -> dict:
    key = _api_key()
    assert key, "no ANTHROPIC_API_KEY in .env"
    done = set()
    # skip pids already generated in the full file AND in the pilot (no duplicate passages)
    for path in (C.LAYER2_GEN, C.LAYER2_PILOT):
        if os.path.exists(path):
            for l in open(path, encoding="utf-8"):
                try:
                    done.add(json.loads(l)["meta"]["pid"])
                except Exception:
                    pass
    _log(f"resuming: {len(done)} pids already generated (full + pilot)")
    passages = select_passages(target, seed)
    agg = _run_batch(passages, key, C.LAYER2_GEN, budget=budget, done_pids=done)
    total = len(done) + agg["n_ok"]
    out = {"target": target, "budget_usd": budget, "new_ok": agg["n_ok"], "new_err": agg["n_err"],
           "errors": agg["errors"], "total_generated": total, "batch_cost_usd": agg["cost_usd"],
           "in_tok": agg["in_tok"], "out_tok": agg["out_tok"]}
    with open(os.path.join(C.CHAT_WORK, "layer2_full_report.json"), "w") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    _log("FULL RUN:\n" + json.dumps(out, indent=2))
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pilot"); p.add_argument("--n", type=int, default=100)
    fu = sub.add_parser("full")
    fu.add_argument("--target", type=int, default=4000)
    fu.add_argument("--budget", type=float, default=75.0)
    args = ap.parse_args()
    os.makedirs(C.CHAT_RAW, exist_ok=True)
    os.makedirs(C.CHAT_WORK, exist_ok=True)
    if args.cmd == "pilot":
        run_pilot(n=args.n)
    elif args.cmd == "full":
        run_full(target=args.target, budget=args.budget)


if __name__ == "__main__":
    main()
