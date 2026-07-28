"""Workstream I v1.1 patch — generate short, informal Kreyòl chat conversations.

Closes two coupled SFT failures: (1) ultra-short informal inputs ("sak pase?") triggering
ht.wikipedia bot-stub completions, and (2) successful replies trailing into stub boilerplate,
because the SFT set skews long so clean termination after 1–3 sentences is undertrained.

Uses the Anthropic **Batch API** (mandatory for the bulk — 50% cheaper). Seed grid: informal
openers × contexts × 1–4 turns, with hard STYLE CONSTRAINTS (1–2 sentence turns, everyday
register, lowercase/punctuation-light user forms, clean termination, no lists/sections unless
asked, a meaningful share of complete 1-turn exchanges). Native Kreyòl (NOT translation),
teacher self-critique with rejection, probe-proverb screen, passage-reference drop.

Provenance: origin `synthetic_unreviewed` per docs/plan.md §5.3 — native review is REQUIRED
before this data or a model trained on it is publicly released; a 50-item blinded review sample
is written alongside. Also mines the kept kakugo set for existing short informal exchanges and
upweights them. Nothing under ml/data/ is committed.

Run:  uv run python -m train.chat_informal generate --n 460      # submit + collect the batch
      uv run python -m train.chat_informal collect --batch <id>   # resume collection
      uv run python -m train.chat_informal mine                   # kakugo short-informal mining
      uv run python -m train.chat_informal sheet                  # build the 50-item review sample
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
import urllib.error
import urllib.request

from . import chat_config as C
from . import config as F
from .chat_layer2 import _api_key, _parse_json, TEACHER_MODEL

BATCH_URL = "https://api.anthropic.com/v1/messages/batches"
API_VERSION = "2023-06-01"
# Batch API list rates for Opus (50% of standard $15/$75) — the dashboard is the true bill.
BATCH_IN_PER_MTOK = 7.5
BATCH_OUT_PER_MTOK = 37.5

INFORMAL_GEN = C.INFORMAL_GENERATED
INFORMAL_MINED = C.INFORMAL_MINED
BATCH_STATE = os.path.join(C.CHAT_WORK, "informal_batch_state.json")

# --- seed grid ----------------------------------------------------------------
CONTEXTS = [
    "de zanmi k ap voye tèks youn ak lòt",
    "yon apèl telefòn ak yon manm fanmi",
    "yon ti pale nan mache a",
    "apre legliz, de moun k ap salye",
    "de kòlèg nan travay",
    "yon moun nan dyaspora k ap rele fanmi l ann Ayiti",
]
# informal opener hints (short, how people actually type — mixed lowercase / punctuation-light)
OPENERS = [
    "sak pase", "sa k ap fèt", "kòman ou ye", "ou la", "sak gen la a", "ki nouvèl",
    "banm nouvèl ou", "alo", "bonjou wi", "bonswa", "eske ou byen", "sa w ap fè la",
    "kote ou ye", "map tann ou", "mèsi anpil", "n a wè pita", "ki jan fanmi an ye",
    "gen lontan m pa wè w", "sa k pase ak ou", "ann pale non",
]


def _log(m):
    print(f"[informal] {m}", flush=True)


def build_seeds(n, seed=C.DL_SEED):
    rng = random.Random(seed)
    seeds = []
    for i in range(n):
        ctx = rng.choice(CONTEXTS)
        # weight toward short: 1 turn 45%, 2 turns 30%, 3 turns 15%, 4 turns 10%
        n_turns = rng.choices([1, 2, 3, 4], weights=[45, 30, 15, 10])[0]
        opener = rng.choice(OPENERS)
        seeds.append({"custom_id": f"inf-{i}", "context": ctx, "n_turns": n_turns,
                      "opener": opener, "one_turn": n_turns == 1})
    return seeds


_SYS = (
    "Ou se yon moun ki pale kreyòl ayisyen natirèl chak jou. W ap ede kreye ti konvèsasyon "
    "enfòmèl otantik pou antrene yon ti asistan chat ki pale kreyòl. Tout tèks DWE an kreyòl "
    "ayisyen natirèl (PA tradiksyon). Reponn SÈLMAN ak yon objè JSON valab."
)


def _prompt(seed):
    one = ("Sa a se yon echanj konplè yon SÈL vire: itilizatè a di yon bagay kout, asistan an "
           "reponn kout epi konvèsasyon an fini la (pa gen lòt vire)."
           if seed["one_turn"] else
           f"Konvèsasyon an gen {seed['n_turns']} vire (yon vire = yon mesaj itilizatè + yon repons).")
    return (
        f"Ekri YON ti konvèsasyon enfòmèl an kreyòl ant yon itilizatè ak yon asistan.\n"
        f"Kontèks: {seed['context']}.\n{one}\n"
        f"Premye mesaj itilizatè a dwe kout epi enfòmèl, nan estil sa a: «{seed['opener']}» "
        f"(ou ka chanje l yon ti kras, men kenbe l kout epi enfòmèl; li ka an lèt minisk san "
        f"anpil pwenktiyasyon, jan moun tape nan mesaj tèks).\n\n"
        f"RÈG STIL (trè enpòtan — se pwen egzèsis la):\n"
        f"- Chak mesaj KOUT: 1 a 2 fraz sèlman. PA gen lis ak nimewo, PA gen tit seksyon, "
        f"PA gen 'Referans', 'Istwa', 'Lyen', ni okenn background style ansiklopedi.\n"
        f"- Rejis chak jou / pale jan moun pale vre.\n"
        f"- Repons asistan an dwe reponn dirèkteman epi FÈMEN pwòp — li pa dwe kontinye ajoute "
        f"enfòmasyon yo pa mande. Lè li fin reponn, li kanpe.\n"
        f"- Pa janm refere a okenn «pasaj», «tèks», oswa «dokiman».\n\n"
        f"Apre ou fin ekri, tcheke tèt ou: èske kreyòl la natirèl? èske chak mesaj kout epi li "
        f"fèmen pwòp san trennen? Si non, korije anvan ou bay repons final la.\n\n"
        f"Retounen JSON konsa: {{\"conversation\": [{{\"role\": \"user\", \"content\": \"...\"}}, "
        f"{{\"role\": \"assistant\", \"content\": \"...\"}}], \"self_score\": 1-5}}"
    )


def _req(seed):
    return {"custom_id": seed["custom_id"], "params": {
        "model": TEACHER_MODEL, "max_tokens": 700, "system": _SYS,
        "messages": [{"role": "user", "content": _prompt(seed)}]}}


# --- Batch API client (urllib) ------------------------------------------------

def _api(method, url, key, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "content-type": "application/json", "x-api-key": key, "anthropic-version": API_VERSION})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


def submit_batch(seeds, key):
    body = {"requests": [_req(s) for s in seeds]}
    resp = _api("POST", BATCH_URL, key, body)
    bid = resp["id"]
    with open(BATCH_STATE, "w") as fh:
        json.dump({"batch_id": bid, "n": len(seeds), "seeds": {s["custom_id"]: s for s in seeds}}, fh)
    _log(f"submitted batch {bid} ({len(seeds)} requests)")
    return bid


def poll_batch(bid, key, interval=20, max_wait=3600):
    t0 = time.time()
    while time.time() - t0 < max_wait:
        b = _api("GET", f"{BATCH_URL}/{bid}", key)
        st = b.get("processing_status")
        counts = b.get("request_counts", {})
        _log(f"batch {st}: {counts}")
        if st == "ended":
            return b
        time.sleep(interval)
    raise TimeoutError(f"batch {bid} not ended after {max_wait}s")


def _fetch_results(results_url, key):
    req = urllib.request.Request(results_url, headers={
        "x-api-key": key, "anthropic-version": API_VERSION})
    with urllib.request.urlopen(req, timeout=300) as r:
        text = r.read().decode()
    return [json.loads(l) for l in text.splitlines() if l.strip()]


def collect_batch(bid, key):
    from . import chat_data as D
    b = poll_batch(bid, key)
    results = _fetch_results(b["results_url"], key)
    seeds = json.load(open(BATCH_STATE))["seeds"] if os.path.exists(BATCH_STATE) else {}
    stats = {"results": len(results), "errored": 0, "unparseable": 0, "bad_conv": 0,
             "low_score": 0, "passage_ref": 0, "probe_leak": 0, "kept": 0,
             "in_tok": 0, "out_tok": 0, "n_one_turn": 0}
    _ref = re.compile(r"\b(dapre|selon|nan)\s+(pasaj|tèks|dokiman)|pasaj\s+(la|sa)|tèks\s+(la|sa)", re.I)
    out = []
    for r in results:
        res = r.get("result", {})
        if res.get("type") != "succeeded":
            stats["errored"] += 1
            continue
        msg = res["message"]
        u = msg.get("usage", {})
        stats["in_tok"] += u.get("input_tokens", 0)
        stats["out_tok"] += u.get("output_tokens", 0)
        text = "".join(b.get("text", "") for b in msg.get("content", []))
        obj = _parse_json(text)
        if not obj:
            stats["unparseable"] += 1
            continue
        conv = obj.get("conversation") or []
        msgs = [{"role": ("assistant" if m.get("role") == "assistant" else "user"),
                 "content": D._norm(str(m.get("content", "")))} for m in conv if str(m.get("content", "")).strip()]
        if len(msgs) < 2 or msgs[0]["role"] != "user" or not any(m["role"] == "assistant" for m in msgs):
            stats["bad_conv"] += 1
            continue
        if (obj.get("self_score") or 5) < 4:
            stats["low_score"] += 1
            continue
        if any(_ref.search(m["content"]) for m in msgs):
            stats["passage_ref"] += 1
            continue
        if any(D._has_probe(m["content"]) for m in msgs):
            stats["probe_leak"] += 1
            continue
        seed = seeds.get(r["custom_id"], {})
        if seed.get("one_turn"):
            stats["n_one_turn"] += 1
        out.append({"messages": msgs, "source": "informal_synth", "layer": 3,
                    "origin": "synthetic_unreviewed",
                    "meta": {"context": seed.get("context"), "n_turns": seed.get("n_turns"),
                             "one_turn": seed.get("one_turn"), "self_score": obj.get("self_score"),
                             "custom_id": r["custom_id"]}})
    os.makedirs(C.CHAT_RAW, exist_ok=True)
    with open(INFORMAL_GEN, "w", encoding="utf-8") as fh:
        for rec in out:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    stats["kept"] = len(out)
    stats["cost_usd"] = round(stats["in_tok"] / 1e6 * BATCH_IN_PER_MTOK
                              + stats["out_tok"] / 1e6 * BATCH_OUT_PER_MTOK, 2)
    with open(os.path.join(C.CHAT_WORK, "informal_gen_report.json"), "w") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)
    _log(f"collected: {json.dumps(stats)}")
    return stats


def generate(n=460):
    key = _api_key()
    assert key, "no ANTHROPIC_API_KEY in .env"
    os.makedirs(C.CHAT_WORK, exist_ok=True)
    seeds = build_seeds(n)
    bid = submit_batch(seeds, key)
    return collect_batch(bid, key)


# --- mine kakugo for short informal exchanges (upweight into the mix) ----------

def mine_kakugo(max_first_chars=70, max_turns=4, cap=1500):
    """Pull short conversations from the kept kakugo set (short first user turn = informal-ish),
    to upweight everyday short exchanges. Tagged distinctly so the report can show the split."""
    if not os.path.exists(C.KAKUGO_CLEAN):
        _log("no kakugo_clean.jsonl — run chat_data download first")
        return {"kept": 0}
    out = []
    for line in open(C.KAKUGO_CLEAN, encoding="utf-8"):
        d = json.loads(line)
        msgs = d["messages"]
        if len(msgs) > max_turns * 2:
            continue
        if msgs[0]["role"] != "user" or len(msgs[0]["content"]) > max_first_chars:
            continue
        # also require the assistant reply to be reasonably short (informal, not an essay)
        asst = [m for m in msgs if m["role"] == "assistant"]
        if not asst or max(len(a["content"]) for a in asst) > 400:
            continue
        out.append({"messages": msgs, "source": "kakugo_short", "layer": 3,
                    "origin": "synthetic_unreviewed", "meta": {"mined": "kakugo_short"}})
    random.Random(C.DL_SEED + 3).shuffle(out)
    out = out[:cap]
    os.makedirs(C.CHAT_RAW, exist_ok=True)
    with open(INFORMAL_MINED, "w", encoding="utf-8") as fh:
        for rec in out:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    _log(f"mined {len(out)} short kakugo exchanges")
    return {"kept": len(out)}


# --- 50-item blinded review sample (native review required per §5.3) ----------

def build_review_sheet(n=50):
    import random as _r
    gen = [json.loads(l) for l in open(INFORMAL_GEN, encoding="utf-8")] if os.path.exists(INFORMAL_GEN) else []
    if not gen:
        _log("no informal_generated.jsonl — nothing to review")
        return
    rng = _r.Random(20260728)
    # stratify by turn count so short + longer both get eyes
    by_t = {}
    for r in gen:
        by_t.setdefault(r["meta"].get("n_turns", 1), []).append(r)
    sample = []
    for t in sorted(by_t):
        rng.shuffle(by_t[t])
    # round-robin across turn buckets up to n
    buckets = [by_t[t] for t in sorted(by_t)]
    i = 0
    while len(sample) < min(n, len(gen)):
        b = buckets[i % len(buckets)]
        if b:
            sample.append(b.pop())
        i += 1
        if all(not b for b in buckets):
            break
    rng.shuffle(sample)
    L = ["# Blinded naturalness review — informal Kreyòl chat (SFT v1.1 patch)", "",
         "> **For a native Haitian-Creole reviewer.** These are **synthetic** informal",
         "> conversations (`synthetic_unreviewed`) generated for the v1.1 patch. Per the",
         "> project's synthetic-data policy, this batch may not ship publicly until a native",
         "> reviewer signs off. Judge the KREYÒL and the register only; order is shuffled.", "",
         "## How to score each item", "",
         "**(1) Naturalness 1–5** (does it read like real everyday Kreyòl texting/talk?) → `____`;",
         "**(2)** tick the box if a turn is unnatural, too long/formal, wrong, or trails into",
         "boilerplate. Short + clean is the goal here.", "", "---", ""]
    for i, r in enumerate(sample, 1):
        L += [f"## Item {i}", ""]
        for m in r["messages"]:
            who = "User" if m["role"] == "user" else "Assistant"
            L += [f"**{who}:**", f"> {m['content'].replace(chr(10), ' ')}", ""]
        L += ["_naturalness (1–5):_ `____`  ",
              "- [ ] **unnatural / too long / trails off** — tick if any turn is off", ""]
    path = os.path.join(F.REPO_ROOT, "reports", "informal_audit_sheet.md")
    open(path, "w", encoding="utf-8").write("\n".join(L))
    _log(f"wrote {path} ({len(sample)} items)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate"); g.add_argument("--n", type=int, default=460)
    c = sub.add_parser("collect"); c.add_argument("--batch", type=str, required=True)
    sub.add_parser("mine")
    sub.add_parser("sheet")
    args = ap.parse_args()
    os.makedirs(C.CHAT_RAW, exist_ok=True)
    os.makedirs(C.CHAT_WORK, exist_ok=True)
    if args.cmd == "generate":
        generate(n=args.n)
        mine_kakugo()
        build_review_sheet()
    elif args.cmd == "collect":
        collect_batch(args.batch, _api_key())
        build_review_sheet()
    elif args.cmd == "mine":
        mine_kakugo()
    elif args.cmd == "sheet":
        build_review_sheet()


if __name__ == "__main__":
    main()
