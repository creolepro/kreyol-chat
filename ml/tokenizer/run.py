"""B2 orchestrator: samples → sweep → metrics → pick → export → tokenizer_v0.md.

Run:  cd ml && uv run python -m tokenizer.run
"""

from __future__ import annotations

import json
import os

from . import config, convert, data, external, report, sweep
from .core import KreyolBPE


def load_env():
    root = os.path.dirname(config.REPO_ROOT)
    path = os.path.join(root, ".env")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _dedup_keep_order(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def eval_tokenizer(kb: KreyolBPE, vocab: int, holdout_by_src: dict, flores: dict,
                   survival_words, core_words, want_robustness=False) -> dict:
    all_holdout = [t for texts in holdout_by_src.values() for t in texts]
    row = {
        "vocab": vocab,
        "embedding": sweep.embedding_cost(vocab),
        "compression_holdout": sweep.compression(kb, all_holdout),
        "compression_holdout_by_source": {
            src: sweep.compression(kb, texts) for src, texts in holdout_by_src.items()},
        "flores": {lang: sweep.compression(kb, sents) for lang, sents in flores.items()},
        "roundtrip": sweep.roundtrip_ok(kb, all_holdout),
        "survival_core": sweep.survival(kb, core_words),
        "survival_top": sweep.survival(kb, survival_words),
    }
    if want_robustness:
        row["robustness"] = sweep.robustness(kb)
    return row


def pick_vocab(rows, rel_gain_threshold=0.03):
    """Smallest vocab where the NEXT size improves holdout bytes/token by < threshold."""
    rows = sorted(rows, key=lambda r: r["vocab"])
    bpt = [r["compression_holdout"]["bytes_per_token"] for r in rows]
    chosen = rows[-1]["vocab"]
    reason = "largest (compression still improving materially)"
    for i in range(len(rows) - 1):
        gain = (bpt[i + 1] - bpt[i]) / bpt[i]
        if gain < rel_gain_threshold:
            chosen = rows[i]["vocab"]
            reason = (f"next size ({rows[i+1]['vocab']:,}) improves bytes/token by only "
                      f"{gain:.1%} (< {rel_gain_threshold:.0%}) — compression has flattened")
            break
    return chosen, reason


def run_ablation(kreyol_16k: KreyolBPE, nat_chars: int) -> dict:
    """English-control: train a 16k tokenizer on size-matched wikitext-103, then
    tokenize the TEACHABLE proverbs with both; show which words shatter."""
    wtext = external.load_wikitext_sample(nat_chars)
    eng = KreyolBPE.train(iter([wtext]), config.ENGLISH_ABLATION_VOCAB, config.CHOSEN_SPLIT_PATTERN)
    # teachable proverbs = proverbs.jsonl minus the probe set
    probe = data.probe_texts()
    proverbs_path = os.path.join(config.DATA, "eval", "proverbs.jsonl")
    teachable = []
    with open(proverbs_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            t = data.nfc(r["kreyol"]).strip()
            if t not in probe:
                teachable.append(t)
    data.assert_no_probe(teachable)
    kr_tok = sum(kreyol_16k.count(t) for t in teachable)
    en_tok = sum(eng.count(t) for t in teachable)
    # word-level: which proverb words survive under kreyol vs shatter under english
    words = _dedup_keep_order(w for t in teachable for w in t.split())
    shatter = []
    for w in words:
        kr = kreyol_16k.count(" " + w)
        en = eng.count(" " + w)
        if kr == 1 and en > 1:
            shatter.append({"word": w, "kreyol_tokens": kr, "english_tokens": en,
                            "english_pieces": [eng.decode([i]) for i in eng.encode_ordinary(" " + w)]})
    return {
        "n_teachable": len(teachable),
        "kreyol_total_tokens": kr_tok,
        "english_total_tokens": en_tok,
        "ratio_english_over_kreyol": round(en_tok / kr_tok, 3) if kr_tok else None,
        "n_words": len(words),
        "n_survive_kreyol_shatter_english": len(shatter),
        "examples": shatter[:25],
        "english_vocab": config.ENGLISH_ABLATION_VOCAB,
    }


def run():
    load_env()
    os.makedirs(config.WORK, exist_ok=True)
    data.log("=== tokenizer v0 sweep ===")

    # 1) training samples (seeded, weighted)
    nat = sweep.materialize_sample("natural", config.TRAIN_SAMPLE_CHARS, config.SAMPLE_SEED)
    sens = sweep.materialize_sample("sensitivity", config.TRAIN_SAMPLE_CHARS, config.SAMPLE_SEED)

    # 2) eval data (holdout + FLORES + top words) — computed once
    data.log("collecting holdout + top-words (corpus passes)")
    holdout = data.holdout_docs()
    holdout_by_src = {}
    for d in holdout:
        holdout_by_src.setdefault(d["source"], []).append(d["text"])
    flores = external.load_flores()
    top_words, _ = data.top_words(config.TOP_WORDS_N)
    survival_words = _dedup_keep_order([w for w in top_words] )
    core_words = config.CORE_WORDS

    # 3) sweep
    results = {"snapshot": config.SNAPSHOT_DATE, "nanochat_commit": config.NANOCHAT_COMMIT,
               "pattern_name": config.CHOSEN_PATTERN_NAME,
               "holdout_docs": len(holdout),
               "holdout_by_source": {s: len(v) for s, v in holdout_by_src.items()},
               "flores_n": {k: len(v) for k, v in flores.items()},
               "sample_natural": nat, "sample_sensitivity": sens,
               "top_words_n": len(top_words),
               "vocab_rows": [], "train_seconds": {}}
    trained = {}
    for vocab in config.VOCAB_SWEEP:
        info = sweep.train_one(nat["path"], vocab, "natural")
        results["train_seconds"][str(vocab)] = info["train_seconds"]
        kb = KreyolBPE.load_pkl(info["pkl"])
        trained[vocab] = kb
        row = eval_tokenizer(kb, vocab, holdout_by_src, flores, survival_words, core_words,
                             want_robustness=False)
        results["vocab_rows"].append(row)
        data.log(f"  eval {vocab}: bytes/tok(ht holdout)="
                 f"{row['compression_holdout']['bytes_per_token']:.3f} "
                 f"survival(core)={row['survival_core']['frac']:.2f}")

    # sensitivity 16k
    sinfo = sweep.train_one(sens["path"], 16384, "sensitivity")
    skb = KreyolBPE.load_pkl(sinfo["pkl"])
    results["sensitivity"] = eval_tokenizer(skb, 16384, holdout_by_src, flores,
                                            survival_words, core_words)
    results["sensitivity"]["train_seconds"] = sinfo["train_seconds"]

    # 4) pick + robustness on chosen
    chosen, reason = pick_vocab(results["vocab_rows"])
    results["chosen_vocab"] = chosen
    results["chosen_reason"] = reason
    results["chosen_robustness"] = sweep.robustness(trained[chosen])
    data.log(f"chosen vocab: {chosen} ({reason})")

    # 5) english-control ablation (16k vs 16k)
    results["ablation"] = run_ablation(trained[config.ENGLISH_ABLATION_VOCAB], nat["chars"])

    # 6) export chosen tokenizer (3 formats + pkl + meta) to committed artifacts
    kb = trained[chosen]
    os.makedirs(config.ARTIFACTS, exist_ok=True)
    convert.export_hf(kb, os.path.join(config.ARTIFACTS, "tokenizer.json"))
    convert.export_browser(kb, os.path.join(config.ARTIFACTS, "vocab_merges.json"),
                           os.path.join(config.ARTIFACTS, "tokenizer_tiktoken.json"))
    kb.save_pkl(os.path.join(config.ARTIFACTS, "tokenizer.pkl"))
    kb.save_meta(os.path.join(config.ARTIFACTS, "meta.json"),
                 extra={"chosen_vocab": chosen, "chosen_reason": reason,
                        "train_sample": {"weighting": "natural",
                                         "chars": nat["chars"],
                                         "composition_chars": nat["composition_chars"]}})
    # parity of the exported HF artifact (record in report)
    from tokenizers import Tokenizer
    hf = Tokenizer.from_file(os.path.join(config.ARTIFACTS, "tokenizer.json"))
    results["export_parity"] = convert.parity(kb, hf, data.probe_lines(per_source=500))

    # 7) persist results + render report
    with open(os.path.join(config.WORK, "sweep_results.json"), "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    report.write_report(results)
    data.log("=== done ===")
    return results


if __name__ == "__main__":
    run()
