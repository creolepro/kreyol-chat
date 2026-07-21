"""External corpora for B2 evaluation: FLORES+ (compression / regression) and
wikitext-103 (the English-control ablation).

FLORES+ is eval-only and never re-hosted (reuses Workstream C's pinned loader).
wikitext-103 is CC-BY-SA 3.0 (noted in the report); used only to train one
control tokenizer, never committed.
"""

from __future__ import annotations

import os

from . import config, data


def load_flores():
    """{'ht':[...], 'en':[...], 'fr':[...]} aligned devtest sentences (NFC)."""
    from fertility.data_sources import load_flores as _lf
    res = _lf(os.environ.get("HF_TOKEN"), data.log)
    texts = res["texts"]  # {'ht','en','fr': [sentences]}
    return {k: [data.nfc(s) for s in v] for k, v in texts.items()}


def load_wikitext_sample(target_chars: int):
    """~target_chars of English wikitext-103-raw train text (CC-BY-SA 3.0)."""
    cache = os.path.join(config.WORK, "wikitext103_sample.txt")
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:
            return f.read()
    from datasets import load_dataset
    data.log(f"  loading {config.WIKITEXT_REPO}/{config.WIKITEXT_CONFIG} (streaming)")
    ds = load_dataset(config.WIKITEXT_REPO, config.WIKITEXT_CONFIG,
                      split=config.WIKITEXT_SPLIT, streaming=True,
                      token=os.environ.get("HF_TOKEN"))
    buf = []
    n = 0
    for row in ds:
        t = row.get("text", "")
        if not t.strip():
            continue
        buf.append(t)
        n += len(t)
        if n >= target_chars:
            break
    text = data.nfc("".join(buf))[:target_chars]
    os.makedirs(config.WORK, exist_ok=True)
    with open(cache, "w", encoding="utf-8") as f:
        f.write(text)
    data.log(f"  wikitext sample: {len(text)/1e6:.1f}M chars")
    return text
