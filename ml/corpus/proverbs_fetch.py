"""Fetch CreolePro's 50 Haitian Creole proverbs into the owned-proverbs file.

The proverbs are CreolePro's own content (rights.yaml owned_proverbs: owned,
redistributable), but per the repo data policy the extracted file lives under the
git-ignored `data/eval/`. This script is the committed, reproducible way to
(re)build it from the source blog — code is committed, data is not.

Source: config.PROVERBS_URL (a server-rendered page; each proverb is a numbered
`<strong>N. Kreyòl</strong>` with the English gloss in the following `<em>`).

Run:  python -m corpus.proverbs_fetch
"""

from __future__ import annotations

import json
import os
import re
import urllib.request

from . import common, config

_NUM_RE = re.compile(r"^\s*(\d+)\.\s*(.+)$", re.S)


def _download_html() -> str:
    req = urllib.request.Request(
        config.PROVERBS_URL,
        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", errors="replace")


def _extract(html: str):
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    art = soup.find("article") or soup
    out, current_cat = {}, None
    for node in art.descendants:
        name = getattr(node, "name", None)
        if name == "h2":
            current_cat = node.get_text(" ", strip=True)
        elif name == "strong":
            m = _NUM_RE.match(node.get_text(" ", strip=True))
            if not m:
                continue
            num = int(m.group(1))
            if not 1 <= num <= config.PROVERBS_EXPECTED_COUNT:
                continue
            p = node.find_parent(["p", "div"])
            em = p.find("em") if p else None
            out[num] = {
                "num": num,
                "kreyol": m.group(2).strip(),
                "english": em.get_text(" ", strip=True) if em else None,
                "category": current_cat,
            }
    return [out[k] for k in sorted(out)]


def fetch_proverbs() -> str:
    common.log(f"  fetching proverbs from {config.PROVERBS_URL}")
    proverbs = _extract(_download_html())
    if len(proverbs) != config.PROVERBS_EXPECTED_COUNT:
        raise RuntimeError(
            f"expected {config.PROVERBS_EXPECTED_COUNT} proverbs, got {len(proverbs)} "
            "— the blog markup may have changed; update corpus/proverbs_fetch.py")
    os.makedirs(os.path.dirname(config.PROVERBS_LOCAL), exist_ok=True)
    common.write_jsonl(config.PROVERBS_LOCAL, proverbs)
    common.log(f"  wrote {len(proverbs)} proverbs -> {config.PROVERBS_LOCAL}")
    return config.PROVERBS_LOCAL


def main():
    fetch_proverbs()


if __name__ == "__main__":
    main()
