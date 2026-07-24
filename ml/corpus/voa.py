"""VOA Nouvèl (voanouvel.com) — sitemap enumeration, polite fetch, extraction.

Workstream J (corpus v0.2). Reusable core shared by the J0 scoping sample and
the J1 full crawl.

Rights: "All text, audio and video material produced exclusively by the Voice of
America is public domain" (17 U.S.C. §105 — US federal work). CARVE-OUT: licensed
AFP/AP/Reuters material embedded in articles is NOT PD and is filtered here (drop
wire-bylined articles entirely; strip wire-credited paragraphs elsewhere; when in
doubt, drop). See rights.yaml key `voa_nouvel`.

Crawl ethics (docs/data.md CRAWLING ETHICS, binding):
  * robots.txt (fetched 2026-07-23): `/a/` article paths are ALLOWED; the
    Disallow list covers /z/, /tv/, /radio/, /schedule/, search, comments, embeds,
    podcast sublinks — none of which we fetch. No Crawl-delay directive → we
    self-impose ~1 req/s (RATE_S) with exponential backoff on 429/5xx.
  * Honest identification: UA carries the project URL + contact.
  * Cloudflare-fronted: a naive fetch 403s; normal browser headers suffice. We do
    NOT go further (no CAPTCHA solving, no proxy rotation). Blocked => report.

Extraction (verified against live 2026-07 HTML):
  * metadata     — JSON-LD NewsArticle (datePublished/dateModified/author/keywords)
  * lede         — div.intro   (bold standfirst, full untruncated)
  * body         — div.wsw     (clean authored prose <p> paragraphs; excludes the
                   related-content / toolbar / categories chrome)
  * article gate — a real text article has a div.wsw with >= MIN_BODY_CHARS of
                   prose. Audio/video "program" pages (Bonjou Ayiti, VOA Direct)
                   have no wsw and are dropped (is_article=False).
"""

from __future__ import annotations

import gzip
import json
import re
import time

import requests
from bs4 import BeautifulSoup

SITE = "https://www.voanouvel.com"
SITEMAP_INDEX = f"{SITE}/sitemap.xml"
ARTICLE_SHARDS = ["/sitemap_413_1.xml.gz", "/sitemap_413_2.xml.gz"]

UA = ("kreyol-chat/0.2 (+https://github.com/creolepro/kreyol-chat; "
      "patricedouge@gmail.com)")
RATE_S = 1.1                 # polite spacing between requests (~1 req/s)
MIN_BODY_CHARS = 200         # below this a page is not a text article

# Wire-agency byline => the whole article is licensed wire, not VOA PD => DROP.
# Applied ONLY to the (short, controlled) author field, so a bare "AP" token is
# safe to treat as Associated Press — VOA credits wire copy exactly that way
# (verified: "AP" bylines are AP gold-price / science wire translations). Word
# boundaries keep it from firing inside names (e.g. "Aparicio").
_WIRE_BYLINE = re.compile(
    r"\b(AFP|AP|Agence\s+France[-\s]?Presse|Associated\s+Press|Reuters|EFE|"
    r"Deutsche\s+Presse|dpa|Xinhua|ANSA)\b", re.I)
# Wire CREDIT inside a paragraph (a source credit, NOT an editorial mention of the
# agency). Matches "(AFP)", "— Reuters", "© AP", "Sous: AFP" — but deliberately
# NOT "ajans nouvèl Reuters" (VOA writing *about* a wire agency).
_WIRE_CREDIT = re.compile(
    r"\(\s*(AFP|AP|Reuters|EFE|Agence\s+France[-\s]?Presse|Associated\s+Press)\s*\)"
    r"|[—–-]\s*(AFP|AP|Reuters|EFE)\s*$"
    r"|©\s*(AFP|AP|Reuters|EFE)"
    r"|\bSous\s*:\s*(AFP|AP|Reuters|EFE)\b", re.I)


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": UA,
        "Accept-Language": "ht,fr;q=0.8,en;q=0.5",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    })
    return s


def get(session: requests.Session, url: str, timeout: int = 30,
        max_retries: int = 4, allow_redirects: bool = False) -> requests.Response:
    """Polite GET with backoff on 429/5xx. Redirects OFF by default so the caller
    can detect the topic-redirect stubs (bare `/a/<id>.html` for a small block of
    the newest IDs 302 -> /t/47.html). Sleeps RATE_S AFTER each attempt."""
    delay = RATE_S
    last = None
    for attempt in range(max_retries):
        try:
            r = session.get(url, timeout=timeout, allow_redirects=allow_redirects)
            last = r
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(delay)
                delay = min(delay * 2, 30)
                continue
            time.sleep(RATE_S)
            return r
        except requests.RequestException as e:
            last = e
            time.sleep(delay)
            delay = min(delay * 2, 30)
    if isinstance(last, requests.Response):
        time.sleep(RATE_S)
        return last
    raise last


def sitemap_entries(session: requests.Session) -> list[dict]:
    """All article URLs across both gzipped shards, newest-first as published.

    Returns [{"url","lastmod","year","id"}]. `id` is the trailing numeric article
    id (stable doc key); `year` is the lastmod year (publication proxy)."""
    out: list[dict] = []
    for shard in ARTICLE_SHARDS:
        r = session.get(SITE + shard, timeout=60)
        r.raise_for_status()
        xml = gzip.decompress(r.content).decode("utf-8")
        for block in re.findall(r"<url>(.*?)</url>", xml, re.S):
            mloc = re.search(r"<loc>(.*?)</loc>", block)
            if not mloc:
                continue
            url = mloc.group(1).strip()
            mlm = re.search(r"<lastmod>(.*?)</lastmod>", block)
            lm = mlm.group(1).strip() if mlm else ""
            mid = re.search(r"/(\d+)\.html$", url)
            year = lm[:4] if lm[:4].isdigit() else None
            out.append({"url": url, "lastmod": lm, "year": year,
                        "id": mid.group(1) if mid else None})
    return out


def _jsonld_news(soup: BeautifulSoup) -> dict:
    for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            d = json.loads(tag.string or tag.get_text() or "")
        except Exception:
            continue
        for it in (d if isinstance(d, list) else [d]):
            if isinstance(it, dict) and it.get("@type") in (
                    "NewsArticle", "Article", "Report"):
                return it
    return {}


def _author_name(author) -> str | None:
    if not author:
        return None
    if isinstance(author, list):
        names = [_author_name(a) for a in author]
        names = [n for n in names if n]
        return ", ".join(names) if names else None
    if isinstance(author, dict):
        return author.get("name")
    if isinstance(author, str):
        return author
    return None


def _og(soup: BeautifulSoup, prop: str) -> str | None:
    m = soup.find("meta", attrs={"property": prop}) or \
        soup.find("meta", attrs={"name": prop})
    return m.get("content") if m else None


def extract(html: str, url: str) -> dict:
    """Parse one article page. Returns a dict with metadata, the clean text pieces,
    and wire flags. `is_article` gates out audio/video program pages."""
    soup = BeautifulSoup(html, "lxml")
    meta = _jsonld_news(soup)

    headline = (meta.get("headline") or _og(soup, "og:title") or "").strip()
    date_pub = meta.get("datePublished")
    date_mod = meta.get("dateModified")
    author = _author_name(meta.get("author"))
    kw = meta.get("keywords")
    if isinstance(kw, list):
        kw = ", ".join(str(x) for x in kw)
    sections = kw or meta.get("articleSection")

    lede_el = soup.select_one("div.intro")
    lede = lede_el.get_text(" ", strip=True) if lede_el else ""

    wsw = soup.select_one("div.wsw")
    paras: list[str] = []
    if wsw:
        for p in wsw.find_all("p"):
            t = p.get_text(" ", strip=True)
            if t:
                paras.append(t)

    body_chars = sum(len(p) for p in paras)
    is_article = bool(wsw is not None and body_chars >= MIN_BODY_CHARS)

    wire_byline = bool(author and _WIRE_BYLINE.search(author))
    wire_para_idx = [i for i, p in enumerate(paras) if _WIRE_CREDIT.search(p)]

    # Clean text = headline + lede + NON-wire body paragraphs.
    kept_paras = [p for i, p in enumerate(paras) if i not in set(wire_para_idx)]
    parts = [x for x in ([headline, lede] + kept_paras) if x]
    clean_text = "\n\n".join(parts)

    return {
        "url": url,
        "headline": headline,
        "lede": lede,
        "paragraphs": paras,
        "kept_paragraphs": kept_paras,
        "clean_text": clean_text,
        "date_published": date_pub,
        "date_modified": date_mod,
        "author": author,
        "sections": sections,
        "is_article": is_article,
        "body_chars": body_chars,
        "wire_byline": wire_byline,
        "wire_para_count": len(wire_para_idx),
        "n_paragraphs": len(paras),
    }


def is_topic_redirect(resp: requests.Response) -> bool:
    """True for the newest-ID stub URLs that 302 -> a /t/<n>.html topic page."""
    if resp.status_code in (301, 302, 303, 307, 308):
        loc = resp.headers.get("location", "")
        return "/t/" in loc
    return False
