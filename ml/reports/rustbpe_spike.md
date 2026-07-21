# rustbpe integration spike (B0) — go/no-go

*Snapshot 2026-07-20. nanochat pinned `92d63d4e8bb4df75c3b71618f31ddde2378b2bcd`; `rustbpe` PyPI wheel (see pyproject). Toy: 16,384 vocab on 10.0MB corpus.*

## Verdict: **GO** — proceed to the sweep with rustbpe.

## 1. Build friction

- `rustbpe` is a **standalone PyPI package** (`rustbpe>=0.1.0`), not vendored in nanochat at the pinned commit — so **no Rust/maturin source build was needed**: a prebuilt wheel installed via `uv add rustbpe` in <1s. (Rust 1.74 + cargo are present as a fallback for an sdist build; not exercised.)
- Toy training: **0.6s** for 16,384 vocab on 10.0MB (18.1 MB/s).
- Vocab size is a clean parameter (`train_from_iterator(iter, vocab_size, pattern=…)`); nanochat's `tok_train.py` default is 32,768, we sweep 8k–32k. rustbpe trains `vocab_size − 9` (the 9 special tokens are appended after training). Inference uses a `tiktoken.Encoding` built from rustbpe's `get_mergeable_ranks()` + `get_pattern()`.

## 2. Round-trip (encode→decode identity)

Accent-heavy Kreyòl (è/ò are multi-byte UTF-8) + code-switched French/English: **6/6 exact**.

## 3. Apostrophe-regex decision → **Kreyòl-aware pattern**

nanochat's GPT-4-style pattern hard-codes an English-contraction clause `'(?i:[sdmt]|ll|ve|re)`. Kreyòl clitics elide on the *other* side of the apostrophe (`m'ap` = mwen ap). The probe shows the clause **misfires** on Kreyòl TMA forms whose marker starts with t/s/d/m (`te`, `ta`, …): it splits `'t` off, shredding the marker.

| form | stock pre-tokenization | Kreyòl-aware pre-tokenization |
|---|---|---|
| `m'ap` | `['m', "'ap"]` | `['m', "'ap"]` |
| `l'ap` | `['l', "'ap"]` | `['l', "'ap"]` |
| `n'ap` | `['n', "'ap"]` | `['n', "'ap"]` |
| `t'ap` | `['t', "'ap"]` | `['t', "'ap"]` |
| `w'ap` | `['w', "'ap"]` | `['w', "'ap"]` |
| `m'te` | `['m', "'t", 'e']` | `['m', "'te"]`  ⚠️ |
| `m'ta` | `['m', "'t", 'a']` | `['m', "'ta"]`  ⚠️ |
| `n'ta` | `['n', "'t", 'a']` | `['n', "'ta"]`  ⚠️ |
| `l'te` | `['l', "'t", 'e']` | `['l', "'te"]`  ⚠️ |
| `sa'k` | `['sa', "'k"]` | `['sa', "'k"]` |
| _(sentence)_ `Li di m'ap vini men n'ta pito rete.` | `['Li', ' di', ' m', "'ap", ' vini', ' men', ' n', "'t", 'a', ' pito', ' rete', '.']` | `['Li', ' di', ' m', "'ap", ' vini', ' men', ' n', "'ta", ' pito', ' rete', '.']` |

**Decision:** drop the English-contraction clause (the only change). Rationale: it **fixes** `m'te`→`['m',"'te"]`, `n'ta`→`['n',"'ta"]` etc. (marker kept whole), leaves the `m'ap`/`l'ap`/`n'ap`/`t'ap`/`w'ap` family **identical** to stock, and leaves English single-letter contractions (`don't`→`['don',"'t"]`) **identical** too — the general letter clause handles `'t` the same way. Net English regression is limited to the rare multi-letter-after-apostrophe-starting-with-s/d/m/t case. The pattern flows through rustbpe→tiktoken cleanly (`train_from_iterator(pattern=…)` → `get_pattern()`).

Token-level on the trained toy (Kreyòl-aware): `m'ap` → ['m', "'ap"]; `n'ta` → ['n', "'", 'ta']; `m'te` → ['m', "'", 'te'].

## 4. Format bridges — exact-ID parity vs tiktoken

Built (a) an HF `tokenizer.json` (BPE with reconstructed merges + a `Split(regex)` + `ByteLevel` pre-tokenizer) and (b) a browser `{vocab, merges}` JSON plus a tiktoken-native base64 dump (js-tiktoken). Re-encoded a **1035-line** source-mixed probe set through the HF bridge and compared token IDs to tiktoken:

- **sentence-exact ID match: 1035/1035 = 100.0%**
- token-level match: 100.00%

The **tiktoken encoding (from rustbpe) is the source of truth**; the HF and browser artifacts are bridges. Parity is reported, not assumed. The browser also ships the tiktoken-native base64 form for js-tiktoken (exact by construction).

## 5. Go/no-go

**GO.** rustbpe builds trivially (wheel), trains fast, round-trips exactly, the apostrophe question is resolved with evidence, and the format bridges reproduce tiktoken IDs exactly. Proceeding to the vocab sweep (Part 2) with rustbpe and the Kreyòl-aware pattern.
