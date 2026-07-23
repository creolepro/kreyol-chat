"""Register the `kreyol-bpe` byte-level BPE pre-tokenizer in llama.cpp SOURCE (C++ side).

Applied at Modal image-build time to the pinned llama.cpp tree BEFORE the cmake build, so
the compiled `llama-cli` / `llama-tokenize` recognise a GGUF whose `tokenizer.ggml.pre`
== "kreyol-bpe" and split with the `kreyol_aware` regex. Three edits (verified patch
sites, commit 67b9b0e7f6ce):

  1. src/llama-vocab.h   — new enum value `LLAMA_VOCAB_PRE_TYPE_KREYOL` after LAGUNA=56.
  2. src/llama-vocab.cpp — `tokenizer_pre == "kreyol-bpe"` branch (name -> pre_type),
                           inserted before the terminal `unknown pre-tokenizer` throw.
  3. src/llama-vocab.cpp — a `case LLAMA_VOCAB_PRE_TYPE_KREYOL` in the regex switch,
                           inserted before `default:`.

NOTE on the regex: our HF `kreyol_aware` pattern uses POSSESSIVE quantifiers (`?+`, `++`)
which llama.cpp's regex splitter (like std::regex) does not support, so the C++ mirror
uses the GREEDY equivalents (`?`, `+`). This is the same simplification llama.cpp applies
to the llama-3 / qwen2 regexes. Whether greedy vs possessive changes any pre-token
boundary is exactly what the gate-4 token-ID parity test measures — any delta is reported,
not hidden.  The Python converter side (chkhsh -> "kreyol-bpe") is patched at CONVERT time
in llama_app.convert (it needs the exact AutoTokenizer hash).

Run: python patch_llamacpp_cpp.py <llama.cpp-root>
"""

import sys

# C++-source form of the kreyol_aware pattern: possessive quantifiers made greedy,
# backslashes doubled as they must appear in the .cpp string literal.
KREYOL_REGEX_CPP = r'"[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,2}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+"'

ENUM_LINE = "    LLAMA_VOCAB_PRE_TYPE_KREYOL           = 57,\n"

# (2) name -> pre_type: insert our branch just before the terminal throw.
THROW_ANCHOR = (
    '            } else {\n'
    '                throw std::runtime_error(format("unknown pre-tokenizer type: \'%s\'", tokenizer_pre.c_str()));'
)
NAME_BRANCH = (
    '            } else if (\n'
    '                    tokenizer_pre == "kreyol-bpe") {\n'
    '                pre_type = LLAMA_VOCAB_PRE_TYPE_KREYOL;\n'
    '                clean_spaces = false;\n'
    '            } else {\n'
    '                throw std::runtime_error(format("unknown pre-tokenizer type: \'%s\'", tokenizer_pre.c_str()));'
)

# (3) pre_type -> regex: insert our case before the default case of the regex switch.
DEFAULT_ANCHOR = (
    "            default:\n"
    "                // default regex for BPE tokenization pre-processing\n"
    "                regex_exprs = {"
)
REGEX_CASE = (
    "            case LLAMA_VOCAB_PRE_TYPE_KREYOL:\n"
    "                regex_exprs = {\n"
    "                    " + KREYOL_REGEX_CPP + ",\n"
    "                };\n"
    "                break;\n"
    + DEFAULT_ANCHOR
)


def _patch_enum(root):
    path = f"{root}/src/llama-vocab.h"
    with open(path, encoding="utf-8") as f:
        src = f.read()
    assert "LLAMA_VOCAB_PRE_TYPE_KREYOL" not in src, "enum already patched"
    e = src.index("enum llama_vocab_pre_type")
    close = src.index("};", e)
    src = src[:close] + ENUM_LINE + src[close:]
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    print(f"[patch] enum -> {path}")


def _replace_once(root, rel, old, new, label):
    path = f"{root}/{rel}"
    with open(path, encoding="utf-8") as f:
        src = f.read()
    n = src.count(old)
    assert n == 1, f"{label}: expected 1 match, got {n}"
    src = src.replace(old, new)
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    print(f"[patch] {label} -> {path}")


def main(root):
    _patch_enum(root)
    _replace_once(root, "src/llama-vocab.cpp", THROW_ANCHOR, NAME_BRANCH, "name->pre_type")
    _replace_once(root, "src/llama-vocab.cpp", DEFAULT_ANCHOR, REGEX_CASE, "pre_type->regex")
    print("[patch_llamacpp_cpp] done (3 edits)")


if __name__ == "__main__":
    main(sys.argv[1])
