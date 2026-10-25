#!/usr/bin/env python3
"""Build lexical-control variants of the SmartIntent binary source task."""

import argparse
import json
import re
from pathlib import Path

from transformers import AutoTokenizer


SOLIDITY_KEYWORDS = set("""
abstract address after alias anonymous apply as assembly assert auto bool break byte bytes
calldata case catch constant constructor continue contract default define delete do else emit enum
error event external fallback false final fixed for from function if immutable implements import in
indexed inline int interface internal is let library mapping match memory modifier mutable new null
of override payable pragma private public pure receive reference relocatable return returns sealed
sizeof static storage string struct super supports switch this throw true try type typedef typeof
ufixed uint unchecked using var view virtual wei while years days hours minutes seconds weeks ether
gwei finney szabo msg tx block abi now require revert selfdestruct suicide keccak256 sha256 ripemd160
ecrecover addmod mulmod gasleft balance caller callvalue calldatasize calldatacopy codesize codecopy
gasprice extcodesize extcodecopy returndatasize returndatacopy extcodehash create create2 call
callcode delegatecall staticcall return stop invalid log0 log1 log2 log3 log4 origin coinbase
timestamp number difficulty prevrandao gaslimit chainid basefee blobbasefee selector value length
sender data sig encode encodePacked encodeWithSelector encodeWithSignature decode concat name symbol
decimals totalSupply balanceOf transfer allowance approve transferFrom owner onlyOwner
""".split())


def strip_comments_and_headers(text):
    out = []
    i, n, state = 0, len(text), "code"
    quote = ""
    while i < n:
        c = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if state == "code":
            if c == "/" and nxt == "/":
                state = "line_comment"
                i += 2
                continue
            if c == "/" and nxt == "*":
                state = "block_comment"
                i += 2
                continue
            if c in ('"', "'"):
                state, quote = "string", c
            out.append(c)
            i += 1
        elif state == "line_comment":
            if c == "\n":
                out.append("\n")
                state = "code"
            i += 1
        elif state == "block_comment":
            if c == "*" and nxt == "/":
                state = "code"
                i += 2
            else:
                if c == "\n":
                    out.append("\n")
                i += 1
        else:
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if c == quote:
                state = "code"
            i += 1
    cleaned = "".join(out)
    cleaned = re.sub(r"(?mi)^\s*pragma\s+solidity\b[^;]*;\s*", "", cleaned)
    cleaned = re.sub(r"(?mi)^\s*pragma\s+experimental\b[^;]*;\s*", "", cleaned)
    cleaned = re.sub(r"(?mi)^\s*SPDX-License-Identifier\s*:[^\n]*", "", cleaned)
    return cleaned


def anonymize_identifiers(text):
    mapping = {}
    counters = {"CONTRACT": 0, "FUNC": 0, "VAR": 0}
    declaration_context = None
    previous = ""

    def placeholder(name, category):
        key = (category, name)
        if key not in mapping:
            counters[category] += 1
            mapping[key] = f"{category}_{counters[category]}"
        return mapping[key]

    token_re = re.compile(
        r"(//[^\n]*|/\*[\s\S]*?\*/|\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|"
        r"\b[A-Za-z_$][A-Za-z0-9_$]*\b|\s+|.)"
    )
    result = []
    for match in token_re.finditer(text):
        tok = match.group(0)
        if tok.startswith("//") or tok.startswith("/*") or tok[:1] in ('"', "'") or tok.isspace():
            result.append(tok)
            continue
        if not re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$]*", tok):
            result.append(tok)
            previous = tok
            continue
        low = tok.lower()
        if low in SOLIDITY_KEYWORDS or tok in SOLIDITY_KEYWORDS:
            result.append(tok)
            declaration_context = low if low in {"contract", "interface", "library", "function", "modifier", "event", "struct", "enum"} else declaration_context
            previous = tok
            continue
        if declaration_context in {"contract", "interface", "library", "struct", "enum"}:
            category = "CONTRACT"
            declaration_context = None
        elif declaration_context in {"function", "modifier", "event"}:
            category = "FUNC"
            declaration_context = None
        elif previous == ".":
            # Preserve member/API names; arbitrary members are part of code behavior.
            result.append(tok)
            previous = tok
            continue
        else:
            category = "VAR"
        result.append(placeholder(tok, category))
        previous = tok
    return "".join(result)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in open(args.data_path, encoding="utf-8") if line.strip()]
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, use_fast=True)
    variants = {"remove_comments_headers": [], "identifier_anonymized": []}
    lengths = []
    for idx, row in enumerate(rows):
        text = str(row["text"])
        stripped = strip_comments_and_headers(text)
        anonymous = anonymize_identifiers(text)
        base = {k: v for k, v in row.items()}
        variants["remove_comments_headers"].append({**base, "text": stripped, "control": "remove_comments_headers"})
        variants["identifier_anonymized"].append({**base, "text": anonymous, "control": "identifier_anonymized"})
        lengths.append({
            "row": idx,
            "address": row.get("address"),
            "label": int(row["label"]),
            "tokens_original": len(tokenizer.encode(text, add_special_tokens=False)),
            "tokens_remove_comments_headers": len(tokenizer.encode(stripped, add_special_tokens=False)),
            "tokens_identifier_anonymized": len(tokenizer.encode(anonymous, add_special_tokens=False)),
        })
    for name, variant_rows in variants.items():
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for row in variant_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(out / "token_lengths.jsonl", "w", encoding="utf-8") as f:
        for row in lengths:
            f.write(json.dumps(row) + "\n")
    with open(out / "metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "source": args.data_path,
            "rows": len(rows),
            "variants": list(variants),
            "notes": {
                "remove_comments_headers": "Lexically removes line/block comments, SPDX lines, and Solidity/experimental pragma directives.",
                "identifier_anonymized": "Deterministic per-contract lexical anonymization; Solidity keywords, literals, strings, comments, and member/API names are retained.",
            },
        }, f, indent=2)


if __name__ == "__main__":
    main()
