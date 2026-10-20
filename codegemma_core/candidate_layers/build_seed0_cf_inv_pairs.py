#!/usr/bin/env python3
"""Build conservative Seed-0 counterfactual and surface pairs for CodeGemma."""

import hashlib
import json
import re
from pathlib import Path

from transformers import AutoTokenizer


ROOT = Path("/root/autodl-tmp/models")
UNION = ROOT / "runs/stage3_smartintent_sliding/sliding_union.jsonl"
MANIFEST = ROOT / "runs/stage4_smartintent_full_trials/manifests/trial_seed_0.json"
MODEL = ROOT / "base/CodeGemma-7B"
OUT = ROOT / "runs/stage6_codegemma_seed0_cf_inv/data"

BLACKLIST_PATTERN = re.compile(
    r"black\w*|denylist\w*|isBot\b|bots?\s*\[|blocked(address(es)?)?\s*\[",
    re.IGNORECASE,
)


def load_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mask_noncode(source):
    """Mask comments and strings with spaces while preserving byte offsets."""
    output = list(source)
    index = 0
    state = "code"
    quote = ""
    while index < len(source):
        if state == "code":
            if source.startswith("//", index):
                output[index] = output[index + 1] = " "
                index += 2
                state = "line_comment"
            elif source.startswith("/*", index):
                output[index] = output[index + 1] = " "
                index += 2
                state = "block_comment"
            elif source[index] in "'\"":
                quote = source[index]
                output[index] = " "
                index += 1
                state = "string"
            else:
                index += 1
        elif state == "line_comment":
            if source[index] == "\n":
                state = "code"
            else:
                output[index] = " "
            index += 1
        elif state == "block_comment":
            if source.startswith("*/", index):
                output[index] = output[index + 1] = " "
                index += 2
                state = "code"
            else:
                output[index] = " "
                index += 1
        else:
            if source[index] == "\\" and index + 1 < len(source):
                output[index] = output[index + 1] = " "
                index += 2
            elif source[index] == quote:
                output[index] = " "
                index += 1
                state = "code"
            else:
                output[index] = " "
                index += 1
    return "".join(output)


def balanced(source):
    masked = mask_noncode(source)
    pairs = {"(": ")", "[": "]", "{": "}"}
    stack = []
    for char in masked:
        if char in pairs:
            stack.append(pairs[char])
        elif char in ")]}" and (not stack or stack.pop() != char):
            return False
    return not stack


def find_closing(masked, opening, left="(", right=")"):
    depth = 1
    index = opening + 1
    while index < len(masked) and depth:
        if masked[index] == left:
            depth += 1
        elif masked[index] == right:
            depth -= 1
        index += 1
    return index - 1 if depth == 0 else None


def first_argument_span(masked, opening, closing):
    depth = 0
    for index in range(opening + 1, closing):
        char = masked[index]
        if char in "([{":
            depth += 1
        elif char in ")]}" and depth:
            depth -= 1
        elif char == "," and depth == 0:
            return opening + 1, index
    return opening + 1, closing


def blacklist_edit(source):
    masked = mask_noncode(source)
    for match in re.finditer(r"\brequire\s*\(", masked):
        opening = masked.find("(", match.start())
        closing = find_closing(masked, opening)
        if closing is None:
            continue
        start, end = first_argument_span(masked, opening, closing)
        condition = masked[start:end]
        if BLACKLIST_PATTERN.search(condition):
            edited = source[:start] + " true " + source[end:]
            return edited, {
                "operation": "neutralize_blacklist_require",
                "edit_start": start,
                "edit_end": end,
                "original_fragment": source[start:end],
                "replacement_fragment": " true ",
            }
    return None, None


def function_spans(source):
    masked = mask_noncode(source)
    spans = []
    for match in re.finditer(r"\bfunction\b", masked):
        opening = masked.find("{", match.end())
        semicolon = masked.find(";", match.end())
        if opening < 0 or (semicolon >= 0 and semicolon < opening):
            continue
        if re.search(r"\bfunction\b", masked[match.end():opening]):
            continue
        closing = find_closing(masked, opening, "{", "}")
        if closing is not None:
            spans.append((match.start(), opening, closing + 1))
    return spans


def mint_edit(source):
    masked = mask_noncode(source)
    for start, opening, end in function_spans(source):
        header = masked[start:opening]
        body = masked[opening:end]
        is_mint = bool(
            re.search(r"\bfunction\s+\w*mint\w*\b", header, re.IGNORECASE)
            or re.search(r"\b_mint\s*\(", body, re.IGNORECASE)
        )
        if not is_mint:
            continue
        guard = re.search(r"\bonlyOwner\b", header)
        if guard:
            guard_start = start + guard.start()
            guard_end = start + guard.end()
            edited = source[:guard_start] + " " * (guard_end - guard_start) + source[guard_end:]
            return edited, {
                "operation": "remove_onlyOwner_from_mint",
                "edit_start": guard_start,
                "edit_end": guard_end,
                "function_start": start,
                "function_end": end,
                "original_fragment": source[guard_start:guard_end],
                "replacement_fragment": " " * (guard_end - guard_start),
            }
    return None, None


def containing_function(source, position):
    for start, opening, end in function_spans(source):
        if start <= position < end:
            return start, opening, end
    return None


def split_top_level(text):
    parts = []
    depth = 0
    start = 0
    for index, char in enumerate(text):
        if char in "([{":
            depth += 1
        elif char in ")]}" and depth:
            depth -= 1
        elif char == "," and depth == 0:
            parts.append((start, index))
            start = index + 1
    parts.append((start, len(text)))
    return parts


def identifier_surface(source, target_position):
    span = containing_function(source, target_position)
    if not span:
        return None
    start, opening, end = span
    masked = mask_noncode(source)
    header = masked[start:opening]
    first_paren = header.find("(")
    if first_paren < 0:
        return None
    header_close = find_closing(header, first_paren)
    if header_close is None:
        return None
    parameters = header[first_paren + 1:header_close]
    keywords = {"memory", "storage", "calldata", "payable", "indexed"}
    candidates = []
    for left, right in split_top_level(parameters):
        segment = parameters[left:right]
        identifiers = re.findall(r"\b[A-Za-z_$][A-Za-z0-9_$]*\b", segment)
        identifiers = [item for item in identifiers if item not in keywords]
        if len(identifiers) >= 2:
            candidates.append(identifiers[-1])
    function_mask = masked[start:end]
    for old in candidates:
        occurrences = list(re.finditer(rf"\b{re.escape(old)}\b", function_mask))
        if len(occurrences) < 2:
            continue
        new = "SURF_PARAM_1"
        if re.search(rf"\b{new}\b", masked):
            continue
        result = source
        for occurrence in reversed(occurrences):
            left = start + occurrence.start()
            right = start + occurrence.end()
            result = result[:left] + new + result[right:]
        return result
    return None


def surface_variants(source, edit):
    position = int(edit["edit_start"])
    variants = {}
    comment = "/* surface-only annotation: behavior unchanged */\n"
    variants["comment_insert"] = source[:position] + comment + source[position:]
    variants["format_insert"] = source[:position] + "\n        " + source[position:]
    spdx_pattern = re.compile(r"(^\s*//\s*SPDX-License-Identifier\s*:\s*)([^\r\n]+)", re.MULTILINE)
    if spdx_pattern.search(source):
        variants["spdx_rewrite"] = spdx_pattern.sub(r"\1UNLICENSED", source, count=1)
    else:
        variants["spdx_rewrite"] = "// SPDX-License-Identifier: UNLICENSED\n" + source
    renamed = identifier_surface(source, position)
    if renamed and renamed != source:
        variants["local_parameter_rename"] = renamed
    return {name: text for name, text in variants.items() if text != source and balanced(text)}


def token_edit_count(tokenizer, original, variant):
    original_ids = tokenizer.encode(original, add_special_tokens=False, verbose=False)
    variant_ids = tokenizer.encode(variant, add_special_tokens=False, verbose=False)
    prefix = 0
    while prefix < min(len(original_ids), len(variant_ids)) and original_ids[prefix] == variant_ids[prefix]:
        prefix += 1
    suffix = 0
    while (
        suffix < min(len(original_ids) - prefix, len(variant_ids) - prefix)
        and original_ids[-1 - suffix] == variant_ids[-1 - suffix]
    ):
        suffix += 1
    return max(len(original_ids) - prefix - suffix, len(variant_ids) - prefix - suffix)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sources = load_jsonl(UNION)
    manifest = json.load(open(MANIFEST, encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, use_fast=True)
    unique_sources = []
    source_lookup = {}

    def add_source(text, metadata):
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest not in source_lookup:
            source_lookup[digest] = len(unique_sources)
            unique_sources.append({"text": text, "sha256": digest, **metadata})
        return source_lookup[digest]

    pairs = []
    rejected = []
    for record in manifest["records"]:
        item = sources[int(record["feature_index"])]
        original = str(item["text"])
        if not balanced(original):
            rejected.append({"address": record["address"], "reason": "unbalanced_original"})
            continue
        transformations = []
        if int(item["labels"].get("Blacklist", 0)):
            edited, info = blacklist_edit(original)
            if edited:
                transformations.append(("Blacklist", edited, info))
        if int(item["labels"].get("Mint", 0)):
            edited, info = mint_edit(original)
            if edited:
                transformations.append(("Mint", edited, info))
        if not transformations and (item["labels"].get("Blacklist") or item["labels"].get("Mint")):
            rejected.append({"address": record["address"], "reason": "no_high_confidence_rule"})
        for intent, counterfactual, edit in transformations:
            if counterfactual == original or not balanced(counterfactual):
                rejected.append({"address": record["address"], "intent": intent, "reason": "invalid_edit"})
                continue
            base_metadata = {
                "address": record["address"],
                "creator": record["group"],
                "split": record["split"],
                "label": int(record["label"]),
                "intent": intent,
            }
            original_index = add_source(original, {**base_metadata, "variant": "original"})
            counterfactual_index = add_source(counterfactual, {**base_metadata, "variant": "counterfactual"})
            cf_edits = token_edit_count(tokenizer, original, counterfactual)
            surfaces = []
            for transform, surface_text in surface_variants(original, edit).items():
                surface_index = add_source(surface_text, {**base_metadata, "variant": transform})
                surfaces.append({
                    "transform": transform,
                    "source_index": surface_index,
                    "token_edit_count": token_edit_count(tokenizer, original, surface_text),
                })
            if not surfaces:
                rejected.append({"address": record["address"], "intent": intent, "reason": "no_surface_variant"})
                continue
            matched = min(surfaces, key=lambda row: abs(row["token_edit_count"] - cf_edits))
            pairs.append({
                "pair_id": f"seed0_{len(pairs):04d}",
                **base_metadata,
                "feature_index": int(record["feature_index"]),
                "original_source_index": original_index,
                "counterfactual_source_index": counterfactual_index,
                "counterfactual_token_edit_count": cf_edits,
                "matched_surface_transform": matched["transform"],
                "matched_surface_source_index": matched["source_index"],
                "matched_surface_token_edit_count": matched["token_edit_count"],
                "surface_variants": surfaces,
                "edit": edit,
                "validation": "comment/string-aware lexical rule plus balanced delimiters",
            })

    with open(OUT / "paired_sources.jsonl", "w", encoding="utf-8") as handle:
        for row in unique_sources:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(OUT / "pair_manifest.jsonl", "w", encoding="utf-8") as handle:
        for row in pairs:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(OUT / "rejected.jsonl", "w", encoding="utf-8") as handle:
        for row in rejected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "model": "CodeGemma-7B",
        "seed": 0,
        "source_task_contracts": 2000,
        "pair_count": len(pairs),
        "unique_source_variants": len(unique_sources),
        "pairs_by_split": {},
        "pairs_by_intent": {},
        "rejected_count": len(rejected),
        "limitations": [
            "No Solidity AST/compiler was available; transformations use conservative offset-preserving lexical rules.",
            "First version covers high-confidence Blacklist require neutralization and Mint onlyOwner removal only.",
            "Honeypot and MaxSell lack adequate fixed-Seed-0 validation/test examples in the 2,000-contract task.",
        ],
    }
    for split in ("train", "val", "test"):
        summary["pairs_by_split"][split] = sum(row["split"] == split for row in pairs)
    for intent in ("Blacklist", "Mint"):
        summary["pairs_by_intent"][intent] = {
            split: sum(row["intent"] == intent and row["split"] == split for row in pairs)
            for split in ("train", "val", "test")
        }
    with open(OUT / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
