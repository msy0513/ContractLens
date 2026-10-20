#!/usr/bin/env python3
"""Build semantics-preserving variants for every Seed-0 binary Train contract."""

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_seed0_binary_cf_inv_v2 import (
    balanced,
    compile_check,
    compiler_version,
    format_normalized,
    make_parser,
    pragma_rewrite,
    reorder_functions,
)
from build_seed0_cf_inv_pairs import identifier_surface, token_edit_count


ROOT = Path("/root/autodl-tmp/models")
UNION = ROOT / "runs/stage3_smartintent_sliding/sliding_union.jsonl"
MANIFEST = ROOT / "runs/stage4_smartintent_full_trials/manifests/trial_seed_0.json"
MODEL = ROOT / "base/CodeGemma-7B"
OUT = ROOT / "runs/stage8_codegemma_seed0_binary_cf_inv_v3/data_inv"


def load_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def strip_comments_preserve_pragma(text):
    output = []
    index = 0
    state = "code"
    quote = ""
    while index < len(text):
        char = text[index]
        nxt = text[index + 1] if index + 1 < len(text) else ""
        if state == "code":
            if char == "/" and nxt == "/":
                line_end = text.find("\n", index)
                comment = text[index:line_end if line_end >= 0 else len(text)]
                if "SPDX-License-Identifier" in comment:
                    pass
                state = "line"
                index += 2
            elif char == "/" and nxt == "*":
                state = "block"
                index += 2
            else:
                output.append(char)
                if char in {'"', "'"}:
                    state = "string"
                    quote = char
                index += 1
        elif state == "line":
            if char == "\n":
                output.append(char)
                state = "code"
            index += 1
        elif state == "block":
            if char == "*" and nxt == "/":
                state = "code"
                index += 2
            else:
                if char == "\n":
                    output.append(char)
                index += 1
        else:
            output.append(char)
            if char == "\\" and index + 1 < len(text):
                output.append(text[index + 1])
                index += 2
            else:
                if char == quote:
                    state = "code"
                index += 1
    return "".join(output)


def scoped_parameter_rename(source):
    for position in (source.find("function "), len(source) // 2, source.rfind("function ")):
        if position < 0:
            continue
        variant = identifier_surface(source, position)
        if variant and variant != source:
            return variant
    return source


def variants_for(source, parser):
    marker = "/* behavior-preserving invariance control */\n"
    variants = {
        "comment_insert": marker + source,
        "remove_comments": strip_comments_preserve_pragma(source),
        "format_normalized": format_normalized(source),
        "pragma_equivalent": pragma_rewrite(source),
        "function_reordered": reorder_functions(source, parser),
        "local_parameter_rename": scoped_parameter_rename(source),
    }
    result = {}
    lexical_low_risk = {"comment_insert", "remove_comments", "format_normalized"}
    for name, text in variants.items():
        if text == source or not text.strip() or not balanced(text):
            continue
        tree = parser.parse(text.encode("utf-8"))
        # Some historical SmartIntent sources are incomplete compilation units
        # or use syntax unsupported by the parser. Comment and whitespace-only
        # edits remain valid invariance controls for those records; structural
        # edits still require a clean AST.
        if name in lexical_low_risk or not tree.root_node.has_error:
            result[name] = text
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(UNION)
    manifest = json.load(open(MANIFEST, encoding="utf-8"))
    train_records = [record for record in manifest["records"] if record["split"] == "train"]
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, use_fast=True)
    parser = make_parser()
    unique_sources = []
    source_lookup = {}
    variants = []
    contract_rows = []
    counts = Counter()

    def add_source(text, metadata):
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest not in source_lookup:
            source_lookup[digest] = len(unique_sources)
            unique_sources.append({"text": text, "sha256": digest, **metadata})
        return source_lookup[digest]

    for ordinal, record in enumerate(train_records):
        row = rows[int(record["feature_index"])]
        source = str(row["text"])
        version = compiler_version(source)
        original_compile = compile_check(source, version)
        generated = variants_for(source, parser)
        accepted = []
        for name, text in generated.items():
            validation = compile_check(text, version) if original_compile["passed"] else {
                "attempted": False,
                "passed": False,
                "reason": "original_not_compile_valid; retained as AST-valid low-risk transform",
                "version": version,
            }
            if original_compile["passed"] and not validation["passed"]:
                counts[f"rejected_{name}"] += 1
                continue
            tier = "ast_and_compile" if original_compile["passed"] and validation["passed"] else "non_compiling_low_risk"
            source_index = add_source(text, {
                "variant": name,
                "validation_tier": tier,
                "original_feature_index": int(record["feature_index"]),
                "label": int(record["label"]),
            })
            item = {
                "contract_index": ordinal,
                "original_feature_index": int(record["feature_index"]),
                "surface_source_index": source_index,
                "address": record["address"],
                "creator": record["group"],
                "label": int(record["label"]),
                "transform": name,
                "validation_tier": tier,
                "token_edit_count": token_edit_count(tokenizer, source, text),
                "original_compile": original_compile,
                "surface_compile": validation,
            }
            variants.append(item)
            accepted.append(item)
            counts[name] += 1
            counts[tier] += 1
        if not accepted:
            raise RuntimeError(f"No valid surface variant for Train contract {record['address']}")
        contract_rows.append({
            "contract_index": ordinal,
            "original_feature_index": int(record["feature_index"]),
            "address": record["address"],
            "creator": record["group"],
            "label": int(record["label"]),
            "variant_count": len(accepted),
            "transforms": [item["transform"] for item in accepted],
        })
        if (ordinal + 1) % 50 == 0:
            print(json.dumps({"processed_contracts": ordinal + 1, "variants": len(variants)}), flush=True)

    for name, payload in (("surface_sources.jsonl", unique_sources), ("surface_manifest.jsonl", variants), ("contracts.jsonl", contract_rows)):
        with open(OUT / name, "w", encoding="utf-8") as handle:
            for item in payload:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    summary = {
        "model": "CodeGemma-7B",
        "seed": 0,
        "train_contracts": len(train_records),
        "unsafe_contracts": sum(int(record["label"]) for record in train_records),
        "normal_contracts": sum(not int(record["label"]) for record in train_records),
        "surface_variants": len(variants),
        "unique_surface_sources": len(unique_sources),
        "counts": dict(counts),
        "complete_train_coverage": len(contract_rows) == len(train_records) and all(row["variant_count"] > 0 for row in contract_rows),
        "selection_uses_test": False,
    }
    with open(OUT / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
