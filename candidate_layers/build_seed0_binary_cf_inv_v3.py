#!/usr/bin/env python3
"""Build a ten-intent, binary unsafe-to-normal counterfactual dataset.

Intent labels are used only to locate and remove unsafe capabilities. Every
accepted pair has a binary transition from at least one unsafe intent to no
remaining annotated unsafe intent. Multi-label contracts are accepted only
when every active intent receives a high-confidence transformation.
"""

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_seed0_binary_cf_inv_v2 import (
    INTENTS,
    annotation_positions,
    apply_edits,
    balanced,
    compile_check,
    compiler_version,
    containing_function,
    format_normalized,
    function_records,
    make_parser,
    mask_noncode_bytes,
    read_annotations,
    walk,
)
from build_seed0_cf_inv_pairs import identifier_surface, token_edit_count


ROOT = Path("/root/autodl-tmp/models")
SOURCE = ROOT / "runs/stage3_smartintent/source_multilabel.jsonl"
UNION = ROOT / "runs/stage3_smartintent_sliding/sliding_union.jsonl"
MANIFEST = ROOT / "runs/stage4_smartintent_full_trials/manifests/trial_seed_0.json"
MODEL = ROOT / "base/CodeGemma-7B"
OUT = ROOT / "runs/stage8_codegemma_seed0_binary_cf_inv_v3/data_cf"

TARGETS = {
    "train": {intent: 60 for intent in INTENTS},
    "val": {intent: 20 for intent in INTENTS},
    "test": {intent: 20 for intent in INTENTS},
}
for split in TARGETS:
    TARGETS[split]["MaxSell"] = {"train": 36, "val": 10, "test": 12}[split]

TERM_PATTERNS = {
    "Blacklist": re.compile(rb"black|block(ed|list)?|deny|bot|locked", re.I),
    "DisableTrading": re.compile(rb"trad(e|ing)|tradingopen|tradingenabled|swapenabled|open", re.I),
    "MaxTX": re.compile(rb"maxtx|maxtransaction|txlimit|maxtransfer|max_tx|maxamount", re.I),
    "MaxSell": re.compile(rb"maxsell|max_sell|max.*sell|sell.*limit", re.I),
}


def load_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stable_value(value):
    return int(hashlib.sha256(f"seed0:{value}".encode()).hexdigest()[:16], 16) / 16**16


def fixed_group_splits():
    manifest = json.load(open(MANIFEST, encoding="utf-8"))
    result = {}
    for record in manifest["records"]:
        group = str(record["group"] or record["address"]).lower()
        previous = result.setdefault(group, record["split"])
        if previous != record["split"]:
            raise RuntimeError(f"Group appears in multiple splits: {group}")
    return result


def split_for(row, fixed):
    group = str(row.get("group") or row.get("creator") or row.get("address") or row["contract_id"]).lower()
    if group in fixed:
        return fixed[group]
    value = stable_value(group)
    return "train" if value < 0.64 else "val" if value < 0.80 else "test"


def first_argument(masked, opening, closing):
    depth = 0
    for index in range(opening + 1, closing):
        char = masked[index]
        if char in b"([{":
            depth += 1
        elif char in b")]}" and depth:
            depth -= 1
        elif char == 44 and depth == 0:
            return opening + 1, index
    return opening + 1, closing


def closing(masked, opening, left=40, right=41):
    depth = 1
    for index in range(opening + 1, len(masked)):
        if masked[index] == left:
            depth += 1
        elif masked[index] == right:
            depth -= 1
            if depth == 0:
                return index
    return None


def normalize_line(value):
    return re.sub(r"\s+", " ", str(value).strip())


def exact_annotation_functions(source, entries, functions):
    lines = []
    offset = 0
    for line in source.splitlines(keepends=True):
        encoded = line.encode("utf-8")
        leading = len(encoded) - len(line.lstrip().encode("utf-8"))
        lines.append((normalize_line(line), offset + leading, offset + len(encoded)))
        offset += len(encoded)
    result = []
    unmatched = []
    for entry in entries:
        target = normalize_line(entry.get("line", ""))
        if len(target) < 4:
            continue
        matches = [(start, end) for text, start, end in lines if text == target]
        if not matches and len(target) >= 24:
            matches = [
                (start, end) for text, start, end in lines
                if len(text) >= 12 and (target in text or text in target)
            ]
        if not matches:
            unmatched.append(target)
            continue
        for start, _ in matches:
            function = containing_function(functions, start)
            if function:
                result.append(function)
    unique = {row["node"].start_byte: row for row in result}
    return list(unique.values()), unmatched


def function_visibility(function):
    header = function["header"]
    return next((name for name in ("public", "external", "internal", "private") if re.search(rf"\b{name}\b", header)), "unknown")


def disable_edit(function, intent, operation="disable_entry"):
    body = function["body"]
    if not body:
        return None
    replacement = b'\n        revert("unsafe intent removed");\n    '
    return (body.start_byte + 1, body.end_byte - 1, replacement, operation, intent, function["node"].start_byte)


def noop_edit(function, intent, operation="neutralize_payout"):
    body = function["body"]
    if not body or re.search(r"\breturns\s*\(", function["header"]):
        return None
    replacement = b"\n        // unsafe intent removed\n    "
    return (body.start_byte + 1, body.end_byte - 1, replacement, operation, intent, function["node"].start_byte)


def condition_edits(source_bytes, intent, assigned_names=()):
    masked = mask_noncode_bytes(source_bytes)
    pattern = TERM_PATTERNS.get(intent)
    names = [name.encode() for name in assigned_names if name]
    edits = []
    for keyword in (rb"\brequire\s*\(", rb"\bif\s*\("):
        for match in re.finditer(keyword, masked):
            opening = masked.find(b"(", match.start())
            end = closing(masked, opening)
            if end is None:
                continue
            start, condition_end = first_argument(masked, opening, end) if keyword.startswith(rb"\brequire") else (opening + 1, end)
            condition = masked[start:condition_end]
            relevant = bool(pattern and pattern.search(condition)) or any(
                re.search(rb"(?<![A-Za-z0-9_$])" + re.escape(name) + rb"(?![A-Za-z0-9_$])", condition)
                for name in names
            )
            if not relevant:
                continue
            function = None
            replacement = b" true "
            operation = "neutralize_restriction"
            if intent == "MaxSell" and keyword.startswith(rb"\bif"):
                replacement = b" false "
            elif intent == "Honeypot" and keyword.startswith(rb"\bif"):
                brace = masked.find(b"{", end)
                brace_end = closing(masked, brace, 123, 125) if brace >= 0 else None
                body = masked[brace + 1:brace_end] if brace_end else b""
                replacement = b" false " if re.search(rb"revert|return\s+false", body, re.I) else b" true "
            edits.append((start, condition_end, replacement, operation, intent, function))
    return edits


def assigned_identifiers(function, source_bytes):
    if not function or not function["body"]:
        return []
    body = mask_noncode_bytes(source_bytes)[function["body"].start_byte:function["body"].end_byte]
    result = []
    for match in re.finditer(rb"(?<![.A-Za-z0-9_$])([A-Za-z_$][A-Za-z0-9_$]*)\s*=\s*([A-Za-z_$][A-Za-z0-9_$]*)", body):
        result.append(match.group(1).decode())
    return result


def annotation_statement_edits(source, source_bytes, entries, intent, functions):
    edits = []
    normalized = {normalize_line(entry.get("line", "")) for entry in entries}
    offset = 0
    for line in source.splitlines(keepends=True):
        encoded = line.encode("utf-8")
        text = normalize_line(line)
        if text in normalized and intent == "Honeypot" and re.search(r"=\s*0\s*;", text):
            statement_start = offset + encoded.find(line.lstrip().encode("utf-8"))
            semicolon = offset + encoded.rfind(b";") + 1
            function = containing_function(functions, statement_start)
            focus = function["node"].start_byte if function else None
            edits.append((statement_start, semicolon, b";", "neutralize_zeroing", intent, focus))
        offset += len(encoded)
    return edits


def collect_intent_edits(source, active_intents, annotations, parser):
    source_bytes = source.encode("utf-8")
    tree = parser.parse(source_bytes)
    if tree.root_node.has_error:
        return None, {"reason": "original_ast_error"}
    functions = function_records(tree, source_bytes)
    edits = []
    success = defaultdict(list)
    unmatched = {}

    def add(edit):
        if edit:
            edits.append(edit)
            success[edit[4]].append(edit[3])

    for intent in active_intents:
        entries = [entry for entry in annotations if entry["mapped_intent"] == intent]
        annotated_functions, missing = exact_annotation_functions(source, entries, functions)
        unmatched[intent] = missing

        if intent == "Mint":
            candidates = [
                function for function in functions
                if function_visibility(function) in {"public", "external"}
                and (
                    re.search(r"mint", function["name"], re.I)
                    or (function["body"] and b"_mint" in source_bytes[function["body"].start_byte:function["body"].end_byte])
                )
            ]
            for function in candidates:
                add(disable_edit(function, intent))

        elif intent == "Fee":
            candidates = [
                function for function in annotated_functions
                if function_visibility(function) in {"public", "external"}
                and re.search(r"fee|tax", function["name"], re.I)
            ]
            for function in candidates:
                add(disable_edit(function, intent))

        elif intent in {"Blacklist", "DisableTrading", "MaxTX", "MaxSell"}:
            name_pattern = TERM_PATTERNS[intent]
            assigned = []
            setters = [
                function for function in annotated_functions
                if function_visibility(function) in {"public", "external"}
                and name_pattern.search(function["name"].encode())
            ]
            for function in setters:
                assigned.extend(assigned_identifiers(function, source_bytes))
                add(disable_edit(function, intent))
            for edit in condition_edits(source_bytes, intent, assigned):
                position = edit[0]
                function = containing_function(functions, position)
                add((*edit[:5], function["node"].start_byte if function else None))
            if intent == "MaxSell" and not any(op == "neutralize_restriction" for op in success[intent]):
                success[intent].clear()

        elif intent == "Honeypot":
            assigned = []
            for function in annotated_functions:
                assigned.extend(assigned_identifiers(function, source_bytes))
                if function_visibility(function) in {"public", "external"}:
                    add(disable_edit(function, intent))
            for edit in condition_edits(source_bytes, intent, assigned):
                function = containing_function(functions, edit[0])
                add((*edit[:5], function["node"].start_byte if function else None))
            for edit in annotation_statement_edits(source, source_bytes, entries, intent, functions):
                add(edit)
            if not any(op in {"neutralize_restriction", "neutralize_zeroing"} for op in success[intent]):
                success[intent].clear()

        elif intent == "Reflection":
            payout = [
                function for function in functions
                if re.fullmatch(r"_?reflectFee|_takeReflection|_redistributeFee", function["name"], re.I)
            ]
            for function in payout:
                add(noop_edit(function, intent, "neutralize_reflection_state_update"))
            for function in functions:
                if function_visibility(function) in {"public", "external"} and re.fullmatch(r"reflect|deliver", function["name"], re.I):
                    add(disable_edit(function, intent))
            if not any(op == "neutralize_reflection_state_update" for op in success[intent]):
                success[intent].clear()

        elif intent == "Reward":
            payout_pattern = re.compile(r"swap.*dividend|send.*dividend|distributeDividend", re.I)
            payout = [function for function in functions if payout_pattern.search(function["name"])]
            for function in payout:
                add(noop_edit(function, intent, "neutralize_reward_payout"))
            for function in annotated_functions:
                if function_visibility(function) in {"public", "external"} and re.search(r"dividend|reward", function["name"], re.I):
                    add(disable_edit(function, intent))
            if not any(op == "neutralize_reward_payout" for op in success[intent]):
                success[intent].clear()

        elif intent == "Rebase":
            candidates = [
                function for function in annotated_functions
                if function_visibility(function) in {"public", "external"}
                and re.search(r"rebase|rebaseToken|^re$", function["name"], re.I)
            ]
            for function in candidates:
                add(disable_edit(function, intent))

    failed = [intent for intent in active_intents if not success[intent]]
    if failed:
        return None, {"reason": "unsupported_active_intent", "failed_intents": failed, "successful_operations": dict(success)}

    disabled_ranges = [
        (edit[0], edit[1]) for edit in edits if edit[3] in {"disable_entry", "neutralize_payout", "neutralize_reflection_state_update", "neutralize_reward_payout"}
    ]
    filtered = []
    for edit in edits:
        if edit[3] not in {"disable_entry", "neutralize_payout", "neutralize_reflection_state_update", "neutralize_reward_payout"} and any(
            left <= edit[0] and edit[1] <= right for left, right in disabled_ranges
        ):
            continue
        filtered.append(edit)
    deduplicated = {}
    for edit in filtered:
        deduplicated[(edit[0], edit[1])] = edit
    filtered = list(deduplicated.values())
    try:
        transformed_bytes = apply_edits(source_bytes, [(a, b, c, d) for a, b, c, d, _, _ in filtered])
    except Exception as exc:
        return None, {"reason": "edit_failure", "detail": str(exc)}
    transformed = transformed_bytes.decode("utf-8", "replace")
    transformed_tree = parser.parse(transformed_bytes)
    if transformed_tree.root_node.has_error or not balanced(transformed):
        return None, {"reason": "counterfactual_ast_error"}

    focus_starts = {edit[5] for edit in filtered if edit[5] is not None}
    original_focus_parts = [
        source_bytes[function["node"].start_byte:function["node"].end_byte].decode("utf-8", "replace")
        for function in functions if function["node"].start_byte in focus_starts
    ]
    focus_names = [function["name"] for function in functions if function["node"].start_byte in focus_starts]
    transformed_functions = function_records(transformed_tree, transformed_bytes)
    remaining_names = list(focus_names)
    transformed_focus_parts = []
    for function in transformed_functions:
        if function["name"] in remaining_names:
            transformed_focus_parts.append(
                transformed_bytes[function["node"].start_byte:function["node"].end_byte].decode("utf-8", "replace")
            )
            remaining_names.remove(function["name"])
    if not original_focus_parts or not transformed_focus_parts:
        return None, {"reason": "empty_focus"}
    return {
        "text": transformed,
        "original_focus_text": "\n\n/* NEXT FOCUS FUNCTION */\n\n".join(original_focus_parts),
        "focus_text": "\n\n/* NEXT FOCUS FUNCTION */\n\n".join(transformed_focus_parts),
        "operations_by_intent": dict(success),
        "unmatched_annotations": unmatched,
    }, None


def local_surface_variants(focus):
    variants = {
        "comment_insert": "/* behavior-preserving local control */\n" + focus,
        "format_normalized": format_normalized(focus),
    }
    renamed = identifier_surface(focus, max(0, len(focus) // 2))
    if renamed and renamed != focus:
        variants["local_parameter_rename"] = renamed
    return {name: text for name, text in variants.items() if text != focus and text.strip()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(OUT))
    parser.add_argument("--candidate-cap-per-intent", type=int, default=320)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(SOURCE)
    fixed = fixed_group_splits()
    positives = []
    global_counts = Counter()
    for row in rows:
        active = tuple(intent for intent in INTENTS if int(row["labels"].get(intent, 0)))
        for intent in active:
            global_counts[intent] += 1
        if active:
            row = dict(row)
            row["active_intents"] = active
            row["split"] = split_for(row, fixed)
            positives.append(row)
    # Select candidates independently for every intent. Otherwise rare
    # multi-label contracts (notably MaxSell) can consume Fee/Reflection/MaxTX
    # capacity before their much cleaner single-label examples are considered.
    candidate_by_contract = {}
    candidate_counts = Counter()
    for intent in INTENTS:
        pool = [
            row for row in positives
            if intent in row["active_intents"] and len(row["active_intents"]) <= 5
        ]
        pool.sort(key=lambda row: (
            len(row["active_intents"]) != 1,
            len(row["active_intents"]),
            stable_value(row.get("sample_key") or row.get("address") or row["contract_id"]),
        ))
        selected_for_intent = pool[:args.candidate_cap_per_intent]
        candidate_counts[intent] = len(selected_for_intent)
        for row in selected_for_intent:
            candidate_by_contract[str(row["contract_id"])] = row
    candidates = list(candidate_by_contract.values())
    candidates.sort(key=lambda row: (
        min(global_counts[intent] for intent in row["active_intents"]),
        len(row["active_intents"]),
        stable_value(row.get("sample_key") or row.get("address") or row["contract_id"]),
    ))
    annotations = read_annotations({str(row["contract_id"]) for row in candidates})
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, use_fast=True)
    solidity_parser = make_parser()
    unique_sources = []
    source_lookup = {}

    def add_source(text, focus_text, metadata):
        digest = hashlib.sha256((text + "\0" + focus_text).encode("utf-8")).hexdigest()
        if digest not in source_lookup:
            source_lookup[digest] = len(unique_sources)
            unique_sources.append({"text": text, "focus_text": focus_text, "sha256": digest, **metadata})
        return source_lookup[digest]

    accepted_counts = {split: Counter() for split in ("train", "val", "test")}
    pairs = []
    rejected = []
    for row in candidates:
        split = row["split"]
        active = list(row["active_intents"])
        if all(accepted_counts[split][intent] >= TARGETS[split][intent] for intent in active):
            continue
        entries = annotations.get(str(row["contract_id"]), [])
        if any(not any(entry["mapped_intent"] == intent for entry in entries) for intent in active):
            rejected.append({"contract_id": row["contract_id"], "active_intents": active, "reason": "missing_annotations"})
            continue
        built, error = collect_intent_edits(str(row["text"]), active, entries, solidity_parser)
        if error:
            rejected.append({"contract_id": row["contract_id"], "active_intents": active, **error})
            continue
        version = compiler_version(str(row["text"]))
        original_compile = compile_check(str(row["text"]), version)
        if not original_compile["passed"]:
            rejected.append({"contract_id": row["contract_id"], "active_intents": active, "reason": "original_compile_failure", "compile": original_compile})
            continue
        counterfactual_compile = compile_check(built["text"], version)
        if not counterfactual_compile["passed"]:
            rejected.append({"contract_id": row["contract_id"], "active_intents": active, "reason": "counterfactual_compile_failure", "compile": counterfactual_compile})
            continue
        metadata = {
            "contract_id": str(row["contract_id"]),
            "address": row.get("address", ""),
            "chain": row.get("chain", ""),
            "creator": row.get("group") or row.get("creator") or row.get("address"),
            "split": split,
            "active_intents": active,
            "original_label": 1,
            "counterfactual_label": 0,
            "validation_tier": "ast_and_compile",
        }
        original_index = add_source(str(row["text"]), built["original_focus_text"], {**metadata, "variant": "original"})
        counterfactual_index = add_source(built["text"], built["focus_text"], {**metadata, "variant": "counterfactual"})
        local_surfaces = []
        for name, local_text in local_surface_variants(built["original_focus_text"]).items():
            index = add_source(str(row["text"]), local_text, {**metadata, "variant": f"local_surface_{name}"})
            local_surfaces.append({
                "transform": name,
                "source_index": index,
                "token_edit_count_local": token_edit_count(tokenizer, built["original_focus_text"], local_text),
            })
        if len(local_surfaces) < 2:
            rejected.append({"contract_id": row["contract_id"], "active_intents": active, "reason": "insufficient_local_surfaces"})
            continue
        pair = {
            "pair_id": f"seed0_binary_v3_{len(pairs):04d}",
            **metadata,
            "original_source_index": original_index,
            "counterfactual_source_index": counterfactual_index,
            "counterfactual_token_edit_count_full": token_edit_count(tokenizer, str(row["text"]), built["text"]),
            "counterfactual_token_edit_count_local": token_edit_count(tokenizer, built["original_focus_text"], built["focus_text"]),
            "local_surface_variants": local_surfaces,
            "operations_by_intent": built["operations_by_intent"],
            "unmatched_annotations": built["unmatched_annotations"],
            "original_compile": original_compile,
            "counterfactual_compile": counterfactual_compile,
        }
        pairs.append(pair)
        for intent in active:
            accepted_counts[split][intent] += 1
        if all(
            accepted_counts[split_name][intent] >= TARGETS[split_name][intent]
            for split_name in ("train", "val", "test") for intent in INTENTS
        ):
            break

    for name, payload in (("paired_sources.jsonl", unique_sources), ("pair_manifest.jsonl", pairs), ("rejected.jsonl", rejected)):
        with open(output / name, "w", encoding="utf-8") as handle:
            for item in payload:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    summary = {
        "model": "CodeGemma-7B",
        "seed": 0,
        "source_pool_contracts": len(rows),
        "candidate_contracts": len(candidates),
        "candidate_counts_by_intent": dict(candidate_counts),
        "accepted_pairs": len(pairs),
        "unique_source_variants": len(unique_sources),
        "rejected": len(rejected),
        "targets": TARGETS,
        "counts_by_split_intent": {split: dict(accepted_counts[split]) for split in accepted_counts},
        "pairs_by_split": {split: sum(pair["split"] == split for pair in pairs) for split in ("train", "val", "test")},
        "single_label_pairs": sum(len(pair["active_intents"]) == 1 for pair in pairs),
        "multi_label_pairs": sum(len(pair["active_intents"]) > 1 for pair in pairs),
        "binary_rule": "Every active intent must be transformed; accepted counterfactual label is zero only after all active intents are removed.",
        "selection_rule": "Intent labels guide construction and macro averaging only; the final task remains binary unsafe versus normal.",
    }
    with open(output / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
