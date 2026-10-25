#!/usr/bin/env python3
"""Build strict Seed-0 binary unsafe-to-normal counterfactual pairs."""

import gzip
import hashlib
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import solcx
from transformers import AutoTokenizer
from tree_sitter import Language, Parser
import tree_sitter_solidity

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_smartintent import parse_scalar, tuples_from_insert
from build_smartintent_controls import anonymize_identifiers, strip_comments_and_headers
from build_seed0_cf_inv_pairs import balanced, identifier_surface, token_edit_count


ROOT = Path("/root/autodl-tmp/models")
UNION = ROOT / "runs/stage3_smartintent_sliding/sliding_union.jsonl"
MANIFEST = ROOT / "runs/stage4_smartintent_full_trials/manifests/trial_seed_0.json"
SQL_DUMP = ROOT / "runs/data/smartintent/web3_all.sql.gz"
MODEL = ROOT / "base/CodeGemma-7B"
OUT = ROOT / "runs/stage7_codegemma_seed0_binary_cf_inv_v2/data"

INTENTS = [
    "Blacklist", "Mint", "Honeypot", "Fee", "DisableTrading",
    "MaxSell", "MaxTX", "Reflection", "Reward", "Rebase",
]
SUPPORTED = {"Mint", "Fee", "DisableTrading", "MaxTX"}
TYPE_MAP = {
    "blacklist": "Blacklist", "mint": "Mint", "honeypot": "Honeypot",
    "fee": "Fee", "disabletrading": "DisableTrading", "maxsell": "MaxSell",
    "maxtx": "MaxTX", "reflect": "Reflection", "reflection": "Reflection",
    "reward": "Reward", "rebase": "Rebase",
}
COMPILER_BY_MINOR = {
    (0, 4): "0.4.26",
    (0, 5): "0.5.17",
    (0, 6): "0.6.12",
    (0, 7): "0.7.6",
    (0, 8): "0.8.26",
}


def load_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def make_parser():
    return Parser(Language(tree_sitter_solidity.language()))


def walk(node):
    yield node
    for child in node.children:
        yield from walk(child)


def function_records(tree, source_bytes):
    records = []
    for node in walk(tree.root_node):
        if node.type not in {"function_definition", "constructor_definition", "fallback_receive_definition"}:
            continue
        body = node.child_by_field_name("body")
        name_node = node.child_by_field_name("name")
        name = source_bytes[name_node.start_byte:name_node.end_byte].decode("utf-8", "replace") if name_node else ""
        header_end = body.start_byte if body else node.end_byte
        header = source_bytes[node.start_byte:header_end].decode("utf-8", "replace")
        records.append({"node": node, "body": body, "name_node": name_node, "name": name, "header": header})
    return records


def containing_function(functions, position):
    candidates = [row for row in functions if row["node"].start_byte <= position < row["node"].end_byte]
    return min(candidates, key=lambda row: row["node"].end_byte - row["node"].start_byte) if candidates else None


def read_annotations(wanted_ids):
    found = {}
    in_tokens = False
    with gzip.open(SQL_DUMP, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("INSERT INTO `tokens`"):
                in_tokens = True
                continue
            if in_tokens and line.startswith("UNLOCK TABLES"):
                break
            if not in_tokens or "(" not in line:
                continue
            for row in tuples_from_insert(line):
                if len(row) < 6 or str(row[5]) not in wanted_ids:
                    continue
                try:
                    raw = json.loads(parse_scalar(row[4]) or "[]")
                except Exception:
                    raw = []
                entries = []
                for item in raw if isinstance(raw, list) else []:
                    if not isinstance(item, dict):
                        continue
                    mapped = TYPE_MAP.get(str(item.get("type", "")).lower())
                    if mapped:
                        entries.append({**item, "mapped_intent": mapped})
                found[str(row[5])] = entries
    return found


def source_lines(source):
    result = []
    offset = 0
    for line in source.splitlines(keepends=True):
        encoded = line.encode("utf-8")
        result.append({"text": line, "stripped": line.strip(), "start": offset, "end": offset + len(encoded)})
        offset += len(encoded)
    return result


def annotation_positions(source, entries):
    lines = source_lines(source)
    positions = []
    unmatched = []
    for entry in entries:
        target = str(entry.get("line", "")).strip()
        if not target:
            continue
        matches = [row for row in lines if row["stripped"] == target]
        if not matches:
            matches = [row for row in lines if target in row["stripped"] or row["stripped"] in target]
        if not matches:
            unmatched.append(target)
            continue
        for match in matches:
            leading = len(match["text"].encode("utf-8")) - len(match["text"].lstrip().encode("utf-8"))
            positions.append(match["start"] + leading)
    return sorted(set(positions)), unmatched


def mask_noncode_bytes(data):
    out = bytearray(data)
    index = 0
    state = "code"
    quote = 0
    while index < len(data):
        if state == "code":
            if data[index:index + 2] == b"//":
                out[index:index + 2] = b"  "; index += 2; state = "line"
            elif data[index:index + 2] == b"/*":
                out[index:index + 2] = b"  "; index += 2; state = "block"
            elif data[index:index + 1] in {b"'", b'"'}:
                quote = data[index]; out[index] = 32; index += 1; state = "string"
            else:
                index += 1
        elif state == "line":
            if data[index] in {10, 13}:
                state = "code"
            else:
                out[index] = 32
            index += 1
        elif state == "block":
            if data[index:index + 2] == b"*/":
                out[index:index + 2] = b"  "; index += 2; state = "code"
            else:
                if data[index] not in {10, 13}:
                    out[index] = 32
                index += 1
        else:
            if data[index] == 92 and index + 1 < len(data):
                out[index:index + 2] = b"  "; index += 2
            elif data[index] == quote:
                out[index] = 32; index += 1; state = "code"
            else:
                if data[index] not in {10, 13}:
                    out[index] = 32
                index += 1
    return bytes(out)


def closing_paren(masked, opening):
    depth = 1
    for index in range(opening + 1, len(masked)):
        if masked[index] == 40:
            depth += 1
        elif masked[index] == 41:
            depth -= 1
            if depth == 0:
                return index
    return None


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


def require_condition_edits(source_bytes, intent):
    masked = mask_noncode_bytes(source_bytes)
    if intent == "DisableTrading":
        pattern = re.compile(rb"trad(e|ing)|tradingopen|tradingenabled|swapenabled", re.I)
    elif intent == "MaxTX":
        pattern = re.compile(rb"maxtx|maxtransaction|txlimit|maxtransfer|max_tx|maxamount", re.I)
    else:
        return []
    edits = []
    for match in re.finditer(rb"\brequire\s*\(", masked):
        opening = masked.find(b"(", match.start())
        closing = closing_paren(masked, opening)
        if closing is None:
            continue
        start, end = first_argument(masked, opening, closing)
        if pattern.search(masked[start:end]):
            edits.append((start, end, b" true ", "neutralize_restriction"))
    return edits


def neutral_name(old, source_bytes, ordinal):
    if not old:
        return old
    prefix = "_" if old.startswith("_") else ""
    width = len(old) - len(prefix)
    alphabet = chr(ord("n") + (ordinal % 10))
    candidate = prefix + alphabet * max(width, 1)
    while re.search(rb"\b" + re.escape(candidate.encode()) + rb"\b", source_bytes):
        alphabet = chr(ord(alphabet) + 1) if alphabet < "z" else "q"
        candidate = prefix + alphabet * max(width, 1)
    return candidate


def apply_edits(data, edits):
    ordered = sorted(edits, key=lambda item: (item[0], item[1]), reverse=True)
    last_start = len(data) + 1
    for start, end, replacement, _ in ordered:
        if end > last_start:
            raise ValueError("overlapping edits")
        data = data[:start] + replacement + data[end:]
        last_start = start
    return data


def build_counterfactual(source, intent, entries, parser):
    source_bytes = source.encode("utf-8")
    tree = parser.parse(source_bytes)
    if tree.root_node.has_error:
        return None, {"reason": "original_ast_error"}
    functions = function_records(tree, source_bytes)
    positions, unmatched = annotation_positions(source, entries)
    annotated = []
    for position in positions:
        function = containing_function(functions, position)
        if function and function["body"]:
            annotated.append(function)
    unique = {row["node"].start_byte: row for row in annotated}
    annotated = list(unique.values())
    disabled = []
    rename_only = []
    if intent == "Mint":
        for function in annotated:
            header = function["header"].lower()
            name = function["name"]
            if "constructor" in header:
                continue
            if re.search(r"\b(public|external)\b", header):
                disabled.append(function)
            elif name and "mint" in name.lower():
                rename_only.append(function)
        if not disabled:
            return None, {"reason": "no_external_mint_function", "unmatched": unmatched}
    elif intent == "Fee":
        disabled = [row for row in annotated if re.search(r"fee", row["name"], re.I)]
        if not disabled:
            return None, {"reason": "no_annotated_fee_function", "unmatched": unmatched}
    elif intent in {"DisableTrading", "MaxTX"}:
        term = re.compile(r"trad|open|status" if intent == "DisableTrading" else r"max|tx|limit", re.I)
        disabled = [row for row in annotated if row["name"] and term.search(row["name"])]
    else:
        return None, {"reason": "unsupported_intent"}

    restriction_edits = require_condition_edits(source_bytes, intent)
    if intent in {"DisableTrading", "MaxTX"} and not disabled and not restriction_edits:
        return None, {"reason": "no_high_confidence_restriction_or_setter", "unmatched": unmatched}

    edits = list(restriction_edits)
    rename_map = {}
    focus_names = []
    target_functions = {row["node"].start_byte: row for row in disabled + rename_only}
    for ordinal, function in enumerate(target_functions.values()):
        old = function["name"]
        if not old or not function["name_node"]:
            continue
        new = neutral_name(old, source_bytes, ordinal)
        rename_map[old] = new
        focus_names.append(old)
    for old, new in rename_map.items():
        for match in re.finditer(rb"(?<![A-Za-z0-9_$])" + re.escape(old.encode()) + rb"(?![A-Za-z0-9_$])", mask_noncode_bytes(source_bytes)):
            edits.append((match.start(), match.end(), new.encode(), "rename_intent_function"))
    for function in disabled:
        body = function["body"]
        replacement = b'\n        revert("unsafe intent removed");\n    '
        edits.append((body.start_byte + 1, body.end_byte - 1, replacement, "disable_intent_function"))

    disabled_ranges = [(row["body"].start_byte, row["body"].end_byte) for row in disabled]
    filtered = []
    for edit in edits:
        if edit[3] in {"neutralize_restriction", "rename_intent_function"} and any(
            left < edit[0] < right for left, right in disabled_ranges
        ):
            continue
        filtered.append(edit)
    try:
        counterfactual_bytes = apply_edits(source_bytes, filtered)
        counterfactual = counterfactual_bytes.decode("utf-8")
    except Exception as exc:
        return None, {"reason": "edit_failure", "detail": str(exc)}
    cf_tree = parser.parse(counterfactual_bytes)
    if cf_tree.root_node.has_error or not balanced(counterfactual):
        return None, {"reason": "counterfactual_ast_error"}

    focus_functions = []
    focus_starts = {row["node"].start_byte for row in disabled + rename_only}
    for edit in restriction_edits:
        function = containing_function(functions, edit[0])
        if function:
            focus_starts.add(function["node"].start_byte)
    for function in functions:
        if function["node"].start_byte in focus_starts:
            focus_functions.append(source_bytes[function["node"].start_byte:function["node"].end_byte].decode("utf-8", "replace"))
    original_focus = "\n\n/* NEXT FOCUS FUNCTION */\n\n".join(focus_functions)
    if not original_focus:
        return None, {"reason": "empty_focus"}
    cf_functions = function_records(cf_tree, counterfactual_bytes)
    wanted_names = set(rename_map.values()) | {row["name"] for row in functions if row["node"].start_byte in focus_starts and row["name"] not in rename_map}
    cf_focus_parts = [
        counterfactual_bytes[row["node"].start_byte:row["node"].end_byte].decode("utf-8", "replace")
        for row in cf_functions if row["name"] in wanted_names
    ]
    counterfactual_focus = "\n\n/* NEXT FOCUS FUNCTION */\n\n".join(cf_focus_parts)
    if not counterfactual_focus:
        return None, {"reason": "empty_counterfactual_focus"}
    return {
        "text": counterfactual,
        "focus_text": counterfactual_focus,
        "original_focus_text": original_focus,
        "edit_operations": [item[3] for item in filtered],
        "rename_map": rename_map,
        "unmatched_annotations": unmatched,
    }, None


def format_normalized(text):
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    return "\n".join(line.replace("\t", "    ").rstrip() for line in lines)


def pragma_rewrite(text):
    pattern = re.compile(r"(?mi)^\s*pragma\s+solidity\s+([^;]+);")
    match = pattern.search(text)
    if not match:
        return text
    versions = re.findall(r"(\d+)\.(\d+)\.(\d+)", match.group(1))
    if not versions:
        return text
    major, minor, patch = versions[0]
    replacement = f"pragma solidity >={major}.{minor}.{patch} <0.{int(minor) + 1}.0;"
    return text[:match.start()] + replacement + text[match.end():]


def reorder_functions(text, parser):
    data = text.encode("utf-8")
    tree = parser.parse(data)
    by_parent = defaultdict(list)
    for row in function_records(tree, data):
        if row["node"].type == "function_definition" and row["body"]:
            by_parent[(row["node"].parent.start_byte, row["node"].parent.end_byte)].append(row)
    candidates = next((rows for rows in by_parent.values() if len(rows) >= 2), None)
    if not candidates:
        return text
    first, second = candidates[0], candidates[1]
    a = data[first["node"].start_byte:first["node"].end_byte]
    b = data[second["node"].start_byte:second["node"].end_byte]
    return apply_edits(data, [
        (first["node"].start_byte, first["node"].end_byte, b, "reorder"),
        (second["node"].start_byte, second["node"].end_byte, a, "reorder"),
    ]).decode("utf-8")


def surface_variants(source, focus, parser):
    comment_removed = strip_comments_and_headers(source)
    focus_comment_removed = strip_comments_and_headers(focus)
    anonymous = anonymize_identifiers(source)
    focus_anonymous = anonymize_identifiers(focus)
    variants = {
        "remove_comments_headers": (comment_removed, focus_comment_removed),
        "identifier_anonymized": (anonymous, focus_anonymous),
        "format_normalized": (format_normalized(source), format_normalized(focus)),
        "pragma_equivalent": (pragma_rewrite(source), pragma_rewrite(focus)),
        "function_reordered": (reorder_functions(source, parser), reorder_functions(focus, parser)),
    }
    position = max(source.find(focus[: min(80, len(focus))]), 0)
    renamed = identifier_surface(source, position)
    if renamed and renamed != source:
        variants["local_parameter_rename"] = (renamed, focus)
    marker = "/* semantic-preserving control annotation */\n"
    variants["comment_insert"] = (source[:position] + marker + source[position:], marker + focus)
    valid = {}
    for name, (full, local) in variants.items():
        if full == source or not local.strip() or not balanced(full):
            continue
        tree = parser.parse(full.encode("utf-8"))
        local_tree = parser.parse(local.encode("utf-8"))
        if not tree.root_node.has_error and not local_tree.root_node.has_error:
            valid[name] = (full, local)
    return valid


def compiler_version(source):
    match = re.search(r"(?i)pragma\s+solidity\s+([^;]+);", source)
    if not match:
        return None
    version = re.search(r"(\d+)\.(\d+)\.(\d+)", match.group(1))
    if not version:
        return None
    return COMPILER_BY_MINOR.get((int(version.group(1)), int(version.group(2))))


def compile_check(source, version):
    if not version:
        return {"attempted": False, "passed": False, "reason": "unsupported_or_missing_pragma"}
    normalized = re.sub(
        r"(?i)pragma\s+solidity\s+[^;]+;",
        f"pragma solidity {version};",
        source,
    )
    executable = str(solcx.install.get_executable(version))
    try:
        completed = subprocess.run(
            [executable, "--combined-json", "abi", "-"],
            input=normalized.encode("utf-8"), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            timeout=30, check=False,
        )
    except subprocess.TimeoutExpired:
        return {"attempted": True, "passed": False, "reason": "timeout", "version": version}
    error = completed.stderr.decode("utf-8", "replace")[-1000:]
    return {
        "attempted": True,
        "passed": completed.returncode == 0,
        "reason": "ok" if completed.returncode == 0 else error,
        "version": version,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(UNION)
    manifest = json.load(open(MANIFEST, encoding="utf-8"))
    selected = []
    for record in manifest["records"]:
        row = rows[int(record["feature_index"])]
        active = [intent for intent in INTENTS if int(row["labels"].get(intent, 0))]
        if len(active) == 1 and active[0] in SUPPORTED:
            selected.append((record, row, active[0]))
    wanted_ids = {str(row["contract_id"]) for _, row, _ in selected}
    annotations = read_annotations(wanted_ids)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, use_fast=True)
    parser = make_parser()
    unique_sources = []
    source_lookup = {}

    def add_source(text, focus_text, metadata):
        digest = hashlib.sha256((text + "\0" + focus_text).encode("utf-8")).hexdigest()
        if digest not in source_lookup:
            source_lookup[digest] = len(unique_sources)
            unique_sources.append({"text": text, "focus_text": focus_text, "sha256": digest, **metadata})
        return source_lookup[digest]

    pairs = []
    rejected = []
    compile_counts = defaultdict(int)
    for record, row, intent in selected:
        contract_id = str(row["contract_id"])
        intent_entries = [item for item in annotations.get(contract_id, []) if item["mapped_intent"] == intent]
        if not intent_entries:
            rejected.append({"contract_id": contract_id, "intent": intent, "reason": "missing_raw_annotations"})
            continue
        built, error = build_counterfactual(str(row["text"]), intent, intent_entries, parser)
        if error:
            rejected.append({"contract_id": contract_id, "intent": intent, **error})
            continue
        surfaces = surface_variants(str(row["text"]), built["original_focus_text"], parser)
        if len(surfaces) < 4:
            rejected.append({"contract_id": contract_id, "intent": intent, "reason": "insufficient_surface_variants", "count": len(surfaces)})
            continue
        version = compiler_version(str(row["text"]))
        original_compile = compile_check(str(row["text"]), version)
        counterfactual_compile = compile_check(built["text"], version) if original_compile["passed"] else {
            "attempted": False, "passed": False, "reason": "original_did_not_compile", "version": version,
        }
        if original_compile["passed"] and not counterfactual_compile["passed"]:
            rejected.append({
                "contract_id": contract_id, "intent": intent,
                "reason": "counterfactual_compile_regression",
                "original_compile": original_compile,
                "counterfactual_compile": counterfactual_compile,
            })
            continue
        tier = "ast_and_compile" if original_compile["passed"] and counterfactual_compile["passed"] else "ast_only"
        compile_counts[tier] += 1
        metadata = {
            "address": record["address"], "creator": record["group"], "split": record["split"],
            "original_label": 1, "counterfactual_label": 0, "intent": intent,
            "contract_id": contract_id, "validation_tier": tier,
        }
        original_index = add_source(str(row["text"]), built["original_focus_text"], {**metadata, "variant": "original"})
        counterfactual_index = add_source(built["text"], built["focus_text"], {**metadata, "variant": "counterfactual"})
        cf_edits_full = token_edit_count(tokenizer, str(row["text"]), built["text"])
        cf_edits_local = token_edit_count(tokenizer, built["original_focus_text"], built["focus_text"])
        surface_rows = []
        for name, (surface_text, surface_focus) in surfaces.items():
            index = add_source(surface_text, surface_focus, {**metadata, "variant": name})
            surface_rows.append({
                "transform": name,
                "source_index": index,
                "token_edit_count_full": token_edit_count(tokenizer, str(row["text"]), surface_text),
                "token_edit_count_local": token_edit_count(tokenizer, built["original_focus_text"], surface_focus),
            })
        matched_full = min(surface_rows, key=lambda item: abs(item["token_edit_count_full"] - cf_edits_full))
        matched_local = min(surface_rows, key=lambda item: abs(item["token_edit_count_local"] - cf_edits_local))
        pairs.append({
            "pair_id": f"seed0_binary_v2_{len(pairs):04d}",
            **metadata,
            "feature_index": int(record["feature_index"]),
            "original_source_index": original_index,
            "counterfactual_source_index": counterfactual_index,
            "counterfactual_token_edit_count_full": cf_edits_full,
            "counterfactual_token_edit_count_local": cf_edits_local,
            "matched_surface_full": matched_full,
            "matched_surface_local": matched_local,
            "surface_variants": surface_rows,
            "edit_operations": built["edit_operations"],
            "rename_map": built["rename_map"],
            "raw_annotation_count": len(intent_entries),
            "unmatched_annotations": built["unmatched_annotations"],
            "original_compile": original_compile,
            "counterfactual_compile": counterfactual_compile,
        })

    for name, payload in (
        ("paired_sources.jsonl", unique_sources),
        ("pair_manifest.jsonl", pairs),
        ("rejected.jsonl", rejected),
    ):
        with open(OUT / name, "w", encoding="utf-8") as handle:
            for item in payload:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    summary = {
        "model": "CodeGemma-7B", "seed": 0, "source_task_contracts": 2000,
        "strict_single_intent_candidates": len(selected), "pair_count": len(pairs),
        "unique_source_variants": len(unique_sources), "rejected_count": len(rejected),
        "supported_intents": sorted(SUPPORTED),
        "pairs_by_split": {split: sum(item["split"] == split for item in pairs) for split in ("train", "val", "test")},
        "pairs_by_intent": {
            intent: {split: sum(item["intent"] == intent and item["split"] == split for item in pairs) for split in ("train", "val", "test")}
            for intent in sorted(SUPPORTED)
        },
        "validation_tiers": dict(compile_counts),
        "binary_rule": "original has exactly one unsafe label; counterfactual removes that complete annotated capability",
        "ast_validation": "Tree-sitter Solidity must parse original, counterfactual, local focus, and every surface variant without errors",
        "compile_validation": "If original compiles under a compatible Solidity 0.4-0.8 compiler, counterfactual must also compile",
    }
    with open(OUT / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
