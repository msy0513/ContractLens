#!/usr/bin/env python3
"""Select one balanced surface transform for every Seed-0 Train contract."""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("/root/autodl-tmp/models")
DATA = ROOT / "runs/stage8_codegemma_seed0_binary_cf_inv_v3/data_inv"
TRANSFORMS = [
    "comment_insert",
    "remove_comments",
    "format_normalized",
    "pragma_equivalent",
    "function_reordered",
    "local_parameter_rename",
]


def load_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stable(value):
    return hashlib.sha256(f"seed0-inv:{value}".encode()).hexdigest()


def main():
    sources = load_jsonl(DATA / "surface_sources.jsonl")
    variants = load_jsonl(DATA / "surface_manifest.jsonl")
    by_contract = defaultdict(list)
    for row in variants:
        by_contract[int(row["contract_index"])].append(row)
    selected = []
    counts = {0: Counter(), 1: Counter()}
    for label in (0, 1):
        contracts = [contract for contract, rows in by_contract.items() if int(rows[0]["label"]) == label]
        contracts.sort(key=lambda contract: stable(by_contract[contract][0]["creator"]))
        for contract in contracts:
            available = {row["transform"]: row for row in by_contract[contract]}
            minimum = min(counts[label][transform] for transform in available)
            tied = [transform for transform in TRANSFORMS if transform in available and counts[label][transform] == minimum]
            rotation = int(stable(by_contract[contract][0]["address"])[:8], 16) % len(tied)
            transform = tied[rotation]
            row = dict(available[transform])
            selected.append(row)
            counts[label][transform] += 1
    selected.sort(key=lambda row: int(row["contract_index"]))
    source_lookup = {}
    selected_sources = []
    for row in selected:
        source = sources[int(row["surface_source_index"])]
        digest = source["sha256"]
        if digest not in source_lookup:
            source_lookup[digest] = len(selected_sources)
            selected_sources.append(source)
        row["surface_source_index"] = source_lookup[digest]
    with open(DATA / "surface_sources_selected.jsonl", "w", encoding="utf-8") as handle:
        for row in selected_sources:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(DATA / "surface_manifest_selected.jsonl", "w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "train_contracts": len(by_contract),
        "selected_variants": len(selected),
        "unique_selected_sources": len(selected_sources),
        "one_variant_per_contract": len(selected) == len(by_contract) and len({row["contract_index"] for row in selected}) == len(by_contract),
        "counts_by_label_transform": {str(label): dict(counts[label]) for label in (0, 1)},
        "rationale": "Complete Train coverage with transform balancing within each binary class; prevents contracts with more valid variants from receiving larger weight.",
    }
    with open(DATA / "selected_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
