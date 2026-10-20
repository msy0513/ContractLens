#!/usr/bin/env python3
"""Build leakage-safe manifests and fetch BSC runtime bytecode."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import requests

from intent_comparison_common import (
    DATA, INTENTS, OLD_MANIFEST, OUT, SPLITS, SPLIT_RATIOS,
    atomic_json, atomic_jsonl, load_jsonl, normalized_source_hash,
    require_disk, stable_key,
)


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, item):
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[right] = left


def canonical_group(row: dict) -> str:
    group = str(row.get("group") or row.get("creator") or "").lower()
    return group or "address:" + str(row.get("address", "")).lower()


def largest_remainder(total: int) -> dict[str, int]:
    exact = {split: total * SPLIT_RATIOS[split] for split in SPLITS}
    result = {split: int(np.floor(exact[split])) for split in SPLITS}
    for split in sorted(SPLITS, key=lambda x: exact[x] - result[x], reverse=True)[:total - sum(result.values())]:
        result[split] += 1
    return result


def build_components(rows: list[dict]):
    uf = UnionFind()
    first_group_by_hash = {}
    for row in rows:
        group = canonical_group(row)
        source_hash = normalized_source_hash(row["text"])
        uf.find(group)
        if source_hash in first_group_by_hash:
            uf.union(group, first_group_by_hash[source_hash])
        else:
            first_group_by_hash[source_hash] = group
    components = collections.defaultdict(list)
    for index, row in enumerate(rows):
        components[uf.find(canonical_group(row))].append(index)
    return uf, dict(components)


def component_vector(rows, indices):
    vector = np.zeros(1 + len(INTENTS), dtype=np.float64)
    vector[0] = len(indices)
    for j, intent in enumerate(INTENTS, start=1):
        vector[j] = sum(int(rows[i]["labels"].get(intent, 0)) for i in indices)
    return vector


def assign_global_split(rows, uf, components, locked_groups):
    total_vector = component_vector(rows, range(len(rows)))
    targets = {split: total_vector * SPLIT_RATIOS[split] for split in SPLITS}
    state = {split: np.zeros_like(total_vector) for split in SPLITS}
    assignment = {}
    locked_roots = {uf.find(group) for group in locked_groups if group in uf.parent}
    for root in locked_roots:
        if root in components:
            assignment[root] = "train"
            state["train"] += component_vector(rows, components[root])

    unlocked = []
    for root, indices in components.items():
        if root not in assignment:
            unlocked.append((stable_key(0, root), root, indices, component_vector(rows, indices)))
    unlocked.sort(key=lambda item: (-len(item[2]), item[0]))

    def objective(candidate_split, vector):
        score = 0.0
        for split in SPLITS:
            projected = state[split] + (vector if split == candidate_split else 0)
            relative = (projected - targets[split]) / np.maximum(targets[split], 1.0)
            score += float(np.sum(relative ** 2))
            overflow = np.maximum(projected - targets[split], 0) / np.maximum(targets[split], 1.0)
            score += 4.0 * float(np.sum(overflow ** 2))
        return score

    for _, root, _, vector in unlocked:
        split = min(SPLITS, key=lambda name: (objective(name, vector), SPLITS.index(name)))
        assignment[root] = split
        state[split] += vector
    row_split = {}
    for root, indices in components.items():
        for index in indices:
            row_split[index] = assignment[root]
    return row_split, assignment, targets, state, locked_roots


def adjusted_allocation(total, availability):
    allocation = largest_remainder(total)
    while True:
        excess = sum(max(0, allocation[s] - availability[s]) for s in SPLITS)
        if excess == 0:
            return allocation
        for split in SPLITS:
            allocation[split] = min(allocation[split], availability[split])
        for _ in range(excess):
            candidates = [s for s in SPLITS if allocation[s] < availability[s]]
            if not candidates:
                raise RuntimeError("Insufficient split capacity")
            chosen = max(candidates, key=lambda s: total * SPLIT_RATIOS[s] - allocation[s])
            allocation[chosen] += 1


def build_manifest():
    require_disk()
    rows = load_jsonl(DATA)
    old = json.loads(OLD_MANIFEST.read_text(encoding="utf-8"))
    locked_groups = {
        str(record.get("group") or "").lower()
        for record in old["records"]
        if str(record["split"]).lower() in {"train", "val", "valid", "validation"}
    }
    uf, components = build_components(rows)
    row_split, assignment, targets, state, locked_roots = assign_global_split(rows, uf, components, locked_groups)

    normal_by_split = {}
    for split in SPLITS:
        candidates = [i for i, row in enumerate(rows) if row_split[i] == split and not any(row["labels"].values())]
        normal_by_split[split] = sorted(candidates, key=lambda i: stable_key(0, "normal", split, rows[i]["address"]))

    task_records = {}
    selected_indices = set()
    for intent in INTENTS:
        positive = {split: [i for i, row in enumerate(rows)
                            if row_split[i] == split and int(row["labels"].get(intent, 0)) == 1]
                    for split in SPLITS}
        cap = min(500, sum(len(v) for v in positive.values()))
        allocation = adjusted_allocation(cap, {s: len(positive[s]) for s in SPLITS})
        records = []
        for split in SPLITS:
            pos = sorted(positive[split], key=lambda i: stable_key(0, intent, "positive", split, rows[i]["address"]))[:allocation[split]]
            neg = normal_by_split[split][:allocation[split]]
            for label, indices in ((1, pos), (0, neg)):
                for index in indices:
                    selected_indices.add(index)
                    records.append({
                        "source_index": index,
                        "address": rows[index]["address"],
                        "group": canonical_group(rows[index]),
                        "component": uf.find(canonical_group(rows[index])),
                        "split": split,
                        "label": label,
                    })
        records.sort(key=lambda row: (SPLITS.index(row["split"]), row["label"], row["address"].lower()))
        task_records[intent] = {"intent": intent, "positive_cap": cap,
                                "allocation_per_class": allocation, "records": records}

    selected = sorted(selected_indices)
    source_to_union = {source: union for union, source in enumerate(selected)}
    union_rows = []
    for source in selected:
        row = dict(rows[source])
        row["source_index"] = source
        row["global_split"] = row_split[source]
        row["component"] = uf.find(canonical_group(row))
        union_rows.append(row)
    for intent, manifest in task_records.items():
        for record in manifest["records"]:
            record["union_index"] = source_to_union[record["source_index"]]
        atomic_json(OUT / "manifests" / "tasks" / f"{intent}.json", manifest)

    atomic_jsonl(OUT / "selected_union.jsonl", union_rows)
    split_groups = {split: {record["component"] for manifest in task_records.values()
                            for record in manifest["records"] if record["split"] == split}
                    for split in SPLITS}
    overlap = {"train_val": sorted(split_groups["train"] & split_groups["val"]),
               "train_test": sorted(split_groups["train"] & split_groups["test"]),
               "val_test": sorted(split_groups["val"] & split_groups["test"])}
    if any(overlap.values()):
        raise RuntimeError(f"Component leakage: {overlap}")
    audit = {
        "seed": 0,
        "source": str(DATA),
        "source_rows": len(rows),
        "selected_union_rows": len(union_rows),
        "component_count": len(components),
        "locked_old_development_groups": len(locked_groups),
        "locked_components": len(locked_roots),
        "global_target_vectors": {s: targets[s].tolist() for s in SPLITS},
        "global_actual_vectors": {s: state[s].tolist() for s in SPLITS},
        "task_counts": {intent: {split: collections.Counter(r["label"] for r in manifest["records"] if r["split"] == split)
                                   for split in SPLITS} for intent, manifest in task_records.items()},
        "component_overlap": overlap,
        "negative_definition": "all ten intent labels are zero",
        "old_binary_train_validation_locked_to_new_train": True,
    }
    atomic_json(OUT / "manifests" / "global_split_audit.json", audit)
    atomic_json(OUT / "config.json", {
        "seed": 0, "data": str(DATA), "old_manifest": str(OLD_MANIFEST),
        "split_ratios": SPLIT_RATIOS, "positive_cap": 500,
        "negative_definition": "all ten labels zero", "max_length": 2048,
        "segments": 256, "backbone_frozen": True, "accuracy_threshold": 0.5,
        "intents": INTENTS, "gptlens_enabled": False,
        "late_fusion_lambda_codegemma": [0.25, 0.5, 0.75, 1.0],
        "native_only_lambda_zero_reported_separately": True,
    })
    external = OUT.parent / "external_baselines" / "malicious_contract_methods"
    paper_sources = [
        {
            "method": "Trapdoor-LGBM",
            "paper": "From Programming Bugs to Multimillion-Dollar Scams: An Analysis of Trapdoor Tokens on Uniswap",
            "paper_url": "https://arxiv.org/abs/2309.04700",
            "paper_file": external / "trapdoor_tokens" / "paper_arxiv_2309.04700.pdf",
            "code_url": "https://github.com/bsdp2023/trapdoor-scam",
            "code_status": "official paper-linked artifact; full semantic TrapdoorAnalyser not published",
        },
        {
            "method": "Honeypot-XGBoost",
            "paper": "A Data Science Approach for Honeypot Detection in Ethereum",
            "paper_url": "https://arxiv.org/abs/1910.01449",
            "paper_file": external / "honeypot_detection" / "paper_arxiv_1910.01449.pdf",
            "code_url": "https://github.com/rcamino/honeypot-detection",
            "code_status": "official repository commit c814790a976fa5da0b5ee4745a8db4385f474336",
        },
        {
            "method": "SCSGuard",
            "paper": "SCSGuard: Deep Scam Detection for Ethereum Smart Contracts",
            "paper_url": "https://arxiv.org/abs/2105.10426",
            "paper_file": external / "scsguard_paper_arxiv_2105.10426.pdf",
            "code_url": None,
            "code_status": "no verified official repository; independent architecture reproduction",
        },
        {
            "method": "Opcode-XAI",
            "paper": "An Explainable AI Model for Detecting Malicious Smart Contracts Based on EVM Opcode Features",
            "paper_url": "https://arxiv.org/abs/2512.08782",
            "paper_file": external / "opcode_xai_paper_arxiv_2512.08782.pdf",
            "code_url": None,
            "code_status": "no verified official repository; independent paper reproduction",
        },
        {
            "method": "SmartIntentNN/SmartIntentV2",
            "paper": "SmartIntent method family",
            "paper_url": "https://arxiv.org/abs/2211.13670",
            "paper_file": None,
            "code_url": "https://github.com/web3se-lab/SmartIntent-py",
            "code_status": "official repository available locally; pretrained USE/SmartBERT weights not reused",
        },
        {
            "method": "GPTLens",
            "paper": "GPTLens",
            "paper_url": "https://arxiv.org/abs/2310.01152",
            "paper_file": None,
            "code_url": "https://github.com/git-disl/GPTLens",
            "code_status": "not run by user decision because of generation cost",
        },
    ]
    inventory = []
    for source in paper_sources:
        paper_file = source.pop("paper_file")
        local = None
        digest = None
        if paper_file is not None and paper_file.exists():
            local = str(paper_file)
            digest = hashlib.sha256(paper_file.read_bytes()).hexdigest()
        inventory.append({**source, "local_paper": local, "paper_sha256": digest})
    atomic_json(OUT / "paper_inventory.json", {
        "checked_at": "2026-08-24", "gptlens_run": False, "sources": inventory,
        "main_table_policy": "frozen CodeGemma adaptations",
        "native_table_policy": "available paper-native branches on the same local split",
    })
    print(json.dumps({"stage": "manifest", "selected_union_rows": len(union_rows),
                      "task_caps": {k: v["positive_cap"] for k, v in task_records.items()}}, indent=2))


def parse_opcode_counts(bytecode: str) -> np.ndarray:
    raw = str(bytecode).removeprefix("0x")
    data = bytes.fromhex(raw) if raw else b""
    counts = np.zeros(256, dtype=np.float32)
    index = 0
    while index < len(data):
        opcode = data[index]
        counts[opcode] += 1
        index += 1 + (opcode - 0x5F if 0x60 <= opcode <= 0x7F else 0)
    return counts


def fetch_bytecode():
    require_disk()
    rows = load_jsonl(OUT / "selected_union.jsonl")
    cache_path = OUT / "bytecode" / "runtime_bytecode.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    rpc_url = os.environ.get("BSC_RPC_URL", "https://bsc-rpc.publicnode.com")
    session = requests.Session()
    pending = [(index, str(row["address"]).lower()) for index, row in enumerate(rows)
               if str(row["address"]).lower() not in cache]
    for start in range(0, len(pending), 25):
        batch = pending[start:start + 25]
        payload = [
            {"jsonrpc": "2.0", "id": index + 1, "method": "eth_getCode", "params": [address, "latest"]}
            for index, address in batch
        ]
        received, batch_error = {}, None
        for attempt in range(4):
            try:
                response = session.post(rpc_url, json=payload, timeout=60)
                response.raise_for_status()
                body = response.json()
                if not isinstance(body, list):
                    raise RuntimeError(f"RPC batch returned {type(body).__name__}")
                received = {int(item["id"]): item for item in body}
                if len(received) == len(batch):
                    break
            except Exception as exc:
                batch_error = repr(exc)
            time.sleep(1.5 * (attempt + 1))
        for index, address in batch:
            item = received.get(index + 1, {})
            cache[address] = {"bytecode": item.get("result"),
                              "error": item.get("error", batch_error or "missing RPC batch response")}
        atomic_json(cache_path, cache)
        print(json.dumps({"stage": "bytecode", "cached": len(cache), "total": len(rows)}), flush=True)
    atomic_json(cache_path, cache)

    opcode_path = OUT / "bytecode" / "opcode_counts.f32.npy"
    opcode_path.parent.mkdir(parents=True, exist_ok=True)
    opcodes = np.lib.format.open_memmap(opcode_path, mode="w+", dtype=np.float32, shape=(len(rows), 256))
    available = []
    for index, row in enumerate(rows):
        record = cache.get(str(row["address"]).lower(), {})
        code = record.get("bytecode")
        valid = isinstance(code, str) and code not in {"", "0x", "0x0"}
        available.append(valid)
        opcodes[index] = parse_opcode_counts(code) if valid else 0
    opcodes.flush()
    availability = np.asarray(available, dtype=np.bool_)
    np.save(OUT / "bytecode" / "available.bool.npy", availability)

    complete_case = {}
    for intent in INTENTS:
        manifest = json.loads((OUT / "manifests" / "tasks" / f"{intent}.json").read_text(encoding="utf-8"))
        records = [record for record in manifest["records"] if availability[record["union_index"]]]
        complete_case[intent] = {split: {str(label): sum(r["split"] == split and r["label"] == label for r in records)
                                                 for label in (0, 1)} for split in SPLITS}
        atomic_json(OUT / "manifests" / "tasks_complete_case" / f"{intent}.json",
                    {**manifest, "records": records, "bytecode_complete_case": True})
    atomic_json(OUT / "bytecode_audit.json", {
        "rpc": rpc_url, "selected": len(rows), "available": int(availability.sum()),
        "missing": int((~availability).sum()), "availability_rate": float(availability.mean()),
        "complete_case_counts": complete_case,
    })
    print(json.dumps({"stage": "bytecode", "available": int(availability.sum()),
                      "total": len(rows), "rate": float(availability.mean())}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["manifest", "bytecode", "all"], default="all")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage in {"manifest", "all"}:
        build_manifest()
    if args.stage in {"bytecode", "all"}:
        fetch_bytecode()


if __name__ == "__main__":
    main()
