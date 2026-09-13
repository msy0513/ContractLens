#!/usr/bin/env python3
"""Build an exclusive-positive, approximately balanced cross-type protocol.

This is a revised manifest builder for the leave-one-intent-out experiment.
Unsafe test contracts may be multi-label, but each test contract is assigned
to exactly one active intent. Train/validation positives are downsampled to
the available normal controls in each fold. Test negatives are normal
controls sampled without replacement within each fold; the same normal pool
may be reused across folds because a leave-one-intent-out protocol otherwise
cannot provide a balanced test set with the current cached feature union.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

from intent_comparison_common import INTENTS, OUT, stable_key


SOURCE = OUT / "selected_union.jsonl"
TARGET = OUT / "cross_type_exclusive_balanced"
SPLITS = ("train", "val", "test")


def active_intents(row: dict) -> list[str]:
    return [intent for intent in INTENTS if int(row["labels"].get(intent, 0)) == 1]


def stable_rows(rows: list[dict], *parts: object) -> list[dict]:
    return sorted(rows, key=lambda row: stable_key(*parts, row["address"]))


def assign_test_owners(rows: list[dict]) -> dict[str, str]:
    """Assign every unsafe test row to one active intent only.

    Rows with fewer active intents are assigned first. Among eligible intents,
    the current smallest owner count is preferred, which spreads multi-label
    rows while respecting rare-intent availability.
    """
    candidates = []
    for row in rows:
        active = active_intents(row)
        if active:
            candidates.append((len(active), row["address"], active))
    candidates.sort(key=lambda item: (item[0], item[1]))
    counts = collections.Counter()
    owners = {}
    for _, address, active in candidates:
        owner = min(active, key=lambda intent: (counts[intent], INTENTS.index(intent)))
        owners[address] = owner
        counts[owner] += 1
    return owners


def normal_rows(rows: list[dict], split: str) -> list[dict]:
    return stable_rows(
        [row for row in rows if row["global_split"] == split and not active_intents(row)],
        "normal",
        split,
    )


def positive_rows(rows: list[dict], split: str, held: str) -> list[dict]:
    # Held-out labels are absent from train/validation by construction.
    return stable_rows(
        [
            row for row in rows
            if row["global_split"] == split
            and active_intents(row)
            and held not in active_intents(row)
        ],
        "train_positive",
        held,
        split,
    )


def record(row: dict, label: int, assigned_intent: str | None = None) -> dict:
    result = {
        "union_index": int(row["union_index"]),
        "label": int(label),
        "group": row.get("component", row.get("group")),
        "address": row["address"],
    }
    if assigned_intent is not None:
        result["assigned_test_intent"] = assigned_intent
        result["active_intents"] = active_intents(row)
    return result


def main() -> None:
    rows = []
    with SOURCE.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            row["union_index"] = index
            rows.append(row)

    by_index = {row["union_index"]: row for row in rows}
    test_unsafe = [
        row for row in rows
        if row["global_split"] == "test" and active_intents(row)
    ]
    owners = assign_test_owners(test_unsafe)
    if len(owners) != len(test_unsafe):
        raise RuntimeError("Every unsafe test row must receive exactly one owner")

    TARGET.mkdir(parents=True, exist_ok=True)
    audits = {}
    for held in INTENTS:
        records = {split: [] for split in SPLITS}

        for split in ("train", "val"):
            negatives = normal_rows(rows, split)
            candidates = positive_rows(rows, split, held)
            selected = stable_rows(candidates, "balanced_positive", held, split)[: len(negatives)]
            records[split].extend(record(row, 1) for row in selected)
            records[split].extend(record(row, 0) for row in negatives)

        test_negatives = normal_rows(rows, "test")
        test_positive = [
            row for row in test_unsafe if owners[row["address"]] == held
        ]
        test_positive = stable_rows(test_positive, "owned_test_positive", held)
        if len(test_positive) > len(test_negatives):
            test_positive = test_positive[: len(test_negatives)]
        test_negatives = stable_rows(test_negatives, "test_negative", held)[: len(test_positive)]
        records["test"].extend(record(row, 1, held) for row in test_positive)
        records["test"].extend(record(row, 0) for row in test_negatives)

        # Validate labels and group separation before writing the fold.
        for split in SPLITS:
            if not records[split]:
                raise RuntimeError(f"Empty {held}/{split} split")
            for item in records[split]:
                source = by_index[item["union_index"]]
                if split in ("train", "val") and held in active_intents(source):
                    raise RuntimeError(f"Held-out intent leaked into {held}/{split}")
                if split == "test" and item["label"] == 1:
                    if held not in active_intents(source):
                        raise RuntimeError(f"Invalid owned positive for {held}")
                    if item.get("assigned_test_intent") != held:
                        raise RuntimeError(f"Duplicate owner mismatch for {held}")

        groups = {split: {item["group"] for item in records[split]} for split in SPLITS}
        overlap = {
            f"{left}_{right}": len(groups[left] & groups[right])
            for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
        }
        audits[held] = {
            "counts": {
                split: {
                    "n": len(records[split]),
                    "positive": sum(item["label"] == 1 for item in records[split]),
                    "negative": sum(item["label"] == 0 for item in records[split]),
                }
                for split in SPLITS
            },
            "group_overlap": overlap,
            "test_positive_definition": "assigned owner intent is active; multi-label positives allowed",
            "test_negative_definition": "all ten intent labels are zero",
            "test_positive_owner_is_exclusive": True,
            "test_negative_pool_reused_across_folds": True,
        }
        (TARGET / f"{held}.json").write_text(
            json.dumps({"intent": held, "records": records, "audit": audits[held]}, indent=2),
            encoding="utf-8",
        )

    owner_counts = collections.Counter(owners.values())
    owner_manifest = {
        row["address"]: owners[row["address"]] for row in test_unsafe
    }
    (TARGET / "test_positive_owners.json").write_text(
        json.dumps({"counts": owner_counts, "owners": owner_manifest}, indent=2),
        encoding="utf-8",
    )
    audit = {
        "protocol": "exclusive test-positive owner assignment with per-fold class balancing",
        "source": str(SOURCE),
        "n_union": len(rows),
        "n_test_unsafe": len(test_unsafe),
        "test_positive_owner_counts": dict(owner_counts),
        "test_positive_owner_unique_rows": len(owner_manifest),
        "test_negative_pool_size": len(normal_rows(rows, "test")),
        "folds": audits,
    }
    (TARGET / "manifest_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
