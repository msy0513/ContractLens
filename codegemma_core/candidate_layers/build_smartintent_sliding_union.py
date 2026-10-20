#!/usr/bin/env python3
"""Build a deduplicated source manifest and indexed probe task files."""
import argparse
import glob
import json
from pathlib import Path


def key_for(row):
    return f"{row.get('chain', '')}:{row.get('address', '')}".lower()


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    data_dir, out = Path(args.data_dir), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = {}
    tasks = {}
    sources = [("binary", data_dir / "binary_probe_balanced.jsonl")]
    sources += [
        (Path(p).stem.removeprefix("intent_"), Path(p))
        for p in sorted(glob.glob(str(data_dir / "intent_*.jsonl")))
    ]
    for task, path in sources:
        task_rows = {}
        for row in read_jsonl(path):
            key = key_for(row)
            rows.setdefault(key, row)
            task_rows[key] = int(row["label"])
        tasks[task] = task_rows

    ordered = sorted(rows.items())
    index = {key: i for i, (key, _) in enumerate(ordered)}
    manifest = out / "sliding_union.jsonl"
    with open(manifest, "w", encoding="utf-8") as f:
        for i, (key, row) in enumerate(ordered):
            item = dict(row)
            item["feature_index"] = i
            item["sample_key"] = key
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    task_dir = out / "sliding_tasks"
    task_dir.mkdir(exist_ok=True)
    summary = {"union_rows": len(ordered), "tasks": {}}
    for task, labels in tasks.items():
        task_path = task_dir / f"{task}.jsonl"
        with open(task_path, "w", encoding="utf-8") as f:
            for key, label in sorted(labels.items(), key=lambda x: index[x[0]]):
                row = rows[key]
                item = {
                    "feature_index": index[key],
                    "label": label,
                    "contract_creator": row.get("creator") or row.get("group") or row.get("address"),
                    "address": row.get("address", ""),
                    "chain": row.get("chain", ""),
                }
                f.write(json.dumps(item) + "\n")
        summary["tasks"][task] = {
            "rows": len(labels),
            "positive": sum(labels.values()),
            "negative": len(labels) - sum(labels.values()),
            "path": str(task_path),
        }
    with open(out / "sliding_union_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
