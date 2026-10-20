#!/usr/bin/env python3
"""Stream SmartIntent's MariaDB dump into source-code probe JSONL files."""
import argparse
import gzip
import json
import random
import re
from pathlib import Path

INTENTS = [
    "Blacklist", "Mint", "Honeypot", "Fee", "DisableTrading",
    "MaxSell", "MaxTX", "Reflection", "Reward", "Rebase",
]
TYPE_MAP = {
    "blacklist": "Blacklist", "mint": "Mint", "honeypot": "Honeypot",
    "fee": "Fee", "disabletrading": "DisableTrading", "maxsell": "MaxSell",
    "maxtx": "MaxTX", "reflect": "Reflection", "reflection": "Reflection",
    "reward": "Reward", "rebase": "Rebase",
}


def tuples_from_insert(line):
    """Parse SQL tuple literals, honoring quoted strings and backslash escapes."""
    start = line.find("VALUES")
    s = line[start + 6:] if start >= 0 else line
    i = 0
    while i < len(s):
        while i < len(s) and s[i] not in "(":
            i += 1
        if i >= len(s):
            break
        i += 1
        fields, buf = [], []
        quoted = False
        while i < len(s):
            c = s[i]
            if quoted:
                if c == "\\" and i + 1 < len(s):
                    # MariaDB dump escaping: preserve the escaped character.
                    nxt = s[i + 1]
                    buf.append({"n": "\n", "r": "\r", "t": "\t"}.get(nxt, nxt))
                    i += 2
                    continue
                if c == "'":
                    if i + 1 < len(s) and s[i + 1] == "'":
                        buf.append("'"); i += 2; continue
                    quoted = False; i += 1; continue
                buf.append(c); i += 1; continue
            if c == "'":
                quoted = True; i += 1; continue
            if c == ",":
                fields.append("".join(buf)); buf = []; i += 1; continue
            if c == ")":
                fields.append("".join(buf)); i += 1
                yield fields
                break
            buf.append(c); i += 1


def parse_scalar(v):
    v = v.strip()
    if v.upper() == "NULL":
        return None
    return v


def read_token_labels(path):
    labels = {}
    in_tokens = False
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("INSERT INTO `tokens`"):
                in_tokens = True
            elif in_tokens and line.startswith("UNLOCK TABLES"):
                in_tokens = False
            if in_tokens and "(" in line:
                for row in tuples_from_insert(line):
                    if len(row) < 6:
                        continue
                    cid = parse_scalar(row[5])
                    scams = parse_scalar(row[4]) or "[]"
                    found = {x: 0 for x in INTENTS}
                    try:
                        entries = json.loads(scams)
                    except Exception:
                        entries = []
                    if isinstance(entries, list):
                        for item in entries:
                            if isinstance(item, dict):
                                intent = TYPE_MAP.get(str(item.get("type", "")).lower())
                                if intent:
                                    found[intent] = 1
                    labels[cid] = found
    return labels


def build(args):
    token_labels = read_token_labels(args.dump)
    rows = []
    wanted = set(token_labels)
    in_contracts = False
    with gzip.open(args.dump, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("INSERT INTO `contracts`"):
                if in_contracts and line.startswith("UNLOCK TABLES"):
                    in_contracts = False
                elif not in_contracts:
                    continue
            else:
                in_contracts = True
            if in_contracts and "(" in line:
              for row in tuples_from_insert(line):
                if len(row) < 18 or row[0] not in wanted:
                    continue
                source = parse_scalar(row[4]) or ""
                if len(source.strip()) < 40:
                    continue
                labels = token_labels[row[0]]
                rows.append({
                    "text": source,
                    "labels": labels,
                    "label": int(any(labels.values())),
                    "contract_id": row[0],
                    "address": parse_scalar(row[3]) or "",
                    "chain": parse_scalar(row[2]) or "",
                    "creator": parse_scalar(row[17]) or "",
                    "group": parse_scalar(row[17]) or (parse_scalar(row[3]) or row[0]),
                })
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    positives = [r for r in rows if r["label"] == 1]
    negatives = [r for r in rows if r["label"] == 0]
    n = min(args.binary_per_class, len(positives), len(negatives))
    binary = positives[:n] + negatives[:n]
    rng.shuffle(binary)
    with open(out / "binary_probe_balanced.jsonl", "w", encoding="utf-8") as f:
        for r in binary:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # Same source rows and fixed order for feature reuse across all intents.
    with open(out / "source_multilabel.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    counts = {"all_joined": len(rows), "positive": len(positives), "negative": len(negatives), "binary_each": n}
    for intent in INTENTS:
        pos = [r for r in rows if r["labels"].get(intent, 0) == 1]
        neg = [r for r in rows if r["labels"].get(intent, 0) == 0]
        cap = min(args.intent_per_class, len(pos), len(neg))
        subset = pos[:cap] + neg[:cap]; rng.shuffle(subset)
        with open(out / f"intent_{intent}.jsonl", "w", encoding="utf-8") as f:
            for r in subset:
                q = dict(r); q["label"] = int(q["labels"].get(intent, 0)); f.write(json.dumps(q, ensure_ascii=False) + "\n")
        counts[intent] = {"positive": len(pos), "negative": len(neg), "balanced_each": cap}
    with open(out / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump({"source": "web3se/smart-contract-intent-vul-dataset/web3_all.sql.gz", "intents": INTENTS, "counts": counts}, f, indent=2)
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--binary-per-class", type=int, default=1000)
    ap.add_argument("--intent-per-class", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    build(ap.parse_args())
