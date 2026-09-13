#!/usr/bin/env python3
"""Shared utilities for the seed-0 ten-intent comparison."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/intent_comparison_seed0"
DATA = ROOT / "runs/stage3_smartintent/source_multilabel.jsonl"
OLD_MANIFEST = ROOT / "runs/stage4_smartintent_full_trials/manifests/trial_seed_0.json"
MODEL = ROOT / "base/CodeGemma-7B"
STABLE_UNITS = ROOT / "runs/stage10_codegemma_seed0_stable_units"

INTENTS = [
    "Blacklist", "Mint", "Honeypot", "Fee", "DisableTrading",
    "MaxSell", "MaxTX", "Reflection", "Reward", "Rebase",
]
CRITICAL_LAYERS = [16, 18, 19, 21]
SPLITS = ("train", "val", "test")
SPLIT_RATIOS = {"train": 0.64, "val": 0.16, "test": 0.20}


def seed_all(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False)
    tmp.replace(path)


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def stable_key(*parts: object) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()


def normalized_source_hash(text: str) -> str:
    normalized = "".join(str(text).lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def free_gib(path: Path = OUT) -> float:
    target = path if path.exists() else path.parent
    return shutil.disk_usage(target).free / (1024 ** 3)


def require_disk(minimum_gib: float = 5.0) -> None:
    available = free_gib()
    if available < minimum_gib:
        raise RuntimeError(f"Disk guard: {available:.2f} GiB free, require {minimum_gib:.2f} GiB")


def load_stable_units() -> dict[int, list[int]]:
    result = {}
    for layer in CRITICAL_LAYERS:
        path = STABLE_UNITS / f"layer_{layer}" / "stability_units.csv"
        units = []
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if str(row.get("stable", "")).lower() == "true":
                    units.append(int(row["unit"]))
        if not units:
            raise RuntimeError(f"No stable units found for Layer {layer}: {path}")
        result[layer] = units
    return result


def metric_bundle(y, probability) -> dict:
    y = np.asarray(y, dtype=np.int64)
    probability = np.asarray(probability, dtype=np.float64)
    prediction = (probability >= 0.5).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)),
        "positive": int(y.sum()),
        "negative": int((y == 0).sum()),
        "accuracy": float(accuracy_score(y, prediction)),
        "auroc": float(roc_auc_score(y, probability)),
        "auprc": float(average_precision_score(y, probability)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def group_bootstrap_ci(y, probability, groups, repeats: int = 1000, seed: int = 0) -> dict:
    y = np.asarray(y, dtype=np.int64)
    probability = np.asarray(probability, dtype=np.float64)
    groups = np.asarray(groups, dtype=object)
    unique = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in unique}
    rng = np.random.default_rng(seed)
    aucs, accuracies = [], []
    for _ in range(repeats):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([members[g] for g in sampled])
        if len(np.unique(y[indices])) != 2:
            continue
        aucs.append(roc_auc_score(y[indices], probability[indices]))
        accuracies.append(accuracy_score(y[indices], probability[indices] >= 0.5))
    def summarize(values):
        values = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(values.mean()),
            "lower_95": float(np.quantile(values, 0.025)),
            "upper_95": float(np.quantile(values, 0.975)),
            "std": float(values.std()),
            "valid_repeats": int(len(values)),
        }
    return {"unit": "creator_component", "requested_repeats": repeats,
            "auroc": summarize(aucs), "accuracy": summarize(accuracies)}


def clipped_logit(probability) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(value) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(value, -40, 40)))


def select_late_fusion(y_val, code_val, native_val, code_test, native_test) -> dict:
    candidates = []
    # The native-only (lambda=0) branch is reported separately.  Main-table
    # adapted rows must actually consume frozen CodeGemma scores.
    for weight in (0.25, 0.5, 0.75, 1.0):
        val = sigmoid(weight * clipped_logit(code_val) + (1 - weight) * clipped_logit(native_val))
        test = sigmoid(weight * clipped_logit(code_test) + (1 - weight) * clipped_logit(native_test))
        candidates.append({"lambda_codegemma": weight,
                           "validation_auroc": float(roc_auc_score(y_val, val)),
                           "validation_probability": val,
                           "test_probability": test})
    best = max(candidates, key=lambda row: (row["validation_auroc"], -abs(row["lambda_codegemma"] - 0.5)))
    return {"lambda_codegemma": best["lambda_codegemma"],
            "validation_auroc": best["validation_auroc"],
            "validation_probability": best["validation_probability"],
            "test_probability": best["test_probability"],
            "grid": [{k: v for k, v in row.items() if not k.endswith("probability")} for row in candidates]}
