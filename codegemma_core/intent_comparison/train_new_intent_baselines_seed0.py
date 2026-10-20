#!/usr/bin/env python3
"""Incremental seed-0 training for DBESN, SCMDetector, and Honeytoken.

This entrypoint deliberately excludes the four retired baselines.  It reuses
the frozen CodeGemma feature cache and the existing task manifests.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import roc_auc_score

from honeytoken_symbolic import analyze_bytecode
from intent_comparison_common import (
    INTENTS, OUT, SPLITS, atomic_json, atomic_jsonl, group_bootstrap_ci,
    load_jsonl, metric_bundle, require_disk, seed_all,
)


FEATURE_DIR = OUT / "features"
RESULT_DIR = OUT / "results"
CHECKPOINT_DIR = OUT / "checkpoints"
HONEYTOKEN_CACHE = OUT / "honeytoken_analysis.jsonl"
HONEYTOKEN_RULES = {
    "Blacklist": "transfer_restriction",
    "Mint": "infinite_minting",
    "Honeypot": "any_rule",
    "Fee": "transfer_quantity_inconsistency",
    "DisableTrading": "transfer_restriction",
}
MAIN_METHODS = [
    "ordinary_probe", "smartintentnn", "smartintentv2", "dbesn",
    "scmdetector", "honeytoken_detector", "smartdna",
]
DISPLAY_NAMES = {
    "ordinary_probe": "CodeGemma ordinary probe",
    "smartintentnn": "CodeGemma + Adapted SmartIntentNN",
    "smartintentv2": "CodeGemma + Adapted SmartIntentV2",
    "dbesn": "CodeGemma + Adapted DBESN",
    "scmdetector": "CodeGemma + Adapted SCMDetector",
    "honeytoken_detector": "Honeytoken-Detector (symbolic reimplementation)",
    "smartdna": "Ours: SmartDNA key-unit probe",
}
RULE_KEYS = [
    "transfer_quantity_inconsistency", "transfer_restriction", "fake_logs",
    "infinite_minting", "balance_manipulation", "proxy_token",
]


def task_data(intent: str) -> dict:
    manifest_path = OUT / "manifests" / "tasks_complete_case" / f"{intent}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = {}
    for split in SPLITS:
        records = [row for row in manifest["records"] if row["split"] == split]
        result[split] = {
            "index": np.asarray([row["union_index"] for row in records], dtype=np.int64),
            "label": np.asarray([row["label"] for row in records], dtype=np.int64),
            "group": np.asarray([row["component"] for row in records], dtype=object),
        }
        if len(np.unique(result[split]["label"])) != 2:
            raise RuntimeError(f"{intent}/{split} does not contain both classes")
    return result


def result_paths(method: str, intent: str) -> tuple[Path, Path]:
    directory = RESULT_DIR / method
    return directory / f"{intent}.json", directory / f"{intent}_probabilities.npz"


def load_finished(method: str, intent: str):
    json_path, probability_path = result_paths(method, intent)
    if not json_path.exists() or not probability_path.exists():
        return None
    result = json.loads(json_path.read_text(encoding="utf-8"))
    with np.load(probability_path) as probability:
        return result, probability["val_probability"], probability["test_probability"]


def save_result(method: str, intent: str, details: dict, data: dict,
                val_probability: np.ndarray, test_probability: np.ndarray) -> dict:
    val_probability = np.asarray(val_probability, dtype=np.float32)
    test_probability = np.asarray(test_probability, dtype=np.float32)
    result = {
        "method": method,
        "intent": intent,
        "seed": 0,
        "validation": metric_bundle(data["val"]["label"], val_probability),
        "test": metric_bundle(data["test"]["label"], test_probability),
        "test_group_bootstrap_95_ci": group_bootstrap_ci(
            data["test"]["label"], test_probability, data["test"]["group"],
            repeats=1000, seed=0,
        ),
        **details,
    }
    json_path, probability_path = result_paths(method, intent)
    atomic_json(json_path, result)
    probability_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = probability_path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp_path, val_probability=val_probability, test_probability=test_probability)
    tmp_path.replace(probability_path)
    return result


def reverse_valid_tokens(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    length = mask.sum(dim=1)
    position = torch.arange(x.shape[1], device=x.device).unsqueeze(0).expand(x.shape[0], -1)
    reverse = torch.where(position < length.unsqueeze(1), length.unsqueeze(1) - 1 - position, position)
    return x.gather(1, reverse.unsqueeze(-1).expand_as(x))


class GRUMB(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gru1 = torch.nn.GRU(1024, 32, batch_first=True)
        self.attention = torch.nn.MultiheadAttention(32, 4, batch_first=True)
        self.gru2 = torch.nn.GRU(32, 32, batch_first=True)
        self.norm = torch.nn.LayerNorm(32)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        first, _ = self.gru1(x)
        attended, _ = self.attention(first, first, first, key_padding_mask=~mask, need_weights=False)
        second, _ = self.gru2(attended)
        residual = self.norm(first + second)
        return torch.cat([attended, residual], dim=-1) * mask.unsqueeze(-1)


class AdaptedDBESN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(3072, 1024)
        self.forward_encoder = GRUMB()
        self.backward_encoder = GRUMB()
        self.final_gru = torch.nn.GRU(128, 16, batch_first=True)
        self.output = torch.nn.Linear(16, 1)

    def encode(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.projection(x) * mask.unsqueeze(-1)
        forward = self.forward_encoder(x, mask)
        reversed_x = reverse_valid_tokens(x, mask)
        backward = reverse_valid_tokens(self.backward_encoder(reversed_x, mask), mask)
        combined, _ = self.final_gru(torch.cat([forward, backward], dim=-1))
        return (combined * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encode(x, mask)
        return self.output(embedding).squeeze(-1), embedding


class MaskedAttentionPool(torch.nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.context = torch.nn.Parameter(torch.empty(dimension))
        torch.nn.init.normal_(self.context, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        score = torch.einsum("bsd,d->bs", torch.tanh(x), self.context)
        score = score.masked_fill(~mask, -1e4)
        weight = torch.softmax(score, dim=1) * mask.float()
        weight = weight / weight.sum(1, keepdim=True).clamp_min(1e-8)
        return (x * weight.unsqueeze(-1)).sum(1)


class AdaptedSCMDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(3072, 300)
        self.word_lstm = torch.nn.LSTM(300, 128, batch_first=True, bidirectional=True)
        self.word_attention = MaskedAttentionPool(256)
        self.line_lstm = torch.nn.LSTM(256, 128, batch_first=True, bidirectional=True)
        self.line_attention = MaskedAttentionPool(256)
        self.dropout = torch.nn.Dropout(0.3)
        self.output = torch.nn.Linear(256, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        projected = self.projection(x) * mask.unsqueeze(-1)
        words = projected.reshape(batch * 16, 16, 300)
        word_mask = mask.reshape(batch * 16, 16)
        encoded_words, _ = self.word_lstm(words)
        lines = self.word_attention(encoded_words, word_mask).reshape(batch, 16, 256)
        line_mask = word_mask.reshape(batch, 16, 16).any(dim=-1)
        encoded_lines, _ = self.line_lstm(lines * line_mask.unsqueeze(-1), None)
        contract = self.line_attention(encoded_lines, line_mask)
        return self.output(self.dropout(contract)).squeeze(-1)


def numpy_batch(features, masks, indices, device):
    x = torch.from_numpy(np.asarray(features[indices], dtype=np.float32)).to(device)
    mask = torch.from_numpy(np.asarray(masks[indices], dtype=np.bool_)).to(device)
    return x, mask


def predict_dbesn(model, features, masks, indices, device) -> np.ndarray:
    model.eval()
    result = []
    with torch.inference_mode():
        for start in range(0, len(indices), 32):
            x, mask = numpy_batch(features, masks, indices[start:start + 32], device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits, _ = model(x, mask)
            result.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(result)


def predict_scm(model, features, masks, indices, device) -> np.ndarray:
    model.eval()
    result = []
    with torch.inference_mode():
        for start in range(0, len(indices), 32):
            x, mask = numpy_batch(features, masks, indices[start:start + 32], device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(x, mask)
            result.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(result)


def balanced_pairs(labels: np.ndarray, epoch: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(epoch)
    anchors, partners, same = [], [], []
    positions = np.arange(len(labels))
    by_class = {value: positions[labels == value] for value in (0, 1)}
    for anchor in positions:
        candidates = by_class[int(labels[anchor])]
        candidates = candidates[candidates != anchor]
        if len(candidates):
            anchors.append(anchor); partners.append(int(rng.choice(candidates))); same.append(1.0)
        opposite = by_class[1 - int(labels[anchor])]
        anchors.append(anchor); partners.append(int(rng.choice(opposite))); same.append(0.0)
    order = rng.permutation(len(anchors))
    return np.asarray(anchors)[order], np.asarray(partners)[order], np.asarray(same, dtype=np.float32)[order]


def _gradient_scaler(device):
    return torch.amp.GradScaler("cuda", enabled=device.type == "cuda")


def train_dbesn(intent: str, data: dict, features, masks) -> None:
    if load_finished("dbesn", intent):
        return
    seed_all(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AdaptedDBESN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scaler = _gradient_scaler(device)
    train_index, train_label = data["train"]["index"], data["train"]["label"]
    best_auc, best_state, stale, history = -1.0, None, 0, []
    for epoch in range(20):
        model.train()
        anchors, partners, same = balanced_pairs(train_label, epoch)
        epoch_loss = []
        for start in range(0, len(anchors), 16):
            a_pos, p_pos = anchors[start:start + 16], partners[start:start + 16]
            a_idx, p_idx = train_index[a_pos], train_index[p_pos]
            ax, am = numpy_batch(features, masks, a_idx, device)
            px, pm = numpy_batch(features, masks, p_idx, device)
            ay = torch.as_tensor(train_label[a_pos], dtype=torch.float32, device=device)
            py = torch.as_tensor(train_label[p_pos], dtype=torch.float32, device=device)
            sy = torch.as_tensor(same[start:start + 16], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                alogit, aemb = model(ax, am)
                plogit, pemb = model(px, pm)
                bce = 0.5 * (torch.nn.functional.binary_cross_entropy_with_logits(alogit, ay) +
                             torch.nn.functional.binary_cross_entropy_with_logits(plogit, py))
                distance = torch.nn.functional.pairwise_distance(aemb.float(), pemb.float())
                contrastive = (sy * distance.square() + (1 - sy) * torch.relu(1 - distance).square()).mean()
                loss = bce + contrastive
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            epoch_loss.append(float(loss.detach().cpu()))
        val = predict_dbesn(model, features, masks, data["val"]["index"], device)
        auc = float(roc_auc_score(data["val"]["label"], val))
        row = {"epoch": epoch + 1, "train_loss": float(np.mean(epoch_loss)), "validation_auroc": auc}
        history.append(row); print(json.dumps({"method": "dbesn", "intent": intent, **row}), flush=True)
        if auc > best_auc + 1e-5:
            best_auc, stale = auc, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 4: break
    if best_state is None: raise RuntimeError(f"DBESN did not produce a checkpoint for {intent}")
    model.load_state_dict(best_state); model.to(device)
    val = predict_dbesn(model, features, masks, data["val"]["index"], device)
    test = predict_dbesn(model, features, masks, data["test"]["index"], device)
    checkpoint = CHECKPOINT_DIR / "dbesn" / f"{intent}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True); torch.save(best_state, checkpoint)
    save_result("dbesn", intent, {
        "backbone_frozen": True, "input_cache": "last_layer_segments.f16.npy",
        "sequence_length": 256, "input_dimension": 3072, "projection_dimension": 1024,
        "gru_hidden": [32, 32, 16], "attention_heads": 4, "contrastive_margin": 1.0,
        "pair_policy": "one same-class and one different-class partner per anchor per epoch",
        "optimizer": "Adam", "learning_rate": 1e-5, "batch_size_pairs": 16,
        "max_epochs": 20, "early_stopping_patience": 4,
        "selection_metric": "validation AUROC", "training_history": history,
        "checkpoint": str(checkpoint), "test_used_for_selection": False,
    }, data, val, test)
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def train_scmdetector(intent: str, data: dict, features, masks) -> None:
    if load_finished("scmdetector", intent):
        return
    seed_all(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AdaptedSCMDetector().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    batches_per_epoch = math.ceil(len(data["train"]["index"]) / 32)
    total_steps, warmup_steps = 20 * batches_per_epoch, max(1, int(0.1 * 20 * batches_per_epoch))
    def schedule(step):
        if step < warmup_steps: return max(step, 1) / warmup_steps
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = _gradient_scaler(device)
    labels = {int(index): int(label) for index, label in zip(data["train"]["index"], data["train"]["label"])}
    best_auc, best_state, stale, history = -1.0, None, 0, []
    for epoch in range(20):
        model.train(); losses = []
        order = np.random.default_rng(epoch).permutation(data["train"]["index"])
        for start in range(0, len(order), 32):
            batch = order[start:start + 32]
            x, mask = numpy_batch(features, masks, batch, device)
            target = torch.as_tensor([labels[int(i)] for i in batch], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x, mask), target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); scheduler.step()
            losses.append(float(loss.detach().cpu()))
        val = predict_scm(model, features, masks, data["val"]["index"], device)
        auc = float(roc_auc_score(data["val"]["label"], val))
        row = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "validation_auroc": auc,
               "learning_rate": float(optimizer.param_groups[0]["lr"])}
        history.append(row); print(json.dumps({"method": "scmdetector", "intent": intent, **row}), flush=True)
        if auc > best_auc + 1e-5:
            best_auc, stale = auc, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 4: break
    if best_state is None: raise RuntimeError(f"SCMDetector did not produce a checkpoint for {intent}")
    model.load_state_dict(best_state); model.to(device)
    val = predict_scm(model, features, masks, data["val"]["index"], device)
    test = predict_scm(model, features, masks, data["test"]["index"], device)
    checkpoint = CHECKPOINT_DIR / "scmdetector" / f"{intent}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True); torch.save(best_state, checkpoint)
    save_result("scmdetector", intent, {
        "backbone_frozen": True, "input_cache": "last_layer_segments.f16.npy",
        "sequence_length": 256, "hierarchy": [16, 16], "input_projection": 300,
        "word_bilstm_hidden_per_direction": 128, "line_bilstm_hidden_per_direction": 128,
        "word_attention": True, "line_attention": True, "dropout": 0.3,
        "optimizer": "AdamW", "learning_rate": 1e-4, "weight_decay": 0.01,
        "batch_size": 32, "max_epochs": 20, "early_stopping_patience": 4,
        "gradient_clip": 1.0, "warmup_steps": warmup_steps,
        "warmup_adaptation": "10% of local total steps instead of paper's scale-dependent 2000 steps",
        "selection_metric": "validation AUROC", "training_history": history,
        "checkpoint": str(checkpoint), "test_used_for_selection": False,
    }, data, val, test)
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def _analyze_job(job):
    union_index, address, bytecode, timeout = job
    try:
        result = analyze_bytecode(bytecode, timeout_seconds=timeout)
        return {"union_index": union_index, "address": address, **result}
    except Exception as error:
        return {"union_index": union_index, "address": address, "status": "error",
                "elapsed_seconds": 0.0, "error": f"{type(error).__name__}: {error}",
                **{key: False for key in RULE_KEYS}}


def prepare_honeytoken() -> None:
    rows = load_jsonl(OUT / "selected_union.jsonl")
    bytecode_cache = json.loads((OUT / "bytecode" / "runtime_bytecode.json").read_text(encoding="utf-8"))
    finished = {}
    if HONEYTOKEN_CACHE.exists():
        for row in load_jsonl(HONEYTOKEN_CACHE): finished[int(row["union_index"])] = row
    jobs = []
    for index, row in enumerate(rows):
        if index in finished and finished[index].get("status") == "ok": continue
        address = str(row["address"]).lower()
        bytecode = bytecode_cache[address]["bytecode"]
        timeout = 60.0 if index in finished else 20.0
        jobs.append((index, address, bytecode, timeout))
    if jobs:
        with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool:
            futures = {pool.submit(_analyze_job, job): job[0] for job in jobs}
            for count, future in enumerate(as_completed(futures), 1):
                result = future.result(); finished[int(result["union_index"])] = result
                if count % 25 == 0 or count == len(jobs):
                    atomic_jsonl(HONEYTOKEN_CACHE, [finished[key] for key in sorted(finished)])
                    print(json.dumps({"stage": "honeytoken_prepare", "completed": len(finished),
                                      "total": len(rows)}), flush=True)
    atomic_jsonl(HONEYTOKEN_CACHE, [finished[key] for key in sorted(finished)])
    status = {}
    for row in finished.values(): status[row.get("status", "unknown")] = status.get(row.get("status", "unknown"), 0) + 1
    atomic_json(OUT / "honeytoken_analysis_audit.json", {
        "total": len(rows), "completed": len(finished), "status_counts": status,
        "initial_timeout_seconds": 20, "retry_timeout_seconds": 60,
        "workers": min(8, os.cpu_count() or 1), "full_traces_persisted": False,
    })


def train_honeytoken() -> None:
    if not HONEYTOKEN_CACHE.exists(): raise RuntimeError("Run honeytoken prepare before train")
    analysis = {int(row["union_index"]): row for row in load_jsonl(HONEYTOKEN_CACHE)}
    for intent, rule in HONEYTOKEN_RULES.items():
        if load_finished("honeytoken_detector", intent): continue
        data = task_data(intent)
        split_probability, coverage = {}, {}
        for split in SPLITS:
            records = [analysis.get(int(index), {}) for index in data[split]["index"]]
            ok = np.asarray([row.get("status") == "ok" for row in records], dtype=bool)
            coverage[split] = float(ok.mean())
            if rule == "any_rule":
                probability = np.asarray([float(any(bool(row.get(key, False)) for key in RULE_KEYS)) for row in records])
            else:
                probability = np.asarray([float(bool(row.get(rule, False))) for row in records])
            split_probability[split] = probability
        if min(coverage.values()) < 0.95:
            atomic_json(RESULT_DIR / "honeytoken_detector" / f"{intent}_coverage_failure.json", {
                "intent": intent, "rule": rule, "coverage": coverage, "required_coverage": 0.95,
            })
            continue
        save_result("honeytoken_detector", intent, {
            "backbone_frozen": None, "uses_codegemma": False, "native_symbolic_baseline": True,
            "paper_rule": rule, "discrete_score": True, "coverage": coverage,
            "failed_analyses_scored_as_no_finding": True, "minimum_coverage": 0.95,
            "engine": "bounded CFG + Z3 independent reimplementation",
            "honeybadger_commit": "ff30c9a1b75a3fdc0f81188234b75d8b22c0df2f",
            "test_used_for_selection": False,
        }, data, split_probability["val"], split_probability["test"])


def _write_table(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Method", *INTENTS])
        writer.writeheader(); writer.writerows(rows)
    markdown = ["| Method | " + " | ".join(INTENTS) + " |",
                "|---|" + "---:|" * len(INTENTS)]
    markdown.extend("| " + " | ".join([str(row["Method"]), *[str(row[x]) for x in INTENTS]]) + " |" for row in rows)
    markdown.extend(["", "Notes:", "- Each populated cell is Test AUROC (%) / Accuracy (%).",
                     "- Honeytoken-Detector is a native symbolic baseline and does not use CodeGemma.",
                     "- Honeytoken N/A means the paper defines no corresponding rule, or analysis coverage was below 95%."])
    path.with_suffix(".md").write_text("\n".join(markdown) + "\n", encoding="utf-8")


def summarize() -> None:
    table, details = [], {}
    for method in MAIN_METHODS:
        row = {"Method": DISPLAY_NAMES[method]}; details[method] = {}
        for intent in INTENTS:
            finished = load_finished(method, intent)
            if finished:
                result = finished[0]
                row[intent] = f"{100 * result['test']['auroc']:.2f} / {100 * result['test']['accuracy']:.2f}"
                details[method][intent] = result
            else:
                row[intent] = "N/A"
        table.append(row)
    _write_table(OUT / "metrics_main.csv", table)
    atomic_json(OUT / "metrics_detailed.json", details)

    figures = OUT / "figures"; figures.mkdir(parents=True, exist_ok=True)
    for metric in ("auroc", "accuracy"):
        matrix = np.full((len(MAIN_METHODS), len(INTENTS)), np.nan)
        for i, method in enumerate(MAIN_METHODS):
            for j, intent in enumerate(INTENTS):
                finished = load_finished(method, intent)
                if finished: matrix[i, j] = 100 * finished[0]["test"][metric]
        fig, ax = plt.subplots(figsize=(15, 6.5))
        sns.heatmap(matrix, annot=True, fmt=".1f", cmap="viridis", vmin=50, vmax=100,
                    xticklabels=INTENTS, yticklabels=[DISPLAY_NAMES[m] for m in MAIN_METHODS], ax=ax)
        ax.set_title(f"Seed0 intent comparison: Test {metric.upper()} (%)")
        ax.set_xlabel("Intent"); ax.set_ylabel("Method")
        fig.tight_layout(); fig.savefig(figures / f"test_{metric}_heatmap.png", dpi=180); plt.close(fig)


def audit() -> None:
    errors, cells = [], []
    expected = {method: set(INTENTS) for method in MAIN_METHODS}
    expected["honeytoken_detector"] = set(HONEYTOKEN_RULES)
    split_audit = json.loads((OUT / "manifests" / "global_split_audit.json").read_text(encoding="utf-8"))
    if any(split_audit["component_overlap"].values()): errors.append("creator/source component split overlap")
    for method in MAIN_METHODS:
        for intent in INTENTS:
            should_exist = intent in expected[method]
            finished = load_finished(method, intent)
            cell = {"method": method, "intent": intent, "expected": should_exist, "present": bool(finished)}
            if should_exist != bool(finished): errors.append(f"expected/present mismatch: {method}/{intent}")
            if finished:
                result, val_probability, test_probability = finished
                data = task_data(intent)
                checks = {
                    "validation_length": len(val_probability) == len(data["val"]["label"]),
                    "test_length": len(test_probability) == len(data["test"]["label"]),
                    "validation_finite": bool(np.isfinite(val_probability).all()),
                    "test_finite": bool(np.isfinite(test_probability).all()),
                    "probability_range": bool(((val_probability >= 0) & (val_probability <= 1)).all() and
                                              ((test_probability >= 0) & (test_probability <= 1)).all()),
                    "reported_test_n": result["test"]["n"] == len(data["test"]["label"]),
                    "test_not_selection_metric": result.get("test_used_for_selection", False) is False,
                }
                cell["checks"] = checks
                if not all(checks.values()): errors.append(f"integrity check failed: {method}/{intent}")
            cells.append(cell)
    honeytoken_repo = Path(__file__).resolve().parent / "external_baselines/malicious_contract_methods/honeytoken_detector/HoneyBadger"
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=honeytoken_repo, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False).stdout.strip()
    reproduction = {
        "passed": not errors, "errors": errors, "cells": cells, "seed": 0,
        "backbone_loaded_during_new_training": False,
        "codegemma_input": "precomputed frozen last-layer segment cache",
        "test_used_for_selection": False, "accuracy_threshold": 0.5,
        "dbesn_status": "paper-structure adapted reimplementation",
        "scmdetector_status": "paper-structure adapted reimplementation",
        "honeytoken_status": "paper-rule independent symbolic reimplementation",
        "honeybadger_expected_commit": "ff30c9a1b75a3fdc0f81188234b75d8b22c0df2f",
        "honeybadger_actual_commit": commit,
        "honeytoken_supported_intents": HONEYTOKEN_RULES,
    }
    atomic_json(OUT / "new_baselines_reproduction_audit.json", reproduction)
    atomic_json(OUT / "final_output_audit.json", reproduction)
    if errors: raise RuntimeError(f"Audit failed with {len(errors)} errors")
    config_path = OUT / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update({
        "main_methods": MAIN_METHODS, "retired_methods_removed": [
            "trapdoor_lgbm", "honeypot_xgboost", "scsguard", "opcode_xai"],
        "new_baseline_entrypoint": str(Path(__file__).resolve()),
        "honeytoken_is_non_codegemma_symbolic_exception": True,
    })
    atomic_json(config_path, config)
    atomic_json(OUT / "completion.json", {
        "status": "complete", "seed": 0, "main_methods": MAIN_METHODS,
        "intents": INTENTS, "final_output_audit_passed": True,
    })


def run_train(method: str) -> None:
    require_disk(5.0)
    features = np.load(FEATURE_DIR / "last_layer_segments.f16.npy", mmap_mode="r")
    masks = np.load(FEATURE_DIR / "segment_mask.bool.npy", mmap_mode="r")
    for intent in INTENTS:
        data = task_data(intent)
        if method in {"dbesn", "all"}: train_dbesn(intent, data, features, masks)
        if method in {"scmdetector", "all"}: train_scmdetector(intent, data, features, masks)
        require_disk(5.0)
    if method in {"honeytoken", "all"}: train_honeytoken()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["dbesn", "scmdetector", "honeytoken", "all"], default="all")
    parser.add_argument("--stage", choices=["prepare", "train", "summarize", "audit"], default="train")
    args = parser.parse_args(); seed_all(0)
    if args.stage == "prepare":
        if args.method not in {"honeytoken", "all"}: raise SystemExit("prepare is only required for honeytoken/all")
        prepare_honeytoken()
    elif args.stage == "train": run_train(args.method)
    elif args.stage == "summarize": summarize()
    else: audit()


if __name__ == "__main__":
    main()

