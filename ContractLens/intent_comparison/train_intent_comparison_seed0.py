#!/usr/bin/env python3
"""Train, evaluate, fuse, and summarize all seed-0 intent baselines."""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from intent_comparison_common import (
    INTENTS, OUT, SPLITS, atomic_json, group_bootstrap_ci, load_jsonl,
    metric_bundle, require_disk, seed_all, select_late_fusion,
)


FEATURE_DIR = OUT / "features"
RESULT_DIR = OUT / "results"
NATIVE_METHODS = ["trapdoor_lgbm", "honeypot_xgboost", "scsguard", "opcode_xai"]
DISPLAY_NAMES = {
    "ordinary_probe": "CodeGemma ordinary probe",
    "smartintentnn": "CodeGemma + Adapted SmartIntentNN",
    "smartintentv2": "CodeGemma + Adapted SmartIntentV2",
    "trapdoor_lgbm": "CodeGemma + Adapted Trapdoor-LGBM",
    "honeypot_xgboost": "CodeGemma + Adapted Honeypot-XGBoost",
    "scsguard": "CodeGemma + Adapted SCSGuard",
    "opcode_xai": "CodeGemma + Adapted Opcode-XAI",
    "smartdna": "Ours: SmartDNA key-unit probe",
}
TRAPDOOR_SUPPORTED = {"Blacklist", "Honeypot", "Fee", "DisableTrading", "MaxSell", "MaxTX"}


def task_data(intent: str) -> dict:
    path = OUT / "manifests" / "tasks_complete_case" / f"{intent}.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for split in SPLITS:
        records = [row for row in manifest["records"] if row["split"] == split]
        result[split] = {
            "index": np.asarray([row["union_index"] for row in records], dtype=np.int64),
            "label": np.asarray([row["label"] for row in records], dtype=np.int64),
            "group": np.asarray([row["component"] for row in records], dtype=object),
        }
        if len(np.unique(result[split]["label"])) != 2:
            raise RuntimeError(f"{intent}/{split} lost one class after complete-case filtering")
    return result


def result_paths(method: str, intent: str):
    base = RESULT_DIR / method
    return base / f"{intent}.json", base / f"{intent}_probabilities.npz"


def save_result(method, intent, details, data, val_probability, test_probability):
    json_path, probability_path = result_paths(method, intent)
    validation = metric_bundle(data["val"]["label"], val_probability)
    test = metric_bundle(data["test"]["label"], test_probability)
    bootstrap = group_bootstrap_ci(
        data["test"]["label"], test_probability, data["test"]["group"], repeats=1000, seed=0,
    )
    result = {
        "method": method, "intent": intent, "seed": 0, "backbone_frozen": True,
        "validation": validation, "test": test, "test_group_bootstrap_95_ci": bootstrap,
        **details,
    }
    atomic_json(json_path, result)
    probability_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        probability_path,
        val_probability=np.asarray(val_probability, dtype=np.float32),
        test_probability=np.asarray(test_probability, dtype=np.float32),
    )
    return result


def load_finished(method, intent):
    json_path, probability_path = result_paths(method, intent)
    if not json_path.exists() or not probability_path.exists():
        return None
    result = json.loads(json_path.read_text(encoding="utf-8"))
    probability = np.load(probability_path)
    return result, probability["val_probability"], probability["test_probability"]


def fit_logistic(x_train, y_train):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced", random_state=0),
    )
    model.fit(x_train, y_train)
    return model


def run_ordinary(intent, data, means):
    if load_finished("ordinary_probe", intent):
        return
    candidates = []
    for layer in range(means.shape[1]):
        model = fit_logistic(np.asarray(means[data["train"]["index"], layer], dtype=np.float32), data["train"]["label"])
        val = model.predict_proba(np.asarray(means[data["val"]["index"], layer], dtype=np.float32))[:, 1]
        test = model.predict_proba(np.asarray(means[data["test"]["index"], layer], dtype=np.float32))[:, 1]
        candidates.append((roc_auc_score(data["val"]["label"], val), layer, val, test))
    _, layer, val, test = max(candidates, key=lambda item: (item[0], -item[1]))
    save_result("ordinary_probe", intent, {
        "selected_layer": layer, "selection_metric": "validation AUROC", "logistic_C": 1.0,
        "all_layer_validation_auroc": {str(row[1]): float(row[0]) for row in candidates},
    }, data, val, test)


def run_smartdna(intent, data, unit_features):
    if load_finished("smartdna", intent):
        return
    model = fit_logistic(np.asarray(unit_features[data["train"]["index"]], dtype=np.float32), data["train"]["label"])
    val = model.predict_proba(np.asarray(unit_features[data["val"]["index"]], dtype=np.float32))[:, 1]
    test = model.predict_proba(np.asarray(unit_features[data["test"]["index"]], dtype=np.float32))[:, 1]
    save_result("smartdna", intent, {
        "critical_layers": [16, 18, 19, 21], "stable_unit_count": int(unit_features.shape[1]),
        "units_rediscovered_per_intent": False, "logistic_C": 1.0,
    }, data, val, test)


def focal_loss(logit, target, gamma=2.0):
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logit, target, reduction="none")
    probability = torch.sigmoid(logit)
    pt = probability * target + (1 - probability) * (1 - target)
    return ((1 - pt).pow(gamma) * bce).mean()


class SmartIntentHead(torch.nn.Module):
    def __init__(self, projection=False, centers=None):
        super().__init__()
        recurrent_dim = 768 if projection else 3072
        self.projection = torch.nn.Linear(3072, recurrent_dim) if projection else torch.nn.Identity()
        self.lstm = torch.nn.LSTM(recurrent_dim, 128, batch_first=True, bidirectional=True)
        self.dropout = torch.nn.Dropout(0.5)
        self.output = torch.nn.Linear(256, 1)
        self.register_buffer("centers", torch.as_tensor(centers, dtype=torch.float32) if centers is not None else torch.empty(0))

    def forward(self, x, mask):
        x = x.float()
        if self.centers.numel():
            distance = 1 - torch.einsum(
                "bsd,kd->bsk", torch.nn.functional.normalize(x, dim=-1),
                torch.nn.functional.normalize(self.centers, dim=-1),
            ).max(-1).values
            x = torch.where((distance < 0.21).unsqueeze(-1), x * 2.0, x)
        x = self.projection(x) * mask.unsqueeze(-1).float()
        _, (hidden, _) = self.lstm(x)
        return self.output(self.dropout(torch.cat([hidden[-2], hidden[-1]], dim=-1))).squeeze(-1)


def sequence_predict(model, features, masks, indices, device):
    model.eval()
    output = []
    with torch.inference_mode():
        for start in range(0, len(indices), 32):
            batch = indices[start:start + 32]
            x = torch.from_numpy(np.asarray(features[batch], dtype=np.float32)).to(device)
            m = torch.from_numpy(np.asarray(masks[batch], dtype=np.bool_)).to(device)
            output.append(torch.sigmoid(model(x, m)).cpu().numpy())
    return np.concatenate(output)


def fit_kmeans(intent, features, masks, train_indices):
    path = FEATURE_DIR / "kmeans" / f"{intent}.npy"
    if path.exists():
        return np.load(path)
    valid = np.argwhere(np.asarray(masks[train_indices], dtype=np.bool_))
    rng = np.random.default_rng(0)
    selected = rng.choice(len(valid), size=min(50000, len(valid)), replace=False)
    pairs = valid[selected]
    rows = train_indices[pairs[:, 0]]
    x = np.asarray(features[rows, pairs[:, 1]], dtype=np.float32)
    model = MiniBatchKMeans(n_clusters=10, random_state=0, n_init=3, batch_size=2048, max_iter=100)
    model.fit(x)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, model.cluster_centers_.astype(np.float32))
    return model.cluster_centers_


def run_sequence(method, intent, data, features, masks):
    if load_finished(method, intent):
        return
    projection = method == "smartintentv2"
    centers = None if projection else fit_kmeans(intent, features, masks, data["train"]["index"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmartIntentHead(projection=projection, centers=centers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4 if projection else 1e-3)
    train_indices = data["train"]["index"]
    labels_by_union = {int(index): int(label) for index, label in zip(train_indices, data["train"]["label"])}
    best_state, best_auc, stale, history = None, -1.0, 0, []
    for epoch in range(20):
        model.train()
        order = np.random.default_rng(epoch).permutation(train_indices)
        for start in range(0, len(order), 16):
            batch = order[start:start + 16]
            x = torch.from_numpy(np.asarray(features[batch], dtype=np.float32)).to(device)
            m = torch.from_numpy(np.asarray(masks[batch], dtype=np.bool_)).to(device)
            y = torch.as_tensor([labels_by_union[int(index)] for index in batch], dtype=torch.float32, device=device)
            loss = focal_loss(model(x, m), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val = sequence_predict(model, features, masks, data["val"]["index"], device)
        auc = float(roc_auc_score(data["val"]["label"], val))
        history.append({"epoch": epoch + 1, "validation_auroc": auc, "loss": float(loss.detach().cpu())})
        print(json.dumps({"method": method, "intent": intent, **history[-1]}), flush=True)
        if auc > best_auc + 1e-5:
            best_auc, stale = auc, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 4:
                break
    model.load_state_dict(best_state)
    model.to(device)
    val = sequence_predict(model, features, masks, data["val"]["index"], device)
    test = sequence_predict(model, features, masks, data["test"]["index"], device)
    save_result(method, intent, {
        "last_layer_sequence": True, "sequence_length": 256, "bilstm_hidden": 128,
        "projection_dim": 768 if projection else None, "kmeans_k": None if projection else 10,
        "highlight_scale": None if projection else 2.0, "training_history": history,
    }, data, val, test)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def tune_lgbm(x_train, y_train, x_val, y_val):
    best = None
    for learning_rate in (0.1, 0.2, 0.5):
        for estimators in (50, 100, 500):
            model = lgb.LGBMClassifier(learning_rate=learning_rate, n_estimators=estimators,
                                       random_state=0, verbosity=-1, n_jobs=4)
            model.fit(x_train, y_train)
            probability = model.predict_proba(x_val)[:, 1]
            candidate = (roc_auc_score(y_val, probability), -estimators, learning_rate, model)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
    return best[3], {"learning_rate": best[2], "n_estimators": -best[1]}


def run_trapdoor(intent, data, opcodes):
    if intent not in TRAPDOOR_SUPPORTED or load_finished("trapdoor_lgbm", intent):
        return
    model, parameters = tune_lgbm(
        np.asarray(opcodes[data["train"]["index"]]), data["train"]["label"],
        np.asarray(opcodes[data["val"]["index"]]), data["val"]["label"],
    )
    val = model.predict_proba(np.asarray(opcodes[data["val"]["index"]]))[:, 1]
    test = model.predict_proba(np.asarray(opcodes[data["test"]["index"]]))[:, 1]
    save_result("trapdoor_lgbm", intent, {
        "native_branch": True, "official_artifact_classifier": "LightGBM",
        "feature": "runtime opcode frequency", "selected_parameters": parameters,
    }, data, val, test)


PRAGMA = re.compile(r"pragma\s+solidity\s+[^0-9]*([0-9]+)(?:\.([0-9]+))?(?:\.([0-9]+))?", re.I)


def source_features(rows, available):
    result = np.zeros((len(rows), 8), dtype=np.float32)
    for index, row in enumerate(rows):
        text = str(row["text"])
        match = PRAGMA.search(text)
        result[index] = [1, int(available[index]), -1, len(text.splitlines()),
                         int(match.group(1)) if match else -1,
                         int(match.group(2) or 0) if match else -1,
                         int(match.group(3) or 0) if match else -1, -1]
    return result


def run_honeypot(data, features):
    if load_finished("honeypot_xgboost", "Honeypot"):
        return
    y_train = data["train"]["label"]
    positives, negatives = y_train.sum(), (y_train == 0).sum()
    model = xgb.XGBClassifier(n_jobs=4, scale_pos_weight=float(negatives / max(positives, 1)),
                              n_estimators=25, max_depth=3, random_state=0,
                              eval_metric="logloss")
    model.fit(features[data["train"]["index"]], y_train)
    val = model.predict_proba(features[data["val"]["index"]])[:, 1]
    test = model.predict_proba(features[data["test"]["index"]])[:, 1]
    save_result("honeypot_xgboost", "Honeypot", {
        "native_branch": True, "official_repository_commit": "c814790a976fa5da0b5ee4745a8db4385f474336",
        "source_only_variant": True, "n_estimators": 25, "max_depth": 3,
        "unavailable_original_fields": ["compiler_runs", "library_id"],
    }, data, val, test)


def entropy(labels):
    if len(labels) == 0:
        return 0.0
    p = np.bincount(labels.astype(np.int64), minlength=2) / len(labels)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def supervised_thresholds(x, y):
    thresholds = np.zeros(x.shape[1], dtype=np.float32)
    parent = entropy(y)
    for feature in range(x.shape[1]):
        values = np.unique(x[:, feature])
        if len(values) <= 1:
            thresholds[feature] = values[0] if len(values) else 0
            continue
        candidates = (values[:-1] + values[1:]) / 2
        best_gain, best_threshold = -1.0, candidates[0]
        for threshold in candidates:
            left = x[:, feature] < threshold
            gain = parent - (left.mean() * entropy(y[left]) + (~left).mean() * entropy(y[~left]))
            if gain > best_gain:
                best_gain, best_threshold = gain, threshold
        thresholds[feature] = best_threshold
    return thresholds


def run_opcode_xai(intent, data, opcodes):
    if load_finished("opcode_xai", intent):
        return
    x_train = np.asarray(opcodes[data["train"]["index"]], dtype=np.float32)
    y_train = data["train"]["label"]
    counts = np.bincount(y_train, minlength=2)
    smote_applied = bool(min(counts) >= 6 and counts[0] != counts[1])
    if smote_applied:
        x_fit, y_fit = SMOTE(random_state=0, k_neighbors=5).fit_resample(x_train, y_train)
    else:
        x_fit, y_fit = x_train, y_train
    thresholds = supervised_thresholds(x_fit, y_fit)
    binary_fit = (x_fit >= thresholds).astype(np.float32)
    selector = SelectFromModel(ExtraTreesClassifier(n_estimators=200, random_state=0, n_jobs=4), threshold="median")
    selected_fit = selector.fit_transform(binary_fit, y_fit)
    model = LogisticRegression(max_iter=5000, class_weight="balanced", random_state=0)
    model.fit(selected_fit, y_fit)
    def predict(indices):
        binary = (np.asarray(opcodes[indices], dtype=np.float32) >= thresholds).astype(np.float32)
        return model.predict_proba(selector.transform(binary))[:, 1]
    val, test = predict(data["val"]["index"]), predict(data["test"]["index"])
    save_result("opcode_xai", intent, {
        "native_branch": True, "smote_train_only": True, "smote_applied": smote_applied,
        "smote_skip_reason": None if smote_applied else "training split already balanced",
        "supervised_binning_train_only": True,
        "extra_trees_train_only": True, "selected_feature_count": int(selected_fit.shape[1]),
        "classifier": "LogisticRegression",
    }, data, val, test)


def byte_ngram_sequence(bytecode, vocabulary, max_length=2048):
    raw = str(bytecode or "").removeprefix("0x")
    data = bytes.fromhex(raw) if raw else b""
    grams = [(data[i] << 8) | data[i + 1] for i in range(max(0, len(data) - 1))]
    sequence = np.zeros(max_length, dtype=np.int64)
    for i, gram in enumerate(grams[:max_length]):
        sequence[i] = vocabulary.get(gram, 0)
    return sequence


class SCSGuard(torch.nn.Module):
    def __init__(self, vocabulary_size):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocabulary_size, 32, padding_idx=0)
        self.gru = torch.nn.GRU(32, 128, batch_first=True)
        self.attention = torch.nn.Parameter(torch.empty(128))
        torch.nn.init.normal_(self.attention, std=0.02)
        self.output = torch.nn.Linear(128, 1)

    def forward(self, token):
        mask = token != 0
        hidden, _ = self.gru(self.embedding(token))
        score = torch.einsum("bsh,h->bs", hidden, self.attention) / math.sqrt(hidden.shape[-1])
        score = score.masked_fill(~mask, -1e4)
        weight = torch.softmax(score, dim=1)
        pooled = (hidden * weight.unsqueeze(-1)).sum(1)
        return self.output(pooled).squeeze(-1)


def scsguard_predict(model, sequence, indices, device):
    model.eval()
    output = []
    with torch.inference_mode():
        for start in range(0, len(indices), 32):
            token = torch.from_numpy(sequence[indices[start:start + 32]]).to(device)
            output.append(torch.sigmoid(model(token)).cpu().numpy())
    return np.concatenate(output)


def run_scsguard(intent, data, bytecodes):
    if load_finished("scsguard", intent):
        return
    train_grams = collections.Counter()
    for index in data["train"]["index"]:
        raw = str(bytecodes[int(index)] or "").removeprefix("0x")
        code = bytes.fromhex(raw) if raw else b""
        train_grams.update((code[i] << 8) | code[i + 1] for i in range(max(0, len(code) - 1)))
    vocabulary = {gram: rank + 1 for rank, (gram, _) in enumerate(train_grams.most_common(19599))}
    sequence = np.stack([byte_ngram_sequence(code, vocabulary) for code in bytecodes])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SCSGuard(len(vocabulary) + 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    labels = {int(i): int(y) for i, y in zip(data["train"]["index"], data["train"]["label"])}
    best_state, best_auc, stale, history = None, -1.0, 0, []
    for epoch in range(100):
        model.train()
        order = np.random.default_rng(epoch).permutation(data["train"]["index"])
        for start in range(0, len(order), 16):
            batch = order[start:start + 16]
            token = torch.from_numpy(sequence[batch]).to(device)
            target = torch.as_tensor([labels[int(i)] for i in batch], dtype=torch.float32, device=device)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(token), target)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        val = scsguard_predict(model, sequence, data["val"]["index"], device)
        auc = float(roc_auc_score(data["val"]["label"], val))
        history.append({"epoch": epoch + 1, "validation_auroc": auc, "loss": float(loss.detach().cpu())})
        if auc > best_auc + 1e-5:
            best_auc, stale = auc, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 10:
                break
    model.load_state_dict(best_state); model.to(device)
    val = scsguard_predict(model, sequence, data["val"]["index"], device)
    test = scsguard_predict(model, sequence, data["test"]["index"], device)
    save_result("scsguard", intent, {
        "native_branch": True, "independent_reimplementation": True, "byte_ngram": 2,
        "vocabulary_size": len(vocabulary) + 1, "embedding_dim": 32, "gru_hidden": 128,
        "scaled_dot_product_attention": True, "max_ngram_sequence": 2048,
        "parameter_count": sum(p.numel() for p in model.parameters()), "training_history": history,
    }, data, val, test)
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def fuse_native(intent, data):
    ordinary = load_finished("ordinary_probe", intent)
    for method in NATIVE_METHODS:
        native = load_finished(method, intent)
        if not native:
            continue
        _, code_val, code_test = ordinary
        native_result, native_val, native_test = native
        fusion = select_late_fusion(data["val"]["label"], code_val, native_val, code_test, native_test)
        save_result(method + "_adapted", intent, {
            "native_method": method, "late_fusion": True,
            "lambda_codegemma": fusion["lambda_codegemma"], "fusion_grid": fusion["grid"],
        }, data, fusion["validation_probability"], fusion["test_probability"])


def run_all():
    require_disk(5.0)
    rows = load_jsonl(OUT / "selected_union.jsonl")
    means = np.load(FEATURE_DIR / "all_layer_means.f16.npy", mmap_mode="r")
    segments = np.load(FEATURE_DIR / "last_layer_segments.f16.npy", mmap_mode="r")
    masks = np.load(FEATURE_DIR / "segment_mask.bool.npy", mmap_mode="r")
    units = np.load(FEATURE_DIR / "stable_mlp_units.f16.npy", mmap_mode="r")
    opcodes = np.load(OUT / "bytecode" / "opcode_counts.f32.npy", mmap_mode="r")
    available = np.load(OUT / "bytecode" / "available.bool.npy")
    bytecode_cache = json.loads((OUT / "bytecode" / "runtime_bytecode.json").read_text(encoding="utf-8"))
    bytecodes = [bytecode_cache[str(row["address"]).lower()]["bytecode"] for row in rows]
    src_features = source_features(rows, available)
    np.save(FEATURE_DIR / "honeypot_source_features.f32.npy", src_features)
    for intent in INTENTS:
        print(json.dumps({"stage": "training", "intent": intent}), flush=True)
        data = task_data(intent)
        run_ordinary(intent, data, means)
        run_smartdna(intent, data, units)
        run_sequence("smartintentnn", intent, data, segments, masks)
        run_sequence("smartintentv2", intent, data, segments, masks)
        run_trapdoor(intent, data, opcodes)
        if intent == "Honeypot": run_honeypot(data, src_features)
        run_opcode_xai(intent, data, opcodes)
        run_scsguard(intent, data, bytecodes)
        fuse_native(intent, data)
        require_disk(5.0)
    summarize()


def summarize():
    main_methods = ["ordinary_probe", "smartintentnn", "smartintentv2", "trapdoor_lgbm",
                    "honeypot_xgboost", "scsguard", "opcode_xai", "smartdna"]
    table, native_table, details = [], [], {}
    for method in main_methods:
        row = {"Method": DISPLAY_NAMES[method]}
        native_row = {"Method": method}
        details[method] = {}
        for intent in INTENTS:
            actual = method + "_adapted" if method in NATIVE_METHODS else method
            finished = load_finished(actual, intent)
            native = load_finished(method, intent) if method in NATIVE_METHODS else None
            if finished:
                result = finished[0]
                row[intent] = f"{100 * result['test']['auroc']:.2f} / {100 * result['test']['accuracy']:.2f}"
                details[method][intent] = result
            else:
                row[intent] = "N/A"
            if method in NATIVE_METHODS:
                native_row[intent] = (f"{100 * native[0]['test']['auroc']:.2f} / "
                                      f"{100 * native[0]['test']['accuracy']:.2f}") if native else "N/A"
        table.append(row)
        if method in NATIVE_METHODS: native_table.append(native_row)

    def write_table(path, rows):
        import csv
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["Method", *INTENTS])
            writer.writeheader(); writer.writerows(rows)
        markdown = ["| Method | " + " | ".join(INTENTS) + " |",
                    "|---|" + "---:|" * len(INTENTS)]
        markdown.extend("| " + " | ".join([str(row["Method"]), *[str(row[i]) for i in INTENTS]]) + " |" for row in rows)
        path.with_suffix(".md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    write_table(OUT / "metrics_main.csv", table)
    write_table(OUT / "metrics_native.csv", native_table)
    atomic_json(OUT / "metrics_detailed.json", details)

    expected_native = {
        "trapdoor_lgbm": TRAPDOOR_SUPPORTED,
        "honeypot_xgboost": {"Honeypot"},
        "scsguard": set(INTENTS),
        "opcode_xai": set(INTENTS),
    }
    audit_errors = []
    audit_cells = []
    for method in main_methods:
        supported = expected_native.get(method, set(INTENTS))
        for intent in INTENTS:
            should_exist = intent in supported
            actual = method + "_adapted" if method in NATIVE_METHODS else method
            finished = load_finished(actual, intent)
            data = task_data(intent)
            cell = {"method": method, "intent": intent, "expected": should_exist,
                    "present": bool(finished)}
            if should_exist != bool(finished):
                audit_errors.append({**cell, "error": "expected/present mismatch"})
            if finished:
                result, val_probability, test_probability = finished
                checks = {
                    "validation_n_matches": len(val_probability) == len(data["val"]["label"]),
                    "test_n_matches": len(test_probability) == len(data["test"]["label"]),
                    "validation_finite": bool(np.isfinite(val_probability).all()),
                    "test_finite": bool(np.isfinite(test_probability).all()),
                    "reported_test_n_matches": result["test"]["n"] == len(data["test"]["label"]),
                }
                if method in NATIVE_METHODS:
                    checks["main_table_uses_codegemma"] = result.get("lambda_codegemma") in {0.25, 0.5, 0.75, 1.0}
                cell["checks"] = checks
                if not all(checks.values()):
                    audit_errors.append({**cell, "error": "cell integrity check failed"})
            audit_cells.append(cell)
    config_path = OUT / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["late_fusion_lambda_codegemma"] = [0.25, 0.5, 0.75, 1.0]
    config["native_only_lambda_zero_reported_separately"] = True
    atomic_json(config_path, config)
    atomic_json(OUT / "final_output_audit.json", {
        "passed": not audit_errors, "errors": audit_errors, "cells": audit_cells,
        "test_used_for_selection": False, "accuracy_threshold": 0.5,
    })
    if audit_errors:
        raise RuntimeError(f"Final output audit failed with {len(audit_errors)} error(s)")

    figures = OUT / "figures"; figures.mkdir(parents=True, exist_ok=True)
    for metric in ("auroc", "accuracy"):
        matrix = np.full((len(main_methods), len(INTENTS)), np.nan)
        for i, method in enumerate(main_methods):
            for j, intent in enumerate(INTENTS):
                actual = method + "_adapted" if method in NATIVE_METHODS else method
                finished = load_finished(actual, intent)
                if finished: matrix[i, j] = 100 * finished[0]["test"][metric]
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.heatmap(matrix, annot=True, fmt=".1f", cmap="viridis", vmin=50, vmax=100,
                    xticklabels=INTENTS, yticklabels=[DISPLAY_NAMES[m] for m in main_methods], ax=ax)
        ax.set_title(f"CodeGemma seed0 intent comparison: Test {metric.upper()} (%)")
        ax.set_xlabel("Intent"); ax.set_ylabel("Method")
        fig.tight_layout(); fig.savefig(figures / f"test_{metric}_heatmap.png", dpi=180); plt.close(fig)
    atomic_json(OUT / "completion.json", {"status": "complete", "gptlens_run": False,
                                           "final_output_audit_passed": True,
                                           "main_methods": main_methods, "intents": INTENTS})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["train", "summarize"], default="train")
    args = parser.parse_args()
    seed_all(0)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    if args.stage == "train": run_all()
    else: summarize()


if __name__ == "__main__":
    main()
