#!/usr/bin/env python3
"""SmartDNA neuron-specificity ablation on the fixed seed-0 binary split.

The experiment compares the concatenated stable MLP units from critical
layers against layer/count-matched random units. CodeGemma stays frozen.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "base/CodeGemma-7B"
UNION = ROOT / "runs/stage3_smartintent_sliding/sliding_union.jsonl"
BINARY = ROOT / "runs/stage3_smartintent_sliding/sliding_tasks/binary.jsonl"
MANIFEST = ROOT / "runs/stage4_smartintent_full_trials/manifests/trial_seed_0.json"
STABLE = ROOT / "runs/stage10_codegemma_seed0_stable_units"
OUT = ROOT / "runs/stage13_codegemma_seed0_neuron_specificity"
LAYERS = (16, 18, 19, 21)
CS = (0.001, 0.01, 0.1, 1.0, 10.0)


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def load_key_units():
    result = {}
    for layer in LAYERS:
        path = STABLE / f"layer_{layer}" / "stability_units.csv"
        with path.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        units = [int(row["unit"]) for row in rows if row["stable"].lower() == "true"]
        if not units:
            raise RuntimeError(f"No stable units found for layer {layer}")
        result[layer] = np.asarray(sorted(units), dtype=np.int64)
    return result


def build_selections(key_units, repeats: int, width: int, seed: int):
    selections = {"key": {str(layer): units.tolist() for layer, units in key_units.items()},
                  "random_in": [], "random_ex": []}
    all_units = np.arange(width, dtype=np.int64)
    for repeat in range(repeats):
        in_rec, ex_rec = {}, {}
        for layer in LAYERS:
            count = len(key_units[layer])
            rng_in = np.random.default_rng(seed + 10000 + repeat * 101 + layer)
            rng_ex = np.random.default_rng(seed + 20000 + repeat * 101 + layer)
            in_rec[str(layer)] = sorted(rng_in.choice(all_units, count, replace=False).tolist())
            complement = np.setdiff1d(all_units, key_units[layer], assume_unique=True)
            ex_rec[str(layer)] = sorted(rng_ex.choice(complement, count, replace=False).tolist())
        selections["random_in"].append(in_rec)
        selections["random_ex"].append(ex_rec)
    return selections


def selection_audit(selections, key_units):
    audit = {"per_layer_key_counts": {str(k): int(len(v)) for k, v in key_units.items()},
             "total_key_count": int(sum(len(v) for v in key_units.values())),
             "random_in_key_overlap": [], "random_ex_key_overlap": []}
    for group in ("random_in", "random_ex"):
        for rec in selections[group]:
            overlap = 0
            for layer in LAYERS:
                chosen = set(rec[str(layer)])
                overlap += len(chosen & set(key_units[layer].tolist()))
                if len(chosen) != len(key_units[layer]):
                    raise RuntimeError("Layer-matched random count failed")
            audit[f"{group}_key_overlap"].append(overlap)
    if any(audit["random_ex_key_overlap"]):
        raise RuntimeError("Random-Ex contains a key unit")
    return audit


def make_windows(tokenizer, text: str, max_length: int, overlap: int):
    ids = tokenizer.encode(text, add_special_tokens=False, verbose=False)
    if not ids:
        ids = [tokenizer.unk_token_id or tokenizer.eos_token_id]
    prefix = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    suffix = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    budget = max_length - len(prefix) - len(suffix)
    step = budget - overlap
    chunks = []
    for start in range(0, len(ids), step):
        content = ids[start:start + budget]
        chunks.append({"ids": prefix + content + suffix,
                       "begin": len(prefix) + (0 if start == 0 else min(overlap, len(content))),
                       "end": len(prefix) + len(content)})
        if start + budget >= len(ids):
            break
    return len(ids), chunks


def union_columns(selections):
    columns = {}
    for layer in LAYERS:
        values = set(selections["key"][str(layer)])
        for group in ("random_in", "random_ex"):
            for rec in selections[group]:
                values.update(rec[str(layer)])
        columns[layer] = np.asarray(sorted(values), dtype=np.int64)
    return columns


def extract_features(args, candidates, selections):
    feature_dir = OUT / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    columns = union_columns(selections)
    expected = {layer: (len(candidates), len(columns[layer])) for layer in LAYERS}
    checkpoint = feature_dir / "checkpoint.json"
    stores = {}
    completed = 0
    if args.resume and checkpoint.exists():
        completed = int(json.load(checkpoint.open())["completed_rows"])
    for layer in LAYERS:
        path = feature_dir / f"layer_{layer}_selected_union.f16.npy"
        if args.resume and path.exists():
            store = np.lib.format.open_memmap(path, mode="r+")
            if store.shape != expected[layer]:
                raise RuntimeError(f"Unexpected feature shape: {path} {store.shape}")
        else:
            store = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16,
                                              shape=expected[layer])
        stores[layer] = store
        np.save(feature_dir / f"layer_{layer}_global_units.npy", columns[layer])

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    model = AutoModel.from_pretrained(args.model_path, local_files_only=True,
                                      dtype=model_dtype, low_cpu_mem_usage=True)
    model.requires_grad_(False)
    model.eval()
    model.config.use_cache = False
    if any(parameter.requires_grad for parameter in model.parameters()) or model.training:
        raise RuntimeError("Frozen-backbone audit failed")
    model.to(args.device)
    indices = {layer: torch.as_tensor(columns[layer], device=args.device) for layer in LAYERS}
    cache, hooks, active_spans = {}, [], {"value": None}
    for layer in LAYERS:
        def hook_factory(layer_id):
            def hook(_module, inputs):
                # Sum token activations before selecting columns. This is
                # exactly equivalent to selecting then summing, but avoids
                # moving a large token-by-unit tensor through the hook.
                spans = active_spans["value"]
                summed = torch.stack([inputs[0][batch_index, begin:end].sum(0)
                                       for batch_index, (begin, end) in enumerate(spans)])
                cache[layer_id] = summed.index_select(-1, indices[layer_id]).detach()
            return hook
        hooks.append(model.layers[layer - 1].mlp.down_proj.register_forward_pre_hook(hook_factory(layer)))
    print(json.dumps({"phase": "extraction", "resume_from": completed,
                      "rows": len(candidates),
                      "union_columns": {str(k): len(v) for k, v in columns.items()},
                      "backbone_frozen": True, "dtype": args.dtype}), flush=True)
    try:
        with torch.inference_mode():
            for block_start in range(completed, len(candidates), args.checkpoint_every):
                block_end = min(block_start + args.checkpoint_every, len(candidates))
                block_size = block_end - block_start
                expected_tokens = np.zeros(block_size, dtype=np.int64)
                counts = np.zeros(block_size, dtype=np.int64)
                all_chunks = []
                for local_row, row_idx in enumerate(range(block_start, block_end)):
                    expected, chunks = make_windows(tokenizer, candidates[row_idx]["text"],
                                                    args.max_length, args.overlap)
                    expected_tokens[local_row] = expected
                    for chunk in chunks:
                        chunk["local_row"] = local_row
                        all_chunks.append(chunk)
                # Length sorting reduces padding while still batching windows
                # from different contracts together.
                all_chunks.sort(key=lambda chunk: len(chunk["ids"]), reverse=True)
                sums = {layer: np.zeros((block_size, len(columns[layer])), dtype=np.float64)
                        for layer in LAYERS}
                for chunk_start in range(0, len(all_chunks), args.chunk_batch_size):
                    batch = all_chunks[chunk_start:chunk_start + args.chunk_batch_size]
                    width = max(len(chunk["ids"]) for chunk in batch)
                    ids = torch.full((len(batch), width), tokenizer.pad_token_id,
                                     dtype=torch.long, device=args.device)
                    mask = torch.zeros_like(ids)
                    spans = []
                    for batch_index, chunk in enumerate(batch):
                        chunk_ids = torch.as_tensor(chunk["ids"], dtype=torch.long, device=args.device)
                        ids[batch_index, :len(chunk_ids)] = chunk_ids
                        mask[batch_index, :len(chunk_ids)] = 1
                        spans.append((chunk["begin"], chunk["end"]))
                    cache.clear()
                    active_spans["value"] = spans
                    model(input_ids=ids, attention_mask=mask)
                    for batch_index, chunk in enumerate(batch):
                        begin, end = chunk["begin"], chunk["end"]
                        if end <= begin:
                            continue
                        local_row = chunk["local_row"]
                        counts[local_row] += end - begin
                        for layer in LAYERS:
                            sums[layer][local_row] += cache[layer][batch_index].float().cpu().numpy()
                if not np.array_equal(counts, expected_tokens):
                    raise RuntimeError(f"Token accounting failed in block {block_start}:{block_end}")
                for layer in LAYERS:
                    stores[layer][block_start:block_end] = (sums[layer] / counts[:, None]).astype(np.float16)
                    stores[layer].flush()
                atomic_json(checkpoint, {"completed_rows": block_end,
                                         "total_rows": len(candidates)})
                print(json.dumps({"phase": "extraction", "processed": block_end,
                                  "total": len(candidates)}), flush=True)
    finally:
        for hook in hooks:
            hook.remove()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return columns


def load_split(binary_rows, manifest):
    address_to_pos = {str(row["address"]).lower(): i for i, row in enumerate(binary_rows)}
    split, groups = {x: [] for x in ("train", "val", "test")}, {x: set() for x in ("train", "val", "test")}
    for rec in manifest["records"]:
        name = rec["split"]
        pos = address_to_pos[str(rec["address"]).lower()]
        if int(binary_rows[pos]["label"]) != int(rec["label"]):
            raise RuntimeError("Manifest label mismatch")
        split[name].append(pos)
        groups[name].add(str(rec["group"]).lower())
    overlap = {"train_val": sorted(groups["train"] & groups["val"]),
               "train_test": sorted(groups["train"] & groups["test"]),
               "val_test": sorted(groups["val"] & groups["test"])}
    if any(overlap.values()):
        raise RuntimeError(f"Group leakage: {overlap}")
    return {k: np.asarray(v, dtype=np.int64) for k, v in split.items()}, overlap


def metric_bundle(y, probability, threshold):
    prediction = probability >= threshold
    return {"accuracy": float(accuracy_score(y, prediction)),
            "auroc": float(roc_auc_score(y, probability)),
            "auprc": float(average_precision_score(y, probability)),
            "f1": float(f1_score(y, prediction))}


def best_threshold(y, probability):
    candidates = np.unique(np.r_[0.5, np.linspace(0.05, 0.95, 181), probability])
    scores = np.asarray([accuracy_score(y, probability >= threshold) for threshold in candidates])
    best = candidates[scores == scores.max()]
    return float(best[np.argmin(np.abs(best - 0.5))])


def build_matrix(stores, columns, selection, indices):
    parts = []
    for layer in LAYERS:
        inverse = {int(unit): pos for pos, unit in enumerate(columns[layer])}
        local = np.asarray([inverse[int(unit)] for unit in selection[str(layer)]], dtype=np.int64)
        parts.append(np.asarray(stores[layer][indices][:, local], dtype=np.float32))
    return np.concatenate(parts, axis=1)


def fit_probe(x_train, y_train, x_val, y_val, x_test, y_test, seed):
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)
    candidates = []
    for c in CS:
        clf = LogisticRegression(C=c, max_iter=3000, class_weight="balanced",
                                 solver="liblinear", random_state=seed)
        clf.fit(x_train, y_train)
        val_probability = clf.predict_proba(x_val)[:, 1]
        candidates.append((roc_auc_score(y_val, val_probability), c, clf, val_probability))
    val_auc, chosen_c, clf, val_probability = max(candidates, key=lambda row: (row[0], -row[1]))
    threshold = best_threshold(y_val, val_probability)
    test_probability = clf.predict_proba(x_test)[:, 1]
    return {"chosen_C": float(chosen_c), "chosen_threshold": threshold,
            "validation": metric_bundle(y_val, val_probability, threshold),
            "test": metric_bundle(y_test, test_probability, threshold)}


def summarize_repeats(rows):
    result = {"repeats": len(rows)}
    for split in ("validation", "test"):
        result[split] = {}
        for metric in ("accuracy", "auroc", "auprc", "f1"):
            values = np.asarray([row[split][metric] for row in rows], dtype=np.float64)
            result[split][metric] = {"mean": float(values.mean()),
                                     "std": float(values.std(ddof=1)),
                                     "ci95_lower": float(np.percentile(values, 2.5)),
                                     "ci95_upper": float(np.percentile(values, 97.5))}
    return result


def plot_results(key, random_in, random_ex):
    figure_dir = OUT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    groups = ["SmartDNA Key", "Random-In", "Random-Ex"]
    colors = ["#2878B5", "#F39B32", "#7A7A7A"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    for ax, metric, title in zip(axes, ("accuracy", "auroc"), ("(a) Accuracy", "(b) AUROC")):
        values = [100 * key["test"][metric],
                  100 * random_in["test"][metric]["mean"],
                  100 * random_ex["test"][metric]["mean"]]
        errors = [0.0, 100 * random_in["test"][metric]["std"],
                  100 * random_ex["test"][metric]["std"]]
        bars = ax.bar(groups, values, color=colors, width=0.68, yerr=errors, capsize=4,
                      edgecolor="white", linewidth=0.8)
        lower = max(0, min(values) - max(errors) - 8)
        ax.set_ylim(lower, 100)
        ax.set_ylabel("Score (%)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.22)
        ax.bar_label(bars, labels=[f"{value:.1f}" for value in values], padding=4, fontsize=9)
    fig.suptitle("CodeGemma Neuron Specificity (Seed 0)", fontsize=13)
    fig.tight_layout()
    fig.savefig(figure_dir / "figure7_neuron_specificity_accuracy_auroc.png", dpi=220,
                bbox_inches="tight")
    fig.savefig(figure_dir / "figure7_neuron_specificity_accuracy_auroc.pdf",
                bbox_inches="tight")
    plt.close(fig)
    # Also export paper-ready single-panel versions.
    for metric, filename, title in (("accuracy", "figure7_neuron_specificity_accuracy", "Neuron Specificity (Accuracy)"),
                                    ("auroc", "figure7_neuron_specificity_auroc", "Neuron Specificity (AUROC)")):
        values = [100 * key["test"][metric],
                  100 * (random_in["test"][metric]["mean"]),
                  100 * (random_ex["test"][metric]["mean"])]
        errors = [0.0, 100 * random_in["test"][metric]["std"],
                  100 * random_ex["test"][metric]["std"]]
        error_bar_options = {} if metric == "auroc" else {"yerr": errors, "capsize": 4}
        fig, ax = plt.subplots(figsize=(5.8, 4.8))
        bars = ax.bar(groups, values, color=colors, **error_bar_options,
                      edgecolor="white", linewidth=0.8)
        displayed_errors = [0.0] * len(values) if metric == "auroc" else errors
        ax.set_ylim(max(0, min(values) - max(displayed_errors) - 8), 100)
        ax.set_ylabel("Score (%)"); ax.set_title(title); ax.grid(axis="y", alpha=0.22)
        ax.bar_label(bars, labels=[f"{v:.1f}" for v in values], padding=4, fontsize=9)
        fig.tight_layout()
        fig.savefig(figure_dir / f"{filename}.png", dpi=220, bbox_inches="tight")
        fig.savefig(figure_dir / f"{filename}.pdf", bbox_inches="tight")
        plt.close(fig)


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=str(MODEL))
    parser.add_argument("--output-dir", default=str(OUT))
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--overlap", type=int, default=512)
    parser.add_argument("--random-repeats", type=int, default=30)
    parser.add_argument("--chunk-batch-size", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    OUT = Path(args.output_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    union_rows = load_jsonl(UNION)
    binary_rows = load_jsonl(BINARY)
    candidates = [union_rows[int(row["feature_index"])] for row in binary_rows]
    manifest = json.load(MANIFEST.open(encoding="utf-8"))
    splits, group_overlap = load_split(binary_rows, manifest)
    labels = np.asarray([int(row["label"]) for row in binary_rows], dtype=np.int64)
    key_units = load_key_units()
    selections = build_selections(key_units, args.random_repeats, 24576, args.seed)
    audit = selection_audit(selections, key_units)
    atomic_json(OUT / "unit_selections.json", selections)
    atomic_json(OUT / "audit.json", {**audit, "split_counts": {k: len(v) for k, v in splits.items()},
                                      "split_label_counts": {k: {"0": int((labels[v] == 0).sum()),
                                                                  "1": int((labels[v] == 1).sum())}
                                                             for k, v in splits.items()},
                                      "group_overlap": group_overlap, "test_used_for_selection": False})
    atomic_json(OUT / "config.json", {"seed": args.seed, "model": str(args.model_path),
                                      "backbone_frozen": True, "layers": list(LAYERS),
                                      "inference_dtype": args.dtype,
                                      "unit_type": "MLP down_proj input", "pooling": "non-overlap token mean",
                                      "max_length": args.max_length, "window_overlap": args.overlap,
                                      "random_repeats": args.random_repeats, "C_grid": list(CS),
                                      "validation_selects": ["C by AUROC", "threshold by Accuracy"],
                                      "test_used_for_selection": False})
    columns = extract_features(args, candidates, selections)
    stores = {layer: np.load(OUT / "features" / f"layer_{layer}_selected_union.f16.npy",
                             mmap_mode="r") for layer in LAYERS}

    def evaluate(selection, seed):
        matrices = {name: build_matrix(stores, columns, selection, splits[name])
                    for name in ("train", "val", "test")}
        return fit_probe(matrices["train"], labels[splits["train"]],
                         matrices["val"], labels[splits["val"]],
                         matrices["test"], labels[splits["test"]], seed)

    key_result = evaluate(selections["key"], args.seed)
    random_results = {"random_in": [], "random_ex": []}
    for group in random_results:
        for repeat, selection in enumerate(selections[group]):
            row = evaluate(selection, args.seed + repeat + (1000 if group == "random_in" else 2000))
            row["repeat"] = repeat
            row["key_overlap"] = audit[f"{group}_key_overlap"][repeat]
            random_results[group].append(row)
            print(json.dumps({"phase": "probe", "group": group, "repeat": repeat + 1,
                              "test_accuracy": row["test"]["accuracy"],
                              "test_auroc": row["test"]["auroc"]}), flush=True)
            atomic_json(OUT / "probe_checkpoint.json", {"key": key_result,
                                                         "random_results": random_results})
    in_summary = summarize_repeats(random_results["random_in"])
    ex_summary = summarize_repeats(random_results["random_ex"])
    summary = {"key": key_result, "random_in": in_summary, "random_ex": ex_summary,
               "interpretation_rule": "Key should exceed layer/count-matched random controls",
               "test_used_for_selection": False}
    atomic_json(OUT / "random_repeat_metrics.json", random_results)
    atomic_json(OUT / "summary.json", summary)
    with (OUT / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["group", "repeats", "accuracy_mean",
                                                    "accuracy_std", "auroc_mean", "auroc_std"])
        writer.writeheader()
        writer.writerow({"group": "SmartDNA Key", "repeats": 1,
                         "accuracy_mean": key_result["test"]["accuracy"], "accuracy_std": 0,
                         "auroc_mean": key_result["test"]["auroc"], "auroc_std": 0})
        for name, value in (("Random-In", in_summary), ("Random-Ex", ex_summary)):
            writer.writerow({"group": name, "repeats": value["repeats"],
                             "accuracy_mean": value["test"]["accuracy"]["mean"],
                             "accuracy_std": value["test"]["accuracy"]["std"],
                             "auroc_mean": value["test"]["auroc"]["mean"],
                             "auroc_std": value["test"]["auroc"]["std"]})
    plot_results(key_result, in_summary, ex_summary)
    print(json.dumps({"status": "complete", "output": str(OUT), "summary": summary},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
