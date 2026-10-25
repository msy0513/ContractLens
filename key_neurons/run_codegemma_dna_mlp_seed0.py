#!/usr/bin/env python3
"""Strict DNA-style MLP-unit discovery for CodeGemma, fixed Seed 0.

The backbone is frozen.  Candidate layers are fixed before this script runs;
only the 2,000-contract binary task is used, with Train for discovery and
Validation/Test held out for functional evaluation.
"""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer


DEFAULT_LAYERS = (16, 18, 19, 21)


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp, path)


def load_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def make_windows(tokenizer, text, max_length, overlap):
    code_ids = tokenizer.encode(text, add_special_tokens=False, verbose=False)
    if not code_ids:
        code_ids = [tokenizer.unk_token_id or tokenizer.eos_token_id]
    prefix = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    suffix = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    budget = max_length - len(prefix) - len(suffix)
    step = budget - overlap
    if step <= 0:
        raise ValueError("overlap must be smaller than the content budget")
    chunks = []
    for start in range(0, len(code_ids), step):
        content = code_ids[start:start + budget]
        chunks.append({
            "ids": prefix + content + suffix,
            "content_offset": len(prefix),
            "content_length": len(content),
            "unique_offset": 0 if start == 0 else min(overlap, len(content)),
        })
        if start + budget >= len(code_ids):
            break
    return len(code_ids), chunks


def metrics(y, probability):
    prediction = probability >= 0.5
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "auroc": float(roc_auc_score(y, probability)),
        "auprc": float(average_precision_score(y, probability)),
        "f1": float(f1_score(y, prediction)),
    }


def bootstrap_auroc(y, probability, seed, repeats=500):
    rng = np.random.default_rng(seed)
    values = []
    n = len(y)
    for _ in range(repeats):
        idx = rng.integers(0, n, size=n)
        yy = y[idx]
        if np.all(yy == yy[0]):
            continue
        values.append(roc_auc_score(yy, probability[idx]))
    if not values:
        return {"lower": None, "upper": None, "repeats": 0}
    return {
        "lower": float(np.percentile(values, 2.5)),
        "upper": float(np.percentile(values, 97.5)),
        "repeats": len(values),
    }


def kneedle(scores):
    order = np.argsort(-scores, kind="stable")
    sorted_scores = np.asarray(scores[order], dtype=np.float64)
    n = len(sorted_scores)
    if n <= 2 or float(sorted_scores[0] - sorted_scores[-1]) <= 0:
        return {"order": order, "selected_count": max(1, min(n, 1)),
                "elbow_rank_zero_based": 0, "paper_signed_argmax_zero_based": 0,
                "fallback": True}
    x = np.linspace(0.0, 1.0, n)
    y = (sorted_scores - sorted_scores[-1]) / (sorted_scores[0] - sorted_scores[-1])
    chord = y[0] + (y[-1] - y[0]) * x
    signed = y - chord
    interior = np.arange(1, n - 1)
    idx = int(interior[np.argmax(signed[interior])])
    # Keep the paper's signed-distance rule, while excluding endpoints.
    return {
        "order": order,
        "selected_count": idx + 1,
        "elbow_rank_zero_based": idx,
        "paper_signed_argmax_zero_based": idx,
        "fallback": False,
        "max_signed_distance": float(signed[idx]),
    }


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_extraction(args, candidates, output):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(
        args.model_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.requires_grad_(False)
    model.eval()
    model.config.use_cache = False
    model.to(args.device)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable or model.training:
        raise RuntimeError("CodeGemma freeze check failed")
    layers = getattr(model, "layers", None)
    if layers is None:
        raise RuntimeError("Expected AutoModel CodeGemma to expose model.layers")
    hidden = int(model.config.intermediate_size)
    stores = {}
    for layer in args.layers:
        path = output / "mlp_features" / f"layer_{layer:02d}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        if args.resume and path.exists():
            store = np.lib.format.open_memmap(path, mode="r+")
            if store.shape != (len(candidates), hidden):
                raise RuntimeError(f"Unexpected store shape for {path}: {store.shape}")
        else:
            store = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16,
                                              shape=(len(candidates), hidden))
        stores[layer] = store

    checkpoint_path = output / "mlp_features" / "extraction_checkpoint.json"
    completed = 0
    if args.resume and checkpoint_path.exists():
        completed = int(json.load(open(checkpoint_path))["completed_rows"])
    activation_cache = {}
    hooks = []
    for layer in args.layers:
        block_index = layer - 1
        if block_index < 0 or block_index >= len(layers):
            raise ValueError(
                f"Hidden-state Layer {layer} does not map to a transformer block; "
                "Layer 0 is the embedding output"
            )
        def make_hook(layer_id):
            def hook(_module, inputs):
                activation_cache[layer_id] = inputs[0].detach()
            return hook
        hooks.append(layers[block_index].mlp.down_proj.register_forward_pre_hook(make_hook(layer)))

    print(json.dumps({
        "phase": "mlp_extraction",
        "rows": len(candidates),
        "layers": list(args.layers),
        "intermediate_size": hidden,
        "resume_from": completed,
        "trainable_parameters": trainable,
        "training_mode": model.training,
    }), flush=True)
    try:
        with torch.inference_mode():
            for row_index in range(completed, len(candidates)):
                text = str(candidates[row_index]["text"])
                expected_tokens, chunks = make_windows(tokenizer, text, args.max_length, args.overlap)
                sums = {layer: np.zeros(hidden, dtype=np.float64) for layer in args.layers}
                count = 0
                for start in range(0, len(chunks), args.batch_size):
                    batch = chunks[start:start + args.batch_size]
                    width = max(len(chunk["ids"]) for chunk in batch)
                    input_ids = torch.full((len(batch), width), tokenizer.pad_token_id,
                                           dtype=torch.long, device=args.device)
                    attention_mask = torch.zeros_like(input_ids)
                    for batch_index, chunk in enumerate(batch):
                        ids = torch.tensor(chunk["ids"], dtype=torch.long, device=args.device)
                        input_ids[batch_index, :len(ids)] = ids
                        attention_mask[batch_index, :len(ids)] = 1
                    activation_cache.clear()
                    model(input_ids=input_ids, attention_mask=attention_mask)
                    for batch_index, chunk in enumerate(batch):
                        begin = chunk["content_offset"] + chunk["unique_offset"]
                        end = chunk["content_offset"] + chunk["content_length"]
                        if end <= begin:
                            continue
                        count += end - begin
                        for layer in args.layers:
                            sums[layer] += activation_cache[layer][batch_index, begin:end].float().sum(0).cpu().numpy()
                if count != expected_tokens:
                    raise RuntimeError(f"Token accounting failed row={row_index}: {count} != {expected_tokens}")
                for layer in args.layers:
                    stores[layer][row_index] = (sums[layer] / count).astype(np.float16)
                    stores[layer].flush()
                atomic_json(checkpoint_path, {"completed_rows": row_index + 1, "total_rows": len(candidates)})
                if (row_index + 1) % 10 == 0 or row_index + 1 == len(candidates):
                    print(json.dumps({"phase": "mlp_extraction", "processed": row_index + 1,
                                      "total": len(candidates)}), flush=True)
    finally:
        for hook in hooks:
            hook.remove()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def evaluate_masked(clf, X, y, units, baseline):
    masked = np.array(X, dtype=np.float32, copy=True)
    masked[:, units] = baseline[units]
    probability = clf.predict_proba(masked)[:, 1]
    return metrics(y, probability), probability


def run_scoring(args, candidates, binary_rows, manifest, output):
    position = {int(row["feature_index"]): i for i, row in enumerate(binary_rows)}
    split_indices = {}
    split_labels = {}
    for split_name in ("train", "val", "test"):
        rows = [row for row in manifest["records"] if row["split"] == split_name]
        split_indices[split_name] = np.asarray([position[int(row["feature_index"])] for row in rows], dtype=np.int64)
        split_labels[split_name] = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    summary_rows = []
    for layer in args.layers:
        store = np.load(output / "mlp_features" / f"layer_{layer:02d}.npy", mmap_mode="r")
        X_train = np.asarray(store[split_indices["train"]], dtype=np.float32)
        X_val = np.asarray(store[split_indices["val"]], dtype=np.float32)
        X_test = np.asarray(store[split_indices["test"]], dtype=np.float32)
        y_train, y_val, y_test = (split_labels[x] for x in ("train", "val", "test"))
        clf = make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=3000, class_weight="balanced", random_state=args.seed,
        ))
        clf.fit(X_train, y_train)
        scaler = clf.named_steps["standardscaler"]
        linear = clf.named_steps["logisticregression"]
        raw_weight = linear.coef_[0] / scaler.scale_
        raw_bias = float(linear.intercept_[0] - np.dot(raw_weight, scaler.mean_))
        p_train = clf.predict_proba(X_train)[:, 1]
        residual = p_train - y_train
        mean_activation = np.mean(X_train, axis=0)
        mean_abs_activation = np.mean(np.abs(X_train), axis=0)
        mean_abs_gradient = np.mean(np.abs(residual)) * np.abs(raw_weight)
        score = np.abs(mean_abs_gradient * mean_activation * raw_weight)
        elbow = kneedle(score)
        order = elbow["order"]
        units = order[:elbow["selected_count"]]
        layer_dir = output / f"layer_{layer:02d}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        np.save(layer_dir / "scores.npy", score.astype(np.float32))
        np.save(layer_dir / "mean_activation.npy", mean_activation.astype(np.float32))
        np.save(layer_dir / "mean_abs_activation.npy", mean_abs_activation.astype(np.float32))
        np.save(layer_dir / "mean_abs_gradient.npy", mean_abs_gradient.astype(np.float32))
        np.save(layer_dir / "raw_probe_weight.npy", raw_weight.astype(np.float32))
        rows = []
        for rank, unit in enumerate(order):
            rows.append({
                "rank": rank + 1,
                "unit": int(unit),
                "score": float(score[unit]),
                "mean_activation": float(mean_activation[unit]),
                "mean_abs_activation": float(mean_abs_activation[unit]),
                "mean_abs_gradient": float(mean_abs_gradient[unit]),
                "raw_probe_weight": float(raw_weight[unit]),
                "selected_by_kneedle": bool(rank < len(units)),
            })
        write_csv(layer_dir / "unit_scores.csv", rows)
        serializable_elbow = {key: value for key, value in elbow.items() if key != "order"}
        atomic_json(layer_dir / "selected_units.json", {
            "seed": args.seed, "layer": layer, "unit_type": "CodeGemma MLP intermediate neurons",
            "transformer_block_zero_based": layer - 1,
            "candidate_units": int(store.shape[1]), "selected_count": int(len(units)),
            "selected_units": [int(u) for u in units],
            "score_formula": "abs(mean_abs_gradient * mean_activation * raw_probe_weight)",
            "probe_raw_bias": raw_bias,
            "kneedle": serializable_elbow,
        })
        full_val_p = clf.predict_proba(X_val)[:, 1]
        full_test_p = clf.predict_proba(X_test)[:, 1]
        full_val = metrics(y_val, full_val_p)
        full_test = metrics(y_test, full_test_p)
        reduced = make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=3000, class_weight="balanced", random_state=args.seed,
        ))
        reduced.fit(X_train[:, units], y_train)
        reduced_val_p = reduced.predict_proba(X_val[:, units])[:, 1]
        reduced_test_p = reduced.predict_proba(X_test[:, units])[:, 1]
        reduced_val = metrics(y_val, reduced_val_p)
        reduced_test = metrics(y_test, reduced_test_p)
        masked_val, masked_val_p = evaluate_masked(clf, X_val, y_val, units, mean_activation)
        masked_test, masked_test_p = evaluate_masked(clf, X_test, y_test, units, mean_activation)
        rng = np.random.default_rng(args.seed + 1000 + layer)
        random_val, random_test = [], []
        for _ in range(args.random_repeats):
            random_units = rng.choice(X_train.shape[1], size=len(units), replace=False)
            rv, _ = evaluate_masked(clf, X_val, y_val, random_units, mean_activation)
            rt, _ = evaluate_masked(clf, X_test, y_test, random_units, mean_activation)
            random_val.append(rv); random_test.append(rt)
        def avg(rows_, key):
            vals = np.asarray([r[key] for r in rows_], dtype=np.float64)
            return {"mean": float(vals.mean()), "std": float(vals.std(ddof=1))}
        row = {
            "seed": args.seed, "layer": layer, "train_contracts": len(y_train),
            "validation_contracts": len(y_val), "test_contracts": len(y_test),
            "candidate_mlp_neurons": int(store.shape[1]), "kneedle_selected_neurons": int(len(units)),
            "full_probe_validation": full_val, "full_probe_test": full_test,
            "selected_neurons_probe_validation": reduced_val,
            "selected_neurons_probe_test": reduced_test,
            "masked_fdu_validation": masked_val, "masked_fdu_test": masked_test,
            "masked_random_validation": {k: avg(random_val, k) for k in full_val},
            "masked_random_test": {k: avg(random_test, k) for k in full_test},
            "validation_auroc_ci": bootstrap_auroc(y_val, full_val_p, args.seed + layer),
            "selected_validation_auroc_ci": bootstrap_auroc(y_val, reduced_val_p, args.seed + 50 + layer),
            "masked_validation_auroc_ci": bootstrap_auroc(y_val, masked_val_p, args.seed + 100 + layer),
            "test_auroc_ci": bootstrap_auroc(y_test, full_test_p, args.seed + 200 + layer),
            "selected_test_auroc_ci": bootstrap_auroc(y_test, reduced_test_p, args.seed + 250 + layer),
            "masked_test_auroc_ci": bootstrap_auroc(y_test, masked_test_p, args.seed + 300 + layer),
            "random_repeats": args.random_repeats,
        }
        summary_rows.append(row)
        print(json.dumps({"phase": "scoring", "layer": layer,
                          "selected_neurons": len(units), "validation_auroc": full_val["auroc"]}), flush=True)
    atomic_json(output / "summary.json", {
        "model": "CodeGemma-7B", "seed": args.seed,
        "task": "SmartIntent binary unsafe vs normal", "critical_layers": list(args.layers),
        "data": {"candidate_contracts": len(candidates), "train": 1280, "validation": 319, "test": 401,
                 "source": str(args.union_path), "binary_task": str(args.binary_path),
                 "manifest": str(args.manifest_path)},
        "selection_uses_test": False, "selection_uses_validation": False,
        "unit_type": "CodeGemma MLP intermediate neurons (intermediate_size=24576)",
        "results": summary_rows,
    })
    write_csv(output / "summary.csv", [{
        "layer": r["layer"], "kneedle_selected_neurons": r["kneedle_selected_neurons"],
        "full_val_auroc": r["full_probe_validation"]["auroc"],
        "full_test_auroc": r["full_probe_test"]["auroc"],
        "selected_neurons_val_auroc": r["selected_neurons_probe_validation"]["auroc"],
        "selected_neurons_test_auroc": r["selected_neurons_probe_test"]["auroc"],
        "masked_fdu_val_auroc": r["masked_fdu_validation"]["auroc"],
        "masked_fdu_test_auroc": r["masked_fdu_test"]["auroc"],
        "random_val_auroc_mean": r["masked_random_validation"]["auroc"]["mean"],
        "random_test_auroc_mean": r["masked_random_test"]["auroc"]["mean"],
    } for r in summary_rows])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="base/CodeGemma-7B")
    parser.add_argument("--union-path", default="runs/stage3_smartintent_sliding/sliding_union.jsonl")
    parser.add_argument("--binary-path", default="runs/stage3_smartintent_sliding/sliding_tasks/binary.jsonl")
    parser.add_argument("--manifest-path", default="runs/stage4_smartintent_full_trials/manifests/trial_seed_0.json")
    parser.add_argument("--output-dir", default="runs/stage9_codegemma_seed0_dna_mlp")
    parser.add_argument("--layers", nargs="+", type=int, default=list(DEFAULT_LAYERS))
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--overlap", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--random-repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    output = Path(args.output_dir)
    union_rows = load_jsonl(args.union_path)
    binary_rows = load_jsonl(args.binary_path)
    candidates = []
    for row in binary_rows:
        candidates.append(union_rows[int(row["feature_index"])])
    manifest = json.load(open(args.manifest_path, encoding="utf-8"))
    split_counts = {}
    for rec in manifest["records"]:
        split_counts[rec["split"]] = split_counts.get(rec["split"], 0) + 1
    atomic_json(output / "dataset_manifest.json", {
        "seed": args.seed, "source_rows": len(union_rows), "candidate_rows": len(candidates),
        "binary_labels": {"unsafe": int(sum(r["label"] for r in binary_rows)),
                          "normal": int(sum(1 - r["label"] for r in binary_rows))},
        "split_counts": split_counts, "candidate_layers": list(args.layers),
        "max_length": args.max_length, "overlap": args.overlap,
        "group_aware": True, "selection_uses_test": False,
    })
    run_extraction(args, candidates, output)
    run_scoring(args, candidates, binary_rows, manifest, output)
    atomic_json(output / "run_metadata.json", {
        "status": "complete", "model": "CodeGemma-7B", "seed": args.seed,
        "critical_layers": list(args.layers), "frozen_backbone": True,
        "feature_type": "mean pooled MLP down_proj pre-hook activation",
        "layer_mapping": "hidden-state Layer i maps to zero-based transformer block i-1; Layer 0 is embeddings",
        "intermediate_size": 24576,
    })
    print(json.dumps({"status": "complete", "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
