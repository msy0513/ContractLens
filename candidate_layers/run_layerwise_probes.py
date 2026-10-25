#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def load_rows(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def split_indices(labels, groups, seed):
    idx = np.arange(len(labels))
    if groups is not None and len(set(groups)) > 1:
        try:
            outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
            train_val, test = next(outer.split(idx, labels, groups))
            inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed + 1000)
            train_rel, val_rel = next(inner.split(train_val, labels[train_val], np.asarray(groups)[train_val]))
            return train_val[train_rel], train_val[val_rel], test
        except Exception:
            pass
    train, test = train_test_split(idx, test_size=0.2, random_state=seed, stratify=labels)
    train, val = train_test_split(train, test_size=0.25, random_state=seed, stratify=labels[train])
    return np.asarray(train), np.asarray(val), np.asarray(test)


def split_indices_from_manifest(rows, manifest_path):
    """Map the fixed group-aware manifest split onto task-row positions."""
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    by_feature = {int(row.get("feature_index", i)): i for i, row in enumerate(rows)}
    split_indices = {name: [] for name in ("train", "val", "test")}
    seen = set()
    for record in manifest["records"]:
        feature_index = int(record["feature_index"])
        if feature_index not in by_feature:
            continue
        if feature_index in seen:
            raise ValueError(f"Duplicate feature_index in split manifest: {feature_index}")
        seen.add(feature_index)
        row_index = by_feature[feature_index]
        if int(record["label"]) != int(rows[row_index]["label"]):
            raise ValueError(f"Label mismatch for feature_index={feature_index}")
        split_indices[record["split"]].append(row_index)
    result = {name: np.asarray(values, dtype=np.int64) for name, values in split_indices.items()}
    if any(len(result[name]) == 0 for name in result):
        raise ValueError("The split manifest did not map to non-empty train/val/test task rows")
    if seen != set(by_feature):
        missing = sorted(set(by_feature) - seen)
        raise ValueError(f"Split manifest is missing {len(missing)} task rows; examples={missing[:10]}")
    return result


def metrics(y, score, pred):
    return {
        "f1": float(f1_score(y, pred)),
        "accuracy": float(accuracy_score(y, pred)),
        "auroc": float(roc_auc_score(y, score)) if len(set(y)) > 1 else float("nan"),
        "auprc": float(average_precision_score(y, score)),
    }


def make_probe(name, seed):
    if name == "logistic_regression":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed),
        )
    if name == "linear_svm":
        return make_pipeline(
            StandardScaler(),
            # The probe set has far fewer rows than hidden dimensions.  A
            # moderate iteration cap and looser tolerance avoid spending
            # minutes on a supplementary baseline while preserving the
            # linear decision boundary.
            LinearSVC(
                class_weight="balanced",
                random_state=seed,
                max_iter=800,
                tol=1e-3,
                dual=True,
            ),
        )
    if name == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(32,),
                max_iter=100,
                early_stopping=True,
                n_iter_no_change=10,
                random_state=seed,
            ),
        )
    raise ValueError(name)


def score_probe(probe, name, X):
    if name == "logistic_regression" or name == "mlp":
        return probe.predict_proba(X)[:, 1]
    return probe.decision_function(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", required=True)
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument(
        "--split-manifest",
        help="Optional fixed group-aware split manifest; when supplied, all estimator seeds use these same partitions.",
    )
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.data_path)
    labels = np.asarray([int(r["label"]) for r in rows], dtype=np.int64)
    feature_indices = np.asarray(
        [int(r.get("feature_index", i)) for i, r in enumerate(rows)], dtype=np.int64
    )
    groups = []
    has_group = False
    for i, row in enumerate(rows):
        group = row.get("contract_creator") or row.get("contract_address") or row.get("group")
        if group:
            has_group = True
            groups.append(str(group))
        else:
            groups.append(f"row-{i}")
    groups = groups if has_group else None
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    fixed_split = split_indices_from_manifest(rows, args.split_manifest) if args.split_manifest else None

    all_rows = []
    layer_paths = sorted(Path(args.features_dir).glob("layer_*.npy"))
    if not layer_paths:
        raise RuntimeError(f"No layer feature files found in {args.features_dir}")
    for seed in seeds:
        if fixed_split is None:
            train_idx, val_idx, test_idx = split_indices(labels, groups, seed)
        else:
            train_idx, val_idx, test_idx = (fixed_split[name] for name in ("train", "val", "test"))
        for layer_path in layer_paths:
            layer = int(layer_path.stem.split("_")[-1])
            X_all = np.load(layer_path, mmap_mode="r")
            X = X_all[feature_indices]
            if not np.isfinite(X).all():
                raise RuntimeError(f"Non-finite features in {layer_path}")
            zero_rows = np.flatnonzero(~np.any(X != 0, axis=1))
            if len(zero_rows):
                raise RuntimeError(
                    f"Unextracted all-zero features in {layer_path}; task rows {zero_rows[:10].tolist()}"
                )
            for probe_name in ["logistic_regression", "linear_svm", "mlp"]:
                probe = make_probe(probe_name, seed)
                probe.fit(X[train_idx], labels[train_idx])
                for split_name, split_idx in [("val", val_idx), ("test", test_idx)]:
                    score = score_probe(probe, probe_name, X[split_idx])
                    pred = (score >= 0).astype(int) if probe_name == "linear_svm" else (score >= 0.5).astype(int)
                    row = {
                        "seed": seed,
                        "layer": layer,
                        "probe": probe_name,
                        "split": split_name,
                        "n": int(len(split_idx)),
                    }
                    row.update(metrics(labels[split_idx], score, pred))
                    all_rows.append(row)

    with open(out / "layer_probe_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    grouped = {}
    for row in all_rows:
        key = (row["layer"], row["probe"], row["split"])
        grouped.setdefault(key, []).append(row)
    aggregate = []
    for (layer, probe, split), rows_for_key in sorted(grouped.items()):
        out_row = {"layer": layer, "probe": probe, "split": split, "n_seeds": len(rows_for_key)}
        for metric in ["f1", "accuracy", "auroc", "auprc"]:
            values = np.asarray([r[metric] for r in rows_for_key], dtype=float)
            out_row[f"{metric}_mean"] = float(np.nanmean(values))
            out_row[f"{metric}_std"] = float(np.nanstd(values))
        aggregate.append(out_row)
    with open(out / "layer_probe_aggregate.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(aggregate[0].keys()))
        writer.writeheader()
        writer.writerows(aggregate)

    with open(out / "split_metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "data_path": args.data_path,
                "features_dir": args.features_dir,
                "seeds": seeds,
                "group_aware": bool(groups is not None),
                "split_manifest": args.split_manifest,
                "fixed_split_shared_across_estimator_seeds": bool(fixed_split is not None),
                "n_rows": int(len(rows)),
                "label_counts": {"0": int((labels == 0).sum()), "1": int((labels == 1).sum())},
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
