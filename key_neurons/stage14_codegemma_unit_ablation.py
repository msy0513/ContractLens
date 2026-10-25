#!/usr/bin/env python3
"""Fixed-probe unit ablation for the CodeGemma unsafe/normal task.

This is a Solidity-specific intervention experiment.  The cached CodeGemma
MLP activations are used as the representation, one probe is fitted on the
unmasked representation, and the same probe is evaluated after masking
stable, random, or magnitude-matched non-stable units.  Test data is never
used for selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


LAYERS = (16, 18, 19, 21)
CS = (0.001, 0.01, 0.1, 1.0, 10.0)


def atomic_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def metric(y, p, threshold):
    pred = np.asarray(p) >= threshold
    return {"accuracy": float(accuracy_score(y, pred)),
            "auroc": float(roc_auc_score(y, p))}


def fit_probe(X, y, groups, seed):
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    X_train = X["train"]
    cv_rows = []
    for C in CS:
        fold = []
        for k, (tr, va) in enumerate(cv.split(X_train, y["train"], groups)):
            scaler = StandardScaler()
            xt = scaler.fit_transform(X_train[tr])
            xv = scaler.transform(X_train[va])
            clf = LogisticRegression(C=C, solver="liblinear", class_weight="balanced",
                                     max_iter=3000, random_state=seed + k)
            clf.fit(xt, y["train"][tr])
            fold.append(roc_auc_score(y["train"][va], clf.predict_proba(xv)[:, 1]))
        cv_rows.append({"C": C, "mean_auroc": float(np.mean(fold)),
                        "std_auroc": float(np.std(fold, ddof=1)),
                        "fold_aurocs": [float(v) for v in fold]})
    best = max(r["mean_auroc"] for r in cv_rows)
    chosen_C = min(r["C"] for r in cv_rows if r["mean_auroc"] >= best - 0.002)
    scaler = StandardScaler()
    xt = scaler.fit_transform(X["train"])
    xv = scaler.transform(X["val"])
    xe = scaler.transform(X["test"])
    clf = LogisticRegression(C=chosen_C, solver="liblinear", class_weight="balanced",
                             max_iter=3000, random_state=seed)
    clf.fit(xt, y["train"])
    val_p = clf.predict_proba(xv)[:, 1]
    # Threshold is locked once on validation; every intervention uses it.
    candidates = np.unique(np.r_[0.5, np.linspace(0.05, 0.95, 181), val_p])
    acc = np.asarray([accuracy_score(y["val"], val_p >= t) for t in candidates])
    threshold = float(candidates[acc == acc.max()][np.argmin(np.abs(candidates[acc == acc.max()] - 0.5))])
    return scaler, clf, chosen_C, threshold, cv_rows


def grouped_bootstrap(y, probs, groups, threshold, repeats, seed):
    """Paired creator-group bootstrap CIs for ACC and AUROC."""
    by_group = defaultdict(list)
    for i, g in enumerate(groups):
        by_group[str(g)].append(i)
    names = np.asarray(list(by_group), dtype=object)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(repeats):
        sampled = rng.choice(names, size=len(names), replace=True)
        ix = np.asarray([i for g in sampled for i in by_group[str(g)]], dtype=np.int64)
        if np.unique(y[ix]).size < 2:
            continue
        vals.append((accuracy_score(y[ix], probs[ix] >= threshold),
                     roc_auc_score(y[ix], probs[ix])))
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "accuracy": {"mean": float(arr[:, 0].mean()), "lower": float(np.percentile(arr[:, 0], 2.5)),
                     "upper": float(np.percentile(arr[:, 0], 97.5))},
        "auroc": {"mean": float(arr[:, 1].mean()), "lower": float(np.percentile(arr[:, 1], 2.5)),
                  "upper": float(np.percentile(arr[:, 1], 97.5))},
        "repeats": int(len(arr)),
    }


def load_stable(path: Path):
    result = {}
    for layer in LAYERS:
        with (path / f"layer_{layer}" / "stability_units.csv").open(encoding="utf-8") as f:
            result[layer] = np.asarray([int(r["unit"]) for r in csv.DictReader(f)
                                        if r["stable"].lower() == "true"], dtype=np.int64)
    return result


def choose_hard_random(stable, available, train_matrix, seed):
    """Select non-key units by nearest train mean-absolute activation."""
    chosen = {}
    diagnostics = {}
    for layer in LAYERS:
        key = np.asarray(stable[layer], dtype=np.int64)
        units = np.asarray(available[layer], dtype=np.int64)
        nonkey = np.setdiff1d(units, key, assume_unique=True)
        key_mag = np.mean(np.abs(train_matrix[layer][:, np.searchsorted(units, key)]), axis=0)
        non_mag = np.mean(np.abs(train_matrix[layer][:, np.searchsorted(units, nonkey)]), axis=0)
        order = np.argsort(non_mag, kind="stable")
        sorted_mag = non_mag[order]
        used = np.zeros(len(nonkey), dtype=bool)
        chosen_pos = []
        # Greedy nearest-neighbour matching in activation-magnitude space.
        # Stable targets are processed in ascending order for deterministic
        # one-to-one matching; the non-key pool is much larger than the key set.
        for target in key_mag[np.argsort(key_mag, kind="stable")]:
            p = int(np.searchsorted(sorted_mag, target, side="left"))
            best = None
            radius = 0
            while best is None:
                for q in (p - radius, p + radius):
                    if 0 <= q < len(sorted_mag) and not used[q]:
                        if best is None or abs(sorted_mag[q] - target) < abs(sorted_mag[best] - target):
                            best = q
                radius += 1
            used[best] = True
            chosen_pos.append(best)
        selected = nonkey[order[np.asarray(chosen_pos, dtype=np.int64)]]
        selected = np.asarray(sorted(selected), dtype=np.int64)
        chosen[layer] = selected
        sel_mag = np.mean(np.abs(train_matrix[layer][:, np.searchsorted(units, selected)]), axis=0)
        diagnostics[str(layer)] = {
            "key_count": int(len(key)), "decoy_count": int(len(selected)),
            "key_mean_abs_activation": float(np.mean(key_mag)),
            "decoy_mean_abs_activation": float(np.mean(sel_mag)),
            "absolute_mean_difference": float(abs(np.mean(key_mag) - np.mean(sel_mag))),
        }
    return chosen, diagnostics


def main():
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--feature-dir", default=str(root / "runs/stage13_codegemma_seed0_neuron_specificity_fp16/features"))
    ap.add_argument("--stable-dir", default=str(root / "runs/stage10_codegemma_seed0_stable_units"))
    ap.add_argument("--binary", default=str(root / "runs/stage3_smartintent_sliding/sliding_tasks/binary.jsonl"))
    ap.add_argument("--manifest", default=str(root / "runs/stage4_smartintent_full_trials/manifests/trial_seed_0.json"))
    ap.add_argument("--output", default=str(root / "runs/stage14_codegemma_seed0_unit_ablation"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--random-repeats", type=int, default=30)
    ap.add_argument("--bootstrap-repeats", type=int, default=1000)
    ap.add_argument("--fixed-C", type=float, default=None,
                    help="Reuse a previously validation-locked C and skip CV")
    args = ap.parse_args()
    out, fdir = Path(args.output), Path(args.feature_dir)
    out.mkdir(parents=True, exist_ok=True)

    binary = load_jsonl(Path(args.binary))
    manifest = json.load(open(args.manifest, encoding="utf-8"))
    pos = {int(r["feature_index"]): i for i, r in enumerate(binary)}
    split = {s: [] for s in ("train", "val", "test")}
    groups = {s: [] for s in split}
    for r in manifest["records"]:
        split[r["split"]].append(pos[int(r["feature_index"])])
        groups[r["split"]].append(str(r["group"]))
    split = {s: np.asarray(v, dtype=np.int64) for s, v in split.items()}
    groups = {s: np.asarray(v, dtype=object) for s, v in groups.items()}
    y_all = np.asarray([int(r["label"]) for r in binary], dtype=np.int64)
    y = {s: y_all[split[s]] for s in split}

    stable = load_stable(Path(args.stable_dir))
    available, arrays = {}, {}
    for layer in LAYERS:
        available[layer] = np.load(fdir / f"layer_{layer}_global_units.npy")
        arrays[layer] = np.asarray(np.load(fdir / f"layer_{layer}_selected_union.f16.npy", mmap_mode="r"), dtype=np.float32)
        if not np.all(np.isin(stable[layer], available[layer])):
            raise RuntimeError(f"Stable units missing from cached columns at layer {layer}")

    # Stage 13 feature columns are sorted global unit IDs, so the mapping is exact.
    X = {s: np.concatenate([arrays[l][split[s]] for l in LAYERS], axis=1) for s in split}
    if args.fixed_C is None:
        scaler, clf, chosen_C, threshold, cv_rows = fit_probe(X, y, groups["train"], args.seed)
    else:
        chosen_C = float(args.fixed_C)
        scaler = StandardScaler(); xt = scaler.fit_transform(X["train"])
        xv = scaler.transform(X["val"]); xe = scaler.transform(X["test"])
        clf = LogisticRegression(C=chosen_C, solver="liblinear", class_weight="balanced",
                                 max_iter=3000, random_state=args.seed)
        clf.fit(xt, y["train"])
        val_p = clf.predict_proba(xv)[:, 1]
        candidates = np.unique(np.r_[0.5, np.linspace(0.05, 0.95, 181), val_p])
        acc = np.asarray([accuracy_score(y["val"], val_p >= t) for t in candidates])
        threshold = float(candidates[acc == acc.max()][np.argmin(np.abs(candidates[acc == acc.max()] - 0.5))])
        cv_rows = [{"C": chosen_C, "mean_auroc": None, "note": "reused validation-locked C"}]
    probs = {"base": {s: clf.predict_proba(scaler.transform(X[s]))[:, 1] for s in split}}

    offsets = {}
    cursor = 0
    for layer in LAYERS:
        offsets[layer] = (cursor, cursor + len(available[layer]))
        cursor += len(available[layer])
    key_cols = {l: np.searchsorted(available[l], stable[l]) for l in LAYERS}
    # Mean absolute activation is calculated only on train, never test.
    hard, hard_diag = choose_hard_random(stable, available,
                                         {l: arrays[l][split["train"]] for l in LAYERS}, args.seed)
    hard_cols = {l: np.searchsorted(available[l], hard[l]) for l in LAYERS}
    rng = np.random.default_rng(args.seed + 14000)
    random_sets = []
    for rep in range(args.random_repeats):
        rec = {}
        for l in LAYERS:
            nonkey = np.setdiff1d(available[l], stable[l], assume_unique=True)
            rec[l] = np.sort(rng.choice(nonkey, size=len(stable[l]), replace=False))
        random_sets.append(rec)

    def mask_matrix(matrix, selected):
        z = np.array(matrix, dtype=np.float32, copy=True)
        for l, units in selected.items():
            cols = np.searchsorted(available[l], units) + offsets[l][0]
            z[:, cols] = 0.0
        return z

    selections = {"stable": stable, "hard_random": hard}
    for name, selected in selections.items():
        probs[name] = {}
        for s in split:
            probs[name][s] = clf.predict_proba(scaler.transform(mask_matrix(X[s], selected)))[:, 1]
    probs["random"] = {"val": [], "test": []}
    for selected in random_sets:
        for s in ("val", "test"):
            probs["random"][s].append(clf.predict_proba(scaler.transform(mask_matrix(X[s], selected)))[:, 1])
    probs["random"]["val"] = np.asarray(probs["random"]["val"])
    probs["random"]["test"] = np.asarray(probs["random"]["test"])

    def table_row(label, p, s="test"):
        m = metric(y[s], p, threshold)
        base = metric(y[s], probs["base"][s], threshold)
        return {"method": label, "accuracy": m["accuracy"],
                "auroc": m["auroc"], "display_accuracy": f"{100*m['accuracy']:.1f} ({100*(m['accuracy']-base['accuracy']):+.1f})",
                "display_auroc": f"{100*m['auroc']:.1f} ({100*(m['auroc']-base['auroc']):+.1f})"}

    base_test = metric(y["test"], probs["base"]["test"], threshold)
    random_mean_metrics = {
        "accuracy": float(np.mean([metric(y["test"], p, threshold)["accuracy"] for p in probs["random"]["test"]])),
        "auroc": float(np.mean([metric(y["test"], p, threshold)["auroc"] for p in probs["random"]["test"]]))
    }
    # Table values for the random control are means of per-repeat metrics,
    # rather than metrics of an averaged probability vector.
    rows = [table_row("Base CodeGemma + Probe", probs["base"]["test"]),
            {"method": "+ Masked Random Units (mean of 30)",
             "accuracy": random_mean_metrics["accuracy"], "auroc": random_mean_metrics["auroc"],
             "display_accuracy": f"{100*random_mean_metrics['accuracy']:.1f} ({100*(random_mean_metrics['accuracy']-base_test['accuracy']):+.1f})",
             "display_auroc": f"{100*random_mean_metrics['auroc']:.1f} ({100*(random_mean_metrics['auroc']-base_test['auroc']):+.1f})"},
            table_row("+ Masked Magnitude-Matched Units", probs["hard_random"]["test"]),
            table_row("+ Masked Key Units", probs["stable"]["test"])]

    # Group-bootstrap CIs on the final fixed test outputs. These are reporting
    # intervals only; no test statistic controls any selection decision.
    cis = {}
    for name in ("base", "stable", "hard_random"):
        cis[name] = grouped_bootstrap(y["test"], probs[name]["test"], groups["test"], threshold,
                                      args.bootstrap_repeats, args.seed + 200 + len(cis))
    random_boot = []
    for i, p in enumerate(probs["random"]["test"]):
        random_boot.append(grouped_bootstrap(y["test"], p, groups["test"], threshold,
                                             max(100, args.bootstrap_repeats // 5), args.seed + 500 + i))
    for metric_name in ("accuracy", "auroc"):
        vals = np.asarray([x[metric_name]["mean"] for x in random_boot])
        cis["random"] = cis.get("random", {})
        cis["random"][metric_name] = {"mean": float(vals.mean()), "lower": float(vals.min()), "upper": float(vals.max()),
                                       "repeat_mean_ci_note": "min/max of per-repeat bootstrap means"}
    for row in rows:
        row["accuracy_percent"] = 100 * row["accuracy"]
        row["auroc_percent"] = 100 * row["auroc"]
        row["test_n"] = int(len(y["test"]))

    atomic_json(out / "config.json", {
        "seed": args.seed, "backbone": "CodeGemma-7B (frozen; cached activations)",
        "task": "unsafe (at least one unsafe intent) vs normal", "layers": list(LAYERS),
        "unit_type": "post-gating MLP intermediate units at down_proj input",
        "intervention": "activation zeroing", "probe": "single fixed full-feature Logistic Regression",
        "chosen_C": chosen_C, "validation_threshold": threshold,
        "random_repeats": args.random_repeats, "bootstrap_repeats": args.bootstrap_repeats,
        "test_used_for_selection": False,
    })
    atomic_json(out / "audit.json", {
        "split_counts": {s: int(len(split[s])) for s in split},
        "split_label_counts": {s: {"0": int((y[s] == 0).sum()), "1": int((y[s] == 1).sum())} for s in split},
        "group_overlap": {"train_val": sorted(set(groups["train"]) & set(groups["val"])),
                          "train_test": sorted(set(groups["train"]) & set(groups["test"])),
                          "val_test": sorted(set(groups["val"]) & set(groups["test"]))},
        "stable_counts": {str(l): int(len(stable[l])) for l in LAYERS},
        "hard_random_diagnostics": hard_diag, "feature_dimensions": int(X["train"].shape[1]),
        "test_used_for_selection": False,
    })
    atomic_json(out / "cv_probe.json", cv_rows)
    atomic_json(out / "bootstrap_cis.json", cis)
    atomic_json(out / "table3_style.json", {"baseline_test": base_test, "rows": rows,
                                             "metric_note": "percentages; parenthesized values are change from Base"})
    with (out / "table3_style.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["Method", "ACC (↑)", "AUROC (↑)"])
        w.writeheader()
        for row in rows:
            w.writerow({"Method": row["method"], "ACC (↑)": row["display_accuracy"],
                        "AUROC (↑)": row["display_auroc"]})
    with (out / "table3_style_per_layer.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["layer", "method", "accuracy", "auroc", "delta_accuracy", "delta_auroc"])
        w.writeheader()
        for l in LAYERS:
            cols = np.arange(offsets[l][0], offsets[l][1])
            # Layer-only report is a diagnostic using the same global probe and
            # masking that layer; it does not replace the combined table.
            for name, selected in (("base", {}), ("random", {l: random_sets[0][l]}),
                                   ("hard_random", {l: hard[l]}), ("stable", {l: stable[l]})):
                pp = probs["base"]["test"] if name == "base" else clf.predict_proba(scaler.transform(mask_matrix(X["test"], selected)))[:, 1]
                m = metric(y["test"], pp, threshold)
                w.writerow({"layer": l, "method": name, "accuracy": m["accuracy"], "auroc": m["auroc"],
                            "delta_accuracy": m["accuracy"] - base_test["accuracy"],
                            "delta_auroc": m["auroc"] - base_test["auroc"]})
    print(json.dumps({"status": "complete", "output": str(out), "table": rows}, indent=2), flush=True)


if __name__ == "__main__":
    main()
