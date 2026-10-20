#!/usr/bin/env python3
"""Validation-locked, stability-aware unsafe-contract MLP unit discovery.

Uses the existing frozen CodeGemma MLP features. Train is used for all model
fitting, scoring, Kneedle and cluster bootstrap. Validation selects among
predeclared score variants. Test is evaluated only after the choice is locked.
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


LAYERS = (16, 18, 19, 21)
CS = (0.01, 0.1, 1.0, 10.0)
VARIANTS = ("signed", "absolute", "class_contrast")


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2)
    os.replace(tmp, path)


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def metric(y, p):
    pred = p >= .5
    return {"accuracy": float(accuracy_score(y, pred)), "auroc": float(roc_auc_score(y, p)),
            "auprc": float(average_precision_score(y, p)), "f1": float(f1_score(y, pred))}


def corrected_kneedle(score):
    order = np.argsort(-score, kind="stable")
    s = np.asarray(score[order], dtype=np.float64)
    n = len(s); x = np.linspace(0., 1., n); span = float(s[0] - s[-1])
    if n <= 2 or span <= 0:
        return order, max(1, n), 0.0
    y = (s[0] - s) / span
    diff = y - x
    interior = np.arange(1, n - 1)
    idx = int(interior[np.argmax(diff[interior])])
    return order, idx + 1, float(diff[idx])


def make_probe(C, seed):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, solver="liblinear", tol=1e-3, max_iter=1000,
                           class_weight="balanced", random_state=seed),
    )


def raw_probe(clf):
    scaler = clf.named_steps["standardscaler"]
    linear = clf.named_steps["logisticregression"]
    w = linear.coef_[0] / scaler.scale_
    b = float(linear.intercept_[0] - np.dot(w, scaler.mean_))
    return w, b


def scores(X, y, clf):
    w, _ = raw_probe(clf)
    residual_factor = float(np.mean(np.abs(clf.predict_proba(X)[:, 1] - y)))
    gbar = residual_factor * np.abs(w)
    signed = np.abs(np.mean(X, axis=0))
    absolute = np.mean(np.abs(X), axis=0)
    class_contrast = np.abs(np.mean(X[y == 1], axis=0) - np.mean(X[y == 0], axis=0))
    pooled_sd = np.sqrt((np.var(X[y == 1], axis=0) + np.var(X[y == 0], axis=0)) / 2.0)
    effect_size = class_contrast / (pooled_sd + 1e-8)
    return {
        "signed": np.abs(gbar * signed * w),
        "absolute": np.abs(gbar * absolute * w),
        "class_contrast": np.abs(gbar * class_contrast * w),
    }, {"raw_weight": w, "mean_abs_gradient": gbar, "signed_activation": signed,
        "absolute_activation": absolute, "class_contrast": class_contrast,
        "standardized_effect_size": effect_size}


def masked_probability(clf, X, units, baseline):
    Z = np.array(X, dtype=np.float32, copy=True)
    if len(units):
        Z[:, units] = baseline[units]
    return clf.predict_proba(Z)[:, 1]


def cv_select_c(X, y, groups, seed):
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    rows = []
    for C in CS:
        vals = []
        for fold, (tr, va) in enumerate(cv.split(X, y, groups)):
            clf = make_probe(C, seed + fold); clf.fit(X[tr], y[tr])
            vals.append(roc_auc_score(y[va], clf.predict_proba(X[va])[:, 1]))
        rows.append({"C": C, "mean_auroc": float(np.mean(vals)), "std_auroc": float(np.std(vals, ddof=1)),
                     "fold_aurocs": [float(x) for x in vals]})
    # Smallest C within 0.002 of the best: regularized, deterministic tie rule.
    best = max(r["mean_auroc"] for r in rows)
    chosen = min(r["C"] for r in rows if r["mean_auroc"] >= best - .002)
    return chosen, rows


def cluster_bootstrap_indices(groups, rng):
    lookup = defaultdict(list)
    for i, g in enumerate(groups): lookup[str(g)].append(i)
    keys = np.asarray(list(lookup), dtype=object)
    sampled = rng.choice(keys, size=len(keys), replace=True)
    return np.asarray([i for g in sampled for i in lookup[str(g)]], dtype=np.int64)


def bootstrap_ci(values):
    v = np.asarray(values, dtype=np.float64)
    return {"mean": float(v.mean()), "lower": float(np.percentile(v, 2.5)),
            "upper": float(np.percentile(v, 97.5)), "std": float(v.std(ddof=1))}


def paired_masking_ci(y, full_p, top_p, random_probabilities, seed, repeats=1000):
    rng = np.random.default_rng(seed); n = len(y); vals = []
    random_probabilities = np.asarray(random_probabilities)
    for _ in range(repeats):
        idx = rng.integers(0, n, n); yy = y[idx]
        if np.unique(yy).size < 2: continue
        r = int(rng.integers(0, len(random_probabilities)))
        top_drop = roc_auc_score(yy, full_p[idx]) - roc_auc_score(yy, top_p[idx])
        random_drop = roc_auc_score(yy, full_p[idx]) - roc_auc_score(yy, random_probabilities[r, idx])
        vals.append(top_drop - random_drop)
    return bootstrap_ci(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage9", default="runs/stage9_codegemma_seed0_dna_mlp")
    ap.add_argument("--binary", default="runs/stage3_smartintent_sliding/sliding_tasks/binary.jsonl")
    ap.add_argument("--manifest", default="runs/stage4_smartintent_full_trials/manifests/trial_seed_0.json")
    ap.add_argument("--output", default="runs/stage10_codegemma_seed0_stable_units")
    ap.add_argument("--bootstrap-repeats", type=int, default=100)
    ap.add_argument("--random-repeats", type=int, default=50)
    ap.add_argument("--stability-threshold", type=float, default=.70)
    ap.add_argument("--sufficiency-delta", type=float, default=.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layers", nargs="+", type=int, default=list(LAYERS))
    args = ap.parse_args()
    stage9, out = Path(args.stage9), Path(args.output); out.mkdir(parents=True, exist_ok=True)
    binary = [json.loads(x) for x in open(args.binary) if x.strip()]
    pos = {int(r["feature_index"]): i for i, r in enumerate(binary)}
    records = json.load(open(args.manifest))["records"]
    ix, labels, groups = {}, {}, {}
    for split in ("train", "val", "test"):
        rr = [r for r in records if r["split"] == split]
        ix[split] = np.asarray([pos[int(r["feature_index"])] for r in rr], dtype=np.int64)
        labels[split] = np.asarray([int(r["label"]) for r in rr], dtype=np.int64)
        groups[split] = np.asarray([str(r["group"]) for r in rr], dtype=object)
    summary = []
    for layer in args.layers:
        print(json.dumps({"phase": "layer_start", "layer": layer}), flush=True)
        store = np.load(stage9 / "mlp_features" / f"layer_{layer:02d}.npy", mmap_mode="r")
        Xtr = np.asarray(store[ix["train"]], dtype=np.float32)
        Xva = np.asarray(store[ix["val"]], dtype=np.float32)
        Xte = np.asarray(store[ix["test"]], dtype=np.float32)
        ytr, yva, yte = labels["train"], labels["val"], labels["test"]
        C, cv_rows = cv_select_c(Xtr, ytr, groups["train"], args.seed + layer)
        clf = make_probe(C, args.seed); clf.fit(Xtr, ytr); baseline = Xtr.mean(axis=0)
        full_va_p = clf.predict_proba(Xva)[:, 1]; full_te_p = clf.predict_proba(Xte)[:, 1]
        full_va, full_te = metric(yva, full_va_p), metric(yte, full_te_p)
        score_map, components = scores(Xtr, ytr, clf)
        variant_rows = []
        rng = np.random.default_rng(args.seed + 1000 + layer)
        for variant in VARIANTS:
            order, k, curvature = corrected_kneedle(score_map[variant]); units = order[:k]
            reduced = make_probe(C, args.seed); reduced.fit(Xtr[:, units], ytr)
            selected_va = metric(yva, reduced.predict_proba(Xva[:, units])[:, 1])
            selected_te = metric(yte, reduced.predict_proba(Xte[:, units])[:, 1])
            top_va_p = masked_probability(clf, Xva, units, baseline)
            random_va_ps = []
            for _ in range(args.random_repeats):
                ru = rng.choice(Xtr.shape[1], size=k, replace=False)
                random_va_ps.append(masked_probability(clf, Xva, ru, baseline))
            random_mean = float(np.mean([roc_auc_score(yva, p) for p in random_va_ps]))
            specificity = random_mean - roc_auc_score(yva, top_va_p)
            sufficient = selected_va["auroc"] >= full_va["auroc"] - args.sufficiency_delta
            variant_rows.append({"variant": variant, "kneedle_count": int(k), "curvature": curvature,
                "selected_validation": selected_va, "selected_test_development_only": selected_te,
                "top_mask_validation_auroc": float(roc_auc_score(yva, top_va_p)),
                "random_mask_validation_auroc_mean": random_mean,
                "specificity_margin": specificity, "sufficient": bool(sufficient)})
        eligible = [r for r in variant_rows if r["sufficient"]]
        pool = eligible if eligible else variant_rows
        chosen_row = max(pool, key=lambda r: (r["specificity_margin"], r["selected_validation"]["auroc"]))
        chosen = chosen_row["variant"]
        original_order, original_k, _ = corrected_kneedle(score_map[chosen])
        original_units = original_order[:original_k]
        frequency = np.zeros(Xtr.shape[1], dtype=np.int32); knee_counts = []
        rng_boot = np.random.default_rng(args.seed + 10000 + layer)
        for b in range(args.bootstrap_repeats):
            bi = cluster_bootstrap_indices(groups["train"], rng_boot)
            Xb, yb = Xtr[bi], ytr[bi]
            if np.unique(yb).size < 2: continue
            bclf = make_probe(C, args.seed + b + 1); bclf.fit(Xb, yb)
            bs, _ = scores(Xb, yb, bclf); bo, bk, _ = corrected_kneedle(bs[chosen])
            frequency[bo[:bk]] += 1; knee_counts.append(int(bk))
            if (b + 1) % 10 == 0:
                atomic_json(out / f"layer_{layer:02d}" / "bootstrap_checkpoint.json",
                            {"completed": b + 1, "total": args.bootstrap_repeats,
                             "chosen_variant": chosen, "knee_counts": knee_counts})
                print(json.dumps({"phase": "bootstrap", "layer": layer, "completed": b + 1,
                                  "total": args.bootstrap_repeats}), flush=True)
        denom = max(1, len(knee_counts)); selection_frequency = frequency / denom
        stable_units = np.flatnonzero(selection_frequency >= args.stability_threshold)
        stable_units = stable_units[np.argsort(-score_map[chosen][stable_units], kind="stable")]
        # Functional validation is defined even if no unit clears stability.
        eval_units = stable_units
        selected_va = selected_te = None
        if len(eval_units):
            reduced = make_probe(C, args.seed); reduced.fit(Xtr[:, eval_units], ytr)
            selected_va = metric(yva, reduced.predict_proba(Xva[:, eval_units])[:, 1])
            selected_te = metric(yte, reduced.predict_proba(Xte[:, eval_units])[:, 1])
        top_va_p = masked_probability(clf, Xva, eval_units, baseline)
        top_te_p = masked_probability(clf, Xte, eval_units, baseline)
        nonstable = np.setdiff1d(np.arange(Xtr.shape[1]), eval_units, assume_unique=False)
        rng_random = np.random.default_rng(args.seed + 20000 + layer)
        rin_va, rin_te, rex_va, rex_te = [], [], [], []
        for _ in range(args.random_repeats):
            count = len(eval_units)
            ru = rng_random.choice(Xtr.shape[1], size=count, replace=False) if count else np.array([], dtype=int)
            ex = rng_random.choice(nonstable, size=count, replace=False) if count and len(nonstable) >= count else ru
            rin_va.append(masked_probability(clf, Xva, ru, baseline)); rin_te.append(masked_probability(clf, Xte, ru, baseline))
            rex_va.append(masked_probability(clf, Xva, ex, baseline)); rex_te.append(masked_probability(clf, Xte, ex, baseline))
        val_ci = paired_masking_ci(yva, full_va_p, top_va_p, rin_va, args.seed + layer)
        test_ci = paired_masking_ci(yte, full_te_p, top_te_p, rin_te, args.seed + 100 + layer)
        layer_dir = out / f"layer_{layer:02d}"; layer_dir.mkdir(parents=True, exist_ok=True)
        np.save(layer_dir / "selection_frequency.npy", selection_frequency.astype(np.float32))
        for name, a in score_map.items(): np.save(layer_dir / f"score_{name}.npy", a.astype(np.float32))
        for name, a in components.items(): np.save(layer_dir / f"component_{name}.npy", np.asarray(a, dtype=np.float32))
        unit_rows = [{"unit": int(k), "selection_frequency": float(selection_frequency[k]),
                      "stable": bool(selection_frequency[k] >= args.stability_threshold),
                      "chosen_score": float(score_map[chosen][k]),
                      "effect_size": float(components["standardized_effect_size"][k])}
                     for k in np.argsort(-selection_frequency, kind="stable")]
        write_csv(layer_dir / "stability_units.csv", unit_rows)
        result = {"layer": layer, "chosen_C": C, "cv": cv_rows, "score_variants": variant_rows,
            "chosen_variant_by_validation": chosen, "original_kneedle_count": int(original_k),
            "bootstrap_repeats_completed": denom, "bootstrap_kneedle_count": bootstrap_ci(knee_counts),
            "stability_threshold": args.stability_threshold, "stable_unit_count": int(len(stable_units)),
            "stable_units": [int(x) for x in stable_units], "full_validation": full_va,
            "full_test_development_only": full_te, "stable_probe_validation": selected_va,
            "stable_probe_test_development_only": selected_te,
            "top_mask_validation": metric(yva, top_va_p), "top_mask_test_development_only": metric(yte, top_te_p),
            "random_in_validation_auroc": bootstrap_ci([roc_auc_score(yva, p) for p in rin_va]),
            "random_ex_validation_auroc": bootstrap_ci([roc_auc_score(yva, p) for p in rex_va]),
            "random_in_test_auroc_development_only": bootstrap_ci([roc_auc_score(yte, p) for p in rin_te]),
            "random_ex_test_auroc_development_only": bootstrap_ci([roc_auc_score(yte, p) for p in rex_te]),
            "paired_specificity_ci_validation": val_ci,
            "paired_specificity_ci_test_development_only": test_ci,
            "passes_sufficiency": bool(selected_va and selected_va["auroc"] >= full_va["auroc"] - args.sufficiency_delta),
            "passes_necessity_validation": bool(val_ci["lower"] > 0),
        }
        atomic_json(layer_dir / "result.json", result); summary.append(result)
        print(json.dumps({"phase": "layer_complete", "layer": layer, "chosen_variant": chosen,
                          "stable_units": len(stable_units), "necessity_ci_lower": val_ci["lower"]}), flush=True)
    atomic_json(out / "summary.json", {"model": "CodeGemma-7B", "seed": args.seed,
        "task": "SmartIntent binary unsafe vs normal", "data": {"train": 1280, "validation": 319,
        "development_test": 401}, "test_used_for_selection": False,
        "validation_selection_rule": "sufficiency within 0.02 then maximum masking specificity margin",
        "bootstrap_unit": "creator group", "bootstrap_repeats": args.bootstrap_repeats,
        "stability_threshold": args.stability_threshold, "results": summary})
    print(json.dumps({"status": "complete", "output": str(out)}), flush=True)


if __name__ == "__main__": main()
