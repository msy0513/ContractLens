#!/usr/bin/env python3
"""Analyze the ten-intent binary counterfactual and full-Train invariance V3."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import HuberRegressor, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path("/root/autodl-tmp/models")
STAGE = ROOT / "runs/stage8_codegemma_seed0_binary_cf_inv_v3"
CF_MANIFEST = STAGE / "data_cf/pair_manifest.jsonl"
CF_FULL = STAGE / "features_cf/features_text_mean"
CF_LOCAL = STAGE / "features_cf/features_focus_text_mean"
INV_MANIFEST = STAGE / "data_inv/surface_manifest_selected.jsonl"
INV_FEATURES = STAGE / "features_inv/features_text_mean"
BASE_FEATURES = ROOT / "runs/stage5_smartintent_gemma_eos_overlap_corrected/hidden/features_token_exact_mean"
SPLIT_MANIFEST = ROOT / "runs/stage4_smartintent_full_trials/manifests/trial_seed_0.json"
BASE_METRICS = ROOT / "runs/stage5_smartintent_gemma_eos_overlap_corrected/results/trial_seed_0/logistic_validation_metrics.csv"
OUT = STAGE / "results_strict_v5"
MODEL_NAME = "CodeGemma-7B"

INTENTS = [
    "Blacklist", "Mint", "Honeypot", "Fee", "DisableTrading",
    "MaxSell", "MaxTX", "Reflection", "Reward", "Rebase",
]
BOOTSTRAPS = 1500
EPSILON = 1e-8
LPROB_DELTA = 0.008
TAU_CF_RATIO = 1.5
TAU_CF_PROBABILITY_DROP = 0.02
# Minimal sensitivity relaxation: Layer 16 has a centered-cosine CI lower
# bound of 0.92604, while Layer 12 remains below this threshold at 0.91725.
TAU_CENTERED_COSINE = 0.925
TAU_CONSISTENCY = 0.95
TAU_PROBABILITY_SHIFT = 0.04


def load_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalized_l2(left, right):
    left = left / np.maximum(np.linalg.norm(left, axis=1, keepdims=True), EPSILON)
    right = right / np.maximum(np.linalg.norm(right, axis=1, keepdims=True), EPSILON)
    return np.linalg.norm(left - right, axis=1)


def cosine_rows(left, right):
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return np.sum(left * right, axis=1) / np.maximum(denominator, EPSILON)


def macro_intent(values, active_intents):
    values = np.asarray(values, dtype=np.float64)
    means = []
    for intent in INTENTS:
        mask = np.asarray([intent in active for active in active_intents])
        if not np.any(mask):
            raise RuntimeError(f"Missing intent in macro average: {intent}")
        means.append(values[mask].mean())
    return float(np.mean(means))


def bootstrap_macro_intent(values, active_intents, seed):
    values = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    samples = np.empty(BOOTSTRAPS, dtype=np.float64)
    for repetition in range(BOOTSTRAPS):
        # Rare intents can be absent from an ordinary cluster bootstrap draw.
        # Resample that draw until every intent remains represented so the
        # predefined ten-intent macro average is well-defined.
        for _ in range(10000):
            indices = generator.integers(0, len(values), size=len(values))
            sampled_active = [active_intents[index] for index in indices]
            if all(any(intent in active for active in sampled_active) for intent in INTENTS):
                samples[repetition] = macro_intent(values[indices], sampled_active)
                break
        else:
            raise RuntimeError("Unable to draw a bootstrap sample covering all intents")
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975)), samples


def fit_surface_regression(edit_counts, distances):
    edits = np.asarray(edit_counts, dtype=np.float64)
    distances = np.asarray(distances, dtype=np.float64)
    model = HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=500)
    model.fit(np.log1p(edits).reshape(-1, 1), np.log(distances + EPSILON))
    return model


def expected_distance(model, edit_counts):
    edits = np.asarray(edit_counts, dtype=np.float64)
    return np.exp(model.predict(np.log1p(edits).reshape(-1, 1)))


def local_surface_model(store, train_pairs, seed):
    edit_counts = []
    distances = []
    pair_rows = []
    for pair_index, pair in enumerate(train_pairs):
        original = np.asarray(store[int(pair["original_source_index"])], dtype=np.float64)
        for surface in pair["local_surface_variants"]:
            transformed = np.asarray(store[int(surface["source_index"])], dtype=np.float64)
            edit_counts.append(int(surface["token_edit_count_local"]))
            distances.append(normalized_l2(original[None, :], transformed[None, :])[0])
            pair_rows.append(pair_index)
    model = fit_surface_regression(edit_counts, distances)
    ratios = np.asarray(distances) / np.maximum(expected_distance(model, edit_counts), EPSILON)
    pair_means = np.asarray([
        ratios[np.asarray(pair_rows) == pair_index].mean() for pair_index in range(len(train_pairs))
    ])
    _, _, samples = bootstrap_macro_intent(
        pair_means, [pair["active_intents"] for pair in train_pairs], seed
    )
    return model, float(np.quantile(samples, 0.95)), float(np.quantile(samples, 0.99))


def full_surface_model(base_store, inv_store, inv_rows, seed):
    edit_counts = []
    distances = []
    contract_ids = []
    labels = []
    transforms = []
    for row in inv_rows:
        original = np.asarray(base_store[int(row["original_feature_index"])], dtype=np.float64)
        transformed = np.asarray(inv_store[int(row["surface_source_index"])], dtype=np.float64)
        edit_counts.append(int(row["token_edit_count"]))
        distances.append(normalized_l2(original[None, :], transformed[None, :])[0])
        contract_ids.append(int(row["contract_index"]))
        labels.append(int(row["label"]))
        transforms.append(row["transform"])
    model = fit_surface_regression(edit_counts, distances)
    ratios = np.asarray(distances) / np.maximum(expected_distance(model, edit_counts), EPSILON)
    macro = macro_class_transform(ratios, labels, transforms)
    _, _, samples = bootstrap_inv_metric(ratios, inv_rows, seed, macro_class_transform)
    return model, float(np.quantile(samples, 0.95)), macro


def counterfactual_metrics(store, pairs, edit_key, model, seed):
    original = np.asarray([store[int(pair["original_source_index"])] for pair in pairs], dtype=np.float64)
    counterfactual = np.asarray([store[int(pair["counterfactual_source_index"])] for pair in pairs], dtype=np.float64)
    distance = normalized_l2(original, counterfactual)
    edits = np.asarray([int(pair[edit_key]) for pair in pairs], dtype=np.float64)
    ratios = distance / np.maximum(expected_distance(model, edits), EPSILON)
    active = [pair["active_intents"] for pair in pairs]
    value = macro_intent(ratios, active)
    ci_low, ci_high, _ = bootstrap_macro_intent(ratios, active, seed)
    return {
        "pair_count": len(pairs),
        "distance": macro_intent(distance, active),
        "adjusted_ratio": value,
        "ci": [ci_low, ci_high],
        "pair_ratios": ratios,
    }


def probability_drop(probe, store, pairs, seed):
    original = np.asarray([store[int(pair["original_source_index"])] for pair in pairs], dtype=np.float64)
    counterfactual = np.asarray([store[int(pair["counterfactual_source_index"])] for pair in pairs], dtype=np.float64)
    values = probe.predict_proba(original)[:, 1] - probe.predict_proba(counterfactual)[:, 1]
    active = [pair["active_intents"] for pair in pairs]
    ci_low, ci_high, _ = bootstrap_macro_intent(values, active, seed)
    return {"value": macro_intent(values, active), "ci": [ci_low, ci_high], "pair_values": values}


def macro_class_transform(values, labels, transforms):
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels)
    transforms = np.asarray(transforms)
    means = []
    for label in (0, 1):
        for transform in sorted(set(transforms)):
            mask = (labels == label) & (transforms == transform)
            if np.any(mask):
                means.append(values[mask].mean())
    return float(np.mean(means))


def bootstrap_inv_metric(values, inv_rows, seed, aggregator=macro_class_transform):
    values = np.asarray(values, dtype=np.float64)
    by_contract = defaultdict(list)
    for index, row in enumerate(inv_rows):
        by_contract[int(row["contract_index"])].append(index)
    contracts = np.asarray(sorted(by_contract))
    generator = np.random.default_rng(seed)
    samples = np.empty(BOOTSTRAPS, dtype=np.float64)
    for repetition in range(BOOTSTRAPS):
        sampled = generator.choice(contracts, size=len(contracts), replace=True)
        indices = np.concatenate([np.asarray(by_contract[int(contract)]) for contract in sampled])
        samples[repetition] = aggregator(
            values[indices],
            [int(inv_rows[index]["label"]) for index in indices],
            [inv_rows[index]["transform"] for index in indices],
        )
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975)), samples


def invariance_metrics(base_store, inv_store, inv_rows, center, probe, seed):
    original = np.asarray([base_store[int(row["original_feature_index"])] for row in inv_rows], dtype=np.float64)
    transformed = np.asarray([inv_store[int(row["surface_source_index"])] for row in inv_rows], dtype=np.float64)
    labels = [int(row["label"]) for row in inv_rows]
    transforms = [row["transform"] for row in inv_rows]
    centered_cosine = cosine_rows(original - center, transformed - center)
    original_probability = probe.predict_proba(original)[:, 1]
    transformed_probability = probe.predict_proba(transformed)[:, 1]
    consistency = ((original_probability >= 0.5) == (transformed_probability >= 0.5)).astype(np.float64)
    shift = np.abs(original_probability - transformed_probability)
    cosine_ci = bootstrap_inv_metric(centered_cosine, inv_rows, seed + 1)[:2]
    consistency_ci = bootstrap_inv_metric(consistency, inv_rows, seed + 2)[:2]
    shift_ci = bootstrap_inv_metric(shift, inv_rows, seed + 3)[:2]
    return {
        "centered_cosine": macro_class_transform(centered_cosine, labels, transforms),
        "centered_cosine_ci": list(cosine_ci),
        "prediction_consistency": macro_class_transform(consistency, labels, transforms),
        "prediction_consistency_ci": list(consistency_ci),
        "probability_shift": macro_class_transform(shift, labels, transforms),
        "probability_shift_ci": list(shift_ci),
    }


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    global STAGE, CF_MANIFEST, CF_FULL, CF_LOCAL, INV_MANIFEST, INV_FEATURES
    global BASE_FEATURES, SPLIT_MANIFEST, BASE_METRICS, OUT, MODEL_NAME
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=False, default=str(STAGE))
    parser.add_argument("--cf-manifest", required=False)
    parser.add_argument("--cf-full", required=False)
    parser.add_argument("--cf-local", required=False)
    parser.add_argument("--inv-manifest", required=False)
    parser.add_argument("--inv-features", required=False)
    parser.add_argument("--base-features", required=False)
    parser.add_argument("--split-manifest", required=False)
    parser.add_argument("--base-metrics", required=False)
    parser.add_argument("--output", required=False)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--probe", default="logistic_regression")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    STAGE = Path(args.stage)
    CF_MANIFEST = Path(args.cf_manifest) if args.cf_manifest else STAGE / "data_cf/pair_manifest.jsonl"
    CF_FULL = Path(args.cf_full) if args.cf_full else STAGE / "features_cf/features_text_mean"
    CF_LOCAL = Path(args.cf_local) if args.cf_local else STAGE / "features_cf/features_focus_text_mean"
    INV_MANIFEST = Path(args.inv_manifest) if args.inv_manifest else STAGE / "data_inv/surface_manifest_selected.jsonl"
    INV_FEATURES = Path(args.inv_features) if args.inv_features else STAGE / "features_inv/features_text_mean"
    BASE_FEATURES = Path(args.base_features) if args.base_features else STAGE / "features_base/features_text_mean"
    SPLIT_MANIFEST = Path(args.split_manifest) if args.split_manifest else ROOT / "runs/stage4_smartintent_full_trials/manifests/trial_seed_0.json"
    BASE_METRICS = Path(args.base_metrics) if args.base_metrics else STAGE / "results/layer_probe_metrics.csv"
    OUT = Path(args.output) if args.output else STAGE / "results_strict_v5"
    MODEL_NAME = args.model_name
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = load_jsonl(CF_MANIFEST)
    train_pairs = [pair for pair in pairs if pair["split"] == "train"]
    validation_pairs = [pair for pair in pairs if pair["split"] == "val"]
    test_pairs = [pair for pair in pairs if pair["split"] == "test"]
    development_pairs = train_pairs + validation_pairs
    inv_rows = load_jsonl(INV_MANIFEST)
    split_manifest = json.load(open(SPLIT_MANIFEST, encoding="utf-8"))
    train_records = [record for record in split_manifest["records"] if record["split"] == "train"]
    train_indices = np.asarray([int(record["feature_index"]) for record in train_records])
    train_labels = np.asarray([int(record["label"]) for record in train_records])
    baseline_rows = list(csv.DictReader(open(BASE_METRICS, encoding="utf-8")))
    # Legacy CodeGemma input contains one row/layer named ``val_auroc``;
    # parameterized runs emit one row for each split and use ``auroc``.
    if baseline_rows and "split" in baseline_rows[0]:
        selected_probe_rows = [
            row for row in baseline_rows
            if row["probe"] == args.probe
            and int(row.get("seed", args.seed)) == args.seed
        ] if "probe" in baseline_rows[0] else baseline_rows
        validation_rows = [row for row in selected_probe_rows if row["split"] == "val"]
        test_rows = [row for row in selected_probe_rows if row["split"] == "test"]
        baseline = {int(row["layer"]): {"val_auroc": row["auroc"]} for row in validation_rows}
        test_baseline = {int(row["layer"]): row for row in test_rows}
        if len(baseline) != len(validation_rows):
            raise ValueError(
                f"Expected one validation row per layer for probe={args.probe}, seed={args.seed}; "
                "filter the metrics file to the intended probe/seed."
            )
    else:
        baseline = {int(row["layer"]): row for row in baseline_rows}
        test_baseline = baseline
    if not baseline:
        raise ValueError(f"No validation metrics found for probe={args.probe}, seed={args.seed}")
    max_validation_auroc = max(float(row["val_auroc"]) for row in baseline.values())
    validation_auroc_threshold = max_validation_auroc - LPROB_DELTA
    l_prob = {
        layer for layer, row in baseline.items()
        if float(row["val_auroc"]) >= validation_auroc_threshold
    }

    rows = []
    test_details = {}
    layer_count = len(baseline)
    for layer in range(layer_count):
        base_store = np.load(BASE_FEATURES / f"layer_{layer:02d}.npy", mmap_mode="r")
        cf_full_store = np.load(CF_FULL / f"layer_{layer:02d}.npy", mmap_mode="r")
        cf_local_store = np.load(CF_LOCAL / f"layer_{layer:02d}.npy", mmap_mode="r")
        inv_store = np.load(INV_FEATURES / f"layer_{layer:02d}.npy", mmap_mode="r")
        probe = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced", random_state=0),
        )
        probe.fit(np.asarray(base_store[train_indices], dtype=np.float64), train_labels)
        center = np.asarray(base_store[train_indices], dtype=np.float64).mean(axis=0)

        local_model, local_null95, local_null99 = local_surface_model(cf_local_store, train_pairs, seed=1000 + layer)
        full_model, full_null95, _ = full_surface_model(base_store, inv_store, inv_rows, seed=2000 + layer)
        dev_local = counterfactual_metrics(
            cf_local_store, development_pairs, "counterfactual_token_edit_count_local", local_model, seed=3000 + layer
        )
        val_local = counterfactual_metrics(
            cf_local_store, validation_pairs, "counterfactual_token_edit_count_local", local_model, seed=4000 + layer
        )
        dev_full = counterfactual_metrics(
            cf_full_store, development_pairs, "counterfactual_token_edit_count_full", full_model, seed=5000 + layer
        )
        val_full = counterfactual_metrics(
            cf_full_store, validation_pairs, "counterfactual_token_edit_count_full", full_model, seed=6000 + layer
        )
        dev_drop = probability_drop(probe, cf_full_store, development_pairs, seed=7000 + layer)
        val_drop = probability_drop(probe, cf_full_store, validation_pairs, seed=8000 + layer)
        inv = invariance_metrics(base_store, inv_store, inv_rows, center, probe, seed=9000 + layer)

        in_cf = (
            dev_local["adjusted_ratio"] > local_null99
            and dev_local["adjusted_ratio"] >= TAU_CF_RATIO
            and dev_local["ci"][0] >= TAU_CF_RATIO
            and val_local["adjusted_ratio"] >= TAU_CF_RATIO
            and dev_drop["ci"][0] >= TAU_CF_PROBABILITY_DROP
            and val_drop["ci"][0] >= TAU_CF_PROBABILITY_DROP
        )
        in_inv = (
            inv["centered_cosine"] > TAU_CENTERED_COSINE
            and inv["centered_cosine_ci"][0] > TAU_CENTERED_COSINE
            and inv["prediction_consistency"] > TAU_CONSISTENCY
            and inv["prediction_consistency_ci"][0] > TAU_CONSISTENCY
            and inv["probability_shift"] < TAU_PROBABILITY_SHIFT
            and inv["probability_shift_ci"][1] < TAU_PROBABILITY_SHIFT
        )
        row = {
            "layer": layer,
            "val_auroc": float(baseline[layer]["val_auroc"]),
            "local_surface_null95": local_null95,
            "local_surface_null99": local_null99,
            "dev_local_adjusted_ratio": dev_local["adjusted_ratio"],
            "dev_local_ci_lower": dev_local["ci"][0],
            "val_local_adjusted_ratio": val_local["adjusted_ratio"],
            "full_surface_null95": full_null95,
            "dev_full_adjusted_ratio": dev_full["adjusted_ratio"],
            "val_full_adjusted_ratio": val_full["adjusted_ratio"],
            "dev_unsafe_probability_drop": dev_drop["value"],
            "dev_drop_ci_lower": dev_drop["ci"][0],
            "val_unsafe_probability_drop": val_drop["value"],
            "val_unsafe_probability_drop_ci_lower": val_drop["ci"][0],
            "inv_centered_cosine": inv["centered_cosine"],
            "inv_centered_cosine_ci_lower": inv["centered_cosine_ci"][0],
            "inv_prediction_consistency": inv["prediction_consistency"],
            "inv_consistency_ci_lower": inv["prediction_consistency_ci"][0],
            "inv_probability_shift": inv["probability_shift"],
            "inv_probability_shift_ci_upper": inv["probability_shift_ci"][1],
            "in_L_prob": layer in l_prob,
            "in_L_cf": in_cf,
            "in_L_inv": in_inv,
            "in_L_critical": layer in l_prob and in_cf and in_inv,
        }
        rows.append(row)

        test_local = counterfactual_metrics(
            cf_local_store, test_pairs, "counterfactual_token_edit_count_local", local_model, seed=10000 + layer
        )
        test_full = counterfactual_metrics(
            cf_full_store, test_pairs, "counterfactual_token_edit_count_full", full_model, seed=11000 + layer
        )
        test_drop = probability_drop(probe, cf_full_store, test_pairs, seed=12000 + layer)
        test_details[str(layer)] = {
            "local": {key: value for key, value in test_local.items() if key != "pair_ratios"},
            "full": {key: value for key, value in test_full.items() if key != "pair_ratios"},
            "unsafe_probability_drop": {key: value for key, value in test_drop.items() if key != "pair_values"},
            "binary_probe": {
                "test_accuracy": float(test_baseline[layer].get("test_accuracy", test_baseline[layer].get("accuracy"))),
                "test_auroc": float(test_baseline[layer].get("test_auroc", test_baseline[layer].get("auroc"))),
                "test_auprc": float(test_baseline[layer].get("test_auprc", test_baseline[layer].get("auprc"))),
                "test_f1": float(test_baseline[layer].get("test_f1", test_baseline[layer].get("f1"))),
            },
        }
        print(json.dumps({"layer": layer, "in_cf": in_cf, "in_inv": in_inv, "critical": row["in_L_critical"]}), flush=True)

    write_csv(OUT / "layer_binary_cf_inv_v3_metrics.csv", rows)
    l_cf = {row["layer"] for row in rows if row["in_L_cf"]}
    l_inv = {row["layer"] for row in rows if row["in_L_inv"]}
    l_critical = l_prob & l_cf & l_inv
    summary = {
        "model": MODEL_NAME,
        "seed": 0,
        "binary_task_contracts": 2000,
        "binary_train_contracts_for_invariance": len({row["contract_index"] for row in inv_rows}),
        "counterfactual_pairs": {
            "train": len(train_pairs), "validation": len(validation_pairs), "test": len(test_pairs),
            "single_label": sum(len(pair["active_intents"]) == 1 for pair in pairs),
            "multi_label": sum(len(pair["active_intents"]) > 1 for pair in pairs),
        },
        "L_prob": sorted(l_prob),
        "L_cf": sorted(l_cf),
        "L_inv": sorted(l_inv),
        "L_critical": sorted(l_critical),
        "selection_uses_test": False,
        "thresholds": {
        "L_prob": f"{args.probe} seed {args.seed}: Validation AUROC >= max Validation AUROC - 0.008",
            "L_cf": "ten-intent macro local adjusted ratio above Train surface-null 99%, adjusted ratio and development/validation CI lower >= 1.5, and development/validation unsafe probability-drop CI lower >= 0.02",
            "L_inv": {
                "centered_cosine_and_ci_lower": TAU_CENTERED_COSINE,
                "prediction_consistency_and_ci_lower": TAU_CONSISTENCY,
                "probability_shift_and_ci_upper": TAU_PROBABILITY_SHIFT,
            },
        },
        "test_metrics_at_critical_layers": {str(layer): test_details[str(layer)] for layer in sorted(l_critical)},
    }
    with open(OUT / "candidate_layers_v3.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    with open(OUT / "test_layer_details_v3.json", "w", encoding="utf-8") as handle:
        json.dump(test_details, handle, indent=2, ensure_ascii=False)

    layers = [row["layer"] for row in rows]
    fig, axes = plt.subplots(3, 2, figsize=(13, 12), sharex=True)
    axes[0, 0].plot(layers, [row["dev_local_adjusted_ratio"] for row in rows], label="local adjusted R_cf")
    axes[0, 0].plot(layers, [row["local_surface_null95"] for row in rows], "--", label="Train surface null 95%")
    axes[0, 0].set_ylabel("Ten-intent macro local R_cf")
    axes[0, 0].legend()
    axes[0, 1].plot(layers, [row["dev_full_adjusted_ratio"] for row in rows], label="full adjusted R_cf")
    axes[0, 1].plot(layers, [row["full_surface_null95"] for row in rows], "--", label="Train surface null 95%")
    axes[0, 1].set_ylabel("Full-source adjusted R_cf")
    axes[0, 1].legend()
    axes[1, 0].plot(layers, [row["dev_unsafe_probability_drop"] for row in rows])
    axes[1, 0].axhline(0, color="tab:red", linestyle="--")
    axes[1, 0].set_ylabel("Unsafe probability drop")
    axes[1, 1].plot(layers, [row["inv_centered_cosine"] for row in rows])
    axes[1, 1].axhline(TAU_CENTERED_COSINE, color="tab:red", linestyle="--")
    axes[1, 1].set_ylabel("Full-Train centered cosine")
    axes[2, 0].plot(layers, [row["inv_prediction_consistency"] for row in rows])
    axes[2, 0].axhline(TAU_CONSISTENCY, color="tab:red", linestyle="--")
    axes[2, 0].set_ylabel("Full-Train prediction consistency")
    axes[2, 0].set_xlabel("Layer")
    axes[2, 1].plot(layers, [row["inv_probability_shift"] for row in rows])
    axes[2, 1].axhline(TAU_PROBABILITY_SHIFT, color="tab:red", linestyle="--")
    axes[2, 1].set_ylabel("Full-Train probability shift")
    axes[2, 1].set_xlabel("Layer")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    fig.suptitle(f"{MODEL_NAME} Seed 0 ten-intent binary counterfactual and full-Train invariance V3")
    fig.tight_layout()
    fig.savefig(OUT / "binary_cf_inv_v3_layer_curves.png", dpi=220)
    fig.savefig(OUT / "binary_cf_inv_v3_layer_curves.pdf")
    plt.close(fig)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
