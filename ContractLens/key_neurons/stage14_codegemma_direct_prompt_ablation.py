#!/usr/bin/env python3
"""Direct-prompt CodeGemma unit ablation for unsafe-contract detection.

The baseline is a zero-shot YES/NO prompt, not a learned probe.  The same
prompt and next-token score are used for all conditions; only selected MLP
units are zeroed during the frozen CodeGemma forward pass.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "base/CodeGemma-7B"
UNION = ROOT / "runs/stage3_smartintent_sliding/sliding_union.jsonl"
BINARY = ROOT / "runs/stage3_smartintent_sliding/sliding_tasks/binary.jsonl"
MANIFEST = ROOT / "runs/stage4_smartintent_full_trials/manifests/trial_seed_0.json"
STABLE = ROOT / "runs/stage10_codegemma_seed0_stable_units"
FEATURES = ROOT / "runs/stage13_codegemma_seed0_neuron_specificity_fp16/features"
LAYERS = (16, 18, 19, 21)
MLP_WIDTH = 24576

PROMPT_PREFIX = (
    "Contract:\n"
)
PROMPT_SUFFIX = (
    "\n\nYou are analyzing Solidity source code for a security benchmark.\n"
    "Question: Does this contract contain any unsafe intent behavior?\n"
    "Answer the question with exactly one token: YES or NO.\n"
    "Answer:"
)


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def prompt(code: str) -> str:
    return PROMPT_PREFIX + code + PROMPT_SUFFIX


def sigmoid_delta(delta):
    delta = np.asarray(delta, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(delta, -60.0, 60.0)))


def metrics(y, probability, threshold=0.5):
    probability = np.asarray(probability, dtype=np.float64)
    return {
        "accuracy": float(accuracy_score(y, probability >= threshold)),
        "auroc": float(roc_auc_score(y, probability)),
    }


def load_split(binary_rows, manifest):
    pos = {int(row["feature_index"]): i for i, row in enumerate(binary_rows)}
    split, groups = {s: [] for s in ("train", "val", "test")}, {s: [] for s in ("train", "val", "test")}
    for record in manifest["records"]:
        i = pos[int(record["feature_index"])]
        split[record["split"]].append(i)
        groups[record["split"]].append(str(record["group"]))
    return ({s: np.asarray(split[s], dtype=np.int64) for s in split},
            {s: np.asarray(groups[s], dtype=object) for s in groups})


def load_stable(path: Path):
    result = {}
    for layer in LAYERS:
        with (path / f"layer_{layer}" / "stability_units.csv").open(encoding="utf-8") as f:
            result[layer] = np.asarray(
                [int(row["unit"]) for row in csv.DictReader(f) if row["stable"].lower() == "true"],
                dtype=np.int64,
            )
    return result


def choose_hard_random(stable, available, train_arrays):
    """One-to-one nearest matching of Train mean absolute activation."""
    selected, diagnostics = {}, {}
    for layer in LAYERS:
        key = np.asarray(stable[layer], dtype=np.int64)
        units = np.asarray(available[layer], dtype=np.int64)
        nonkey = np.setdiff1d(units, key, assume_unique=True)
        key_pos = np.searchsorted(units, key)
        non_pos = np.searchsorted(units, nonkey)
        key_mag = np.mean(np.abs(train_arrays[layer][:, key_pos]), axis=0)
        non_mag = np.mean(np.abs(train_arrays[layer][:, non_pos]), axis=0)
        order = np.argsort(non_mag, kind="stable")
        sorted_mag = non_mag[order]
        used = np.zeros(len(nonkey), dtype=bool)
        positions = []
        for target in key_mag[np.argsort(key_mag, kind="stable")]:
            center = int(np.searchsorted(sorted_mag, target, side="left"))
            best = None
            radius = 0
            while best is None:
                for candidate in (center - radius, center + radius):
                    if 0 <= candidate < len(sorted_mag) and not used[candidate]:
                        if best is None or abs(sorted_mag[candidate] - target) < abs(sorted_mag[best] - target):
                            best = candidate
                radius += 1
            used[best] = True
            positions.append(best)
        decoy = np.asarray(sorted(nonkey[order[np.asarray(positions, dtype=np.int64)]]), dtype=np.int64)
        selected[layer] = decoy
        decoy_mag = np.mean(np.abs(train_arrays[layer][:, np.searchsorted(units, decoy)]), axis=0)
        diagnostics[str(layer)] = {
            "key_count": int(len(key)),
            "decoy_count": int(len(decoy)),
            "key_mean_abs_activation": float(np.mean(key_mag)),
            "decoy_mean_abs_activation": float(np.mean(decoy_mag)),
            "absolute_mean_difference": float(abs(np.mean(key_mag) - np.mean(decoy_mag))),
        }
    return selected, diagnostics


def group_bootstrap(y, probability, groups, repeats, seed, threshold=0.5):
    by_group = defaultdict(list)
    for i, group in enumerate(groups):
        by_group[str(group)].append(i)
    names = np.asarray(list(by_group), dtype=object)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        sampled = rng.choice(names, size=len(names), replace=True)
        indices = np.asarray([i for group in sampled for i in by_group[str(group)]], dtype=np.int64)
        if np.unique(y[indices]).size < 2:
            continue
        values.append(metrics(y[indices], probability[indices], threshold))
    acc = np.asarray([x["accuracy"] for x in values], dtype=np.float64)
    auc = np.asarray([x["auroc"] for x in values], dtype=np.float64)
    return {
        "repeats": int(len(values)),
        "accuracy": {"mean": float(acc.mean()), "lower": float(np.percentile(acc, 2.5)),
                      "upper": float(np.percentile(acc, 97.5))},
        "auroc": {"mean": float(auc.mean()), "lower": float(np.percentile(auc, 2.5)),
                  "upper": float(np.percentile(auc, 97.5))},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(MODEL))
    parser.add_argument("--union", default=str(UNION))
    parser.add_argument("--binary", default=str(BINARY))
    parser.add_argument("--manifest", default=str(MANIFEST))
    parser.add_argument("--stable", default=str(STABLE))
    parser.add_argument("--features", default=str(FEATURES))
    parser.add_argument("--output", default=str(ROOT / "runs/stage14_codegemma_seed0_direct_prompt_ablation"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--random-repeats", type=int, default=30)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    binary = load_jsonl(Path(args.binary))
    union = load_jsonl(Path(args.union))
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    split, groups = load_split(binary, manifest)
    labels = np.asarray([int(row["label"]) for row in binary], dtype=np.int64)
    stable = load_stable(Path(args.stable))

    # The cached Stage 13 columns are the available population for both
    # count-matched random controls and activation-matched decoys.
    available, train_arrays = {}, {}
    for layer in LAYERS:
        available[layer] = np.load(Path(args.features) / f"layer_{layer}_global_units.npy")
        train_arrays[layer] = np.asarray(
            np.load(Path(args.features) / f"layer_{layer}_selected_union.f16.npy", mmap_mode="r")[split["train"]],
            dtype=np.float32,
        )
        if not np.all(np.isin(stable[layer], available[layer])):
            raise RuntimeError(f"Stable units missing from cached columns at layer {layer}")
    hard, hard_diag = choose_hard_random(stable, available, train_arrays)

    rng = np.random.default_rng(args.seed + 14000)
    random_sets = []
    for _ in range(args.random_repeats):
        record = {}
        for layer in LAYERS:
            nonkey = np.setdiff1d(available[layer], stable[layer], assume_unique=True)
            record[layer] = np.asarray(
                np.sort(rng.choice(nonkey, size=len(stable[layer]), replace=False)), dtype=np.int64
            )
        random_sets.append(record)

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    yes = tokenizer.encode(" YES", add_special_tokens=False)
    no = tokenizer.encode(" NO", add_special_tokens=False)
    if len(yes) != 1 or len(no) != 1:
        raise RuntimeError(f"YES/NO must be single tokens, got {yes}, {no}")
    yes_id, no_id = yes[0], no[0]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, torch_dtype=dtype, low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    model.requires_grad_(False)
    model.config.use_cache = False
    if any(parameter.requires_grad for parameter in model.parameters()) or model.training:
        raise RuntimeError("CodeGemma freeze audit failed")

    state = {layer: torch.empty(0, dtype=torch.long, device=device) for layer in LAYERS}
    hooks = []
    for layer in LAYERS:
        def make_hook(layer_id):
            def hook(_module, inputs):
                x = inputs[0]
                units = state[layer_id]
                if units.numel() == 0:
                    return (x,)
                z = x.clone()
                z.index_fill_(-1, units, 0.0)
                return (z,)
            return hook
        hooks.append(model.model.layers[layer - 1].mlp.down_proj.register_forward_pre_hook(make_hook(layer)))

    test_indices = split["test"]
    test_rows = [union[int(binary[i]["feature_index"])] for i in test_indices]
    y_test = labels[test_indices]

    def score(mask):
        for layer in LAYERS:
            state[layer] = torch.as_tensor(mask.get(layer, []), dtype=torch.long, device=device)
        scores = []
        with torch.inference_mode():
            for start in range(0, len(test_rows), args.batch_size):
                batch_rows = test_rows[start:start + args.batch_size]
                encoded = tokenizer(
                    [prompt(str(row["text"])) for row in batch_rows], return_tensors="pt",
                    padding=True, truncation=True, truncation_side="left", max_length=args.max_length,
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                output = model(**encoded)
                last = encoded["attention_mask"].sum(dim=1) - 1
                row_ids = torch.arange(len(batch_rows), device=device)
                logits = output.logits[row_ids, last]
                delta = (logits[:, yes_id] - logits[:, no_id]).float().cpu().numpy()
                scores.extend(delta.tolist())
                if min(start + len(batch_rows), len(test_rows)) % 25 == 0:
                    print(json.dumps({"phase": "direct_prompt", "processed": min(start + len(batch_rows), len(test_rows)),
                                      "total": len(test_rows)}), flush=True)
        return sigmoid_delta(np.asarray(scores, dtype=np.float64))

    probabilities = {"base": score({}), "key": score(stable), "hard_random": score(hard)}
    probabilities["random"] = np.asarray([score(selection) for selection in random_sets])

    base = metrics(y_test, probabilities["base"])
    random_metrics = [metrics(y_test, p) for p in probabilities["random"]]
    table = [
        ("Base CodeGemma Direct Prompt", base),
        ("+ Masked Random Units (mean of 30)", {
            "accuracy": float(np.mean([x["accuracy"] for x in random_metrics])),
            "auroc": float(np.mean([x["auroc"] for x in random_metrics])),
        }),
        ("+ Masked Magnitude-Matched Units", metrics(y_test, probabilities["hard_random"])),
        ("+ Masked Key Units", metrics(y_test, probabilities["key"])),
    ]

    def display(value, baseline):
        return f"{100*value:.1f} ({100*(value-baseline):+.1f})"

    table_rows = [{"method": name, "accuracy": values["accuracy"], "auroc": values["auroc"],
                   "display_accuracy": display(values["accuracy"], base["accuracy"]),
                   "display_auroc": display(values["auroc"], base["auroc"]),
                   "test_n": int(len(y_test))}
                  for name, values in table]

    boot = {
        "base": group_bootstrap(y_test, probabilities["base"], groups["test"], args.bootstrap_repeats, args.seed + 1),
        "key": group_bootstrap(y_test, probabilities["key"], groups["test"], args.bootstrap_repeats, args.seed + 2),
        "hard_random": group_bootstrap(y_test, probabilities["hard_random"], groups["test"], args.bootstrap_repeats, args.seed + 3),
    }
    random_boot = [group_bootstrap(y_test, p, groups["test"], max(100, args.bootstrap_repeats // 5),
                                   args.seed + 100 + i) for i, p in enumerate(probabilities["random"])]
    for metric_name in ("accuracy", "auroc"):
        means = np.asarray([x[metric_name]["mean"] for x in random_boot], dtype=np.float64)
        boot["random"] = boot.get("random", {})
        boot["random"][metric_name] = {
            "mean": float(means.mean()), "lower": float(np.percentile(means, 2.5)),
            "upper": float(np.percentile(means, 97.5)), "repeats": int(len(means)),
        }

    np.savez_compressed(out / "test_probabilities.npz", base=probabilities["base"],
                        key=probabilities["key"], hard_random=probabilities["hard_random"],
                        random=probabilities["random"], label=y_test)
    atomic_json(out / "config.json", {
        "seed": args.seed, "model": str(args.model), "backbone_frozen": True,
        "baseline": "direct prompt; no classifier trained",
        "prompt": "fixed binary unsafe-intent question; answer exactly YES or NO",
        "score": "sigmoid(logit(' YES') - logit(' NO'))", "accuracy_threshold": 0.5,
        "max_length": args.max_length, "truncation_side": "left",
        "layers": list(LAYERS), "unit_type": "post-gating MLP units at down_proj input",
        "intervention": "activation zeroing during CodeGemma forward pass",
        "random_repeats": args.random_repeats, "bootstrap_repeats": args.bootstrap_repeats,
        "batch_size": args.batch_size,
        "test_used_for_selection": False,
        "hard_random_population": "Stage 13 cached union columns",
    })
    atomic_json(out / "audit.json", {
        "split_counts": {s: int(len(split[s])) for s in split},
        "split_label_counts": {s: {"0": int((labels[split[s]] == 0).sum()),
                                    "1": int((labels[split[s]] == 1).sum())} for s in split},
        "group_overlap": {"train_val": sorted(set(groups["train"]) & set(groups["val"])),
                          "train_test": sorted(set(groups["train"]) & set(groups["test"])),
                          "val_test": sorted(set(groups["val"]) & set(groups["test"]))},
        "stable_counts": {str(l): int(len(stable[l])) for l in LAYERS},
        "hard_random_diagnostics": hard_diag,
        "random_repeats": args.random_repeats,
        "yes_token_id": int(yes_id), "no_token_id": int(no_id),
        "test_used_for_selection": False,
    })
    atomic_json(out / "bootstrap_cis.json", boot)
    atomic_json(out / "table3_style.json", {
        "baseline_test": base, "rows": table_rows,
        "metric_note": "percentages; parenthesized values are changes from direct-prompt baseline",
    })
    with (out / "table3_style.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Method", "ACC (↑)", "AUROC (↑)"])
        writer.writeheader()
        for row in table_rows:
            writer.writerow({"Method": row["method"], "ACC (↑)": row["display_accuracy"],
                             "AUROC (↑)": row["display_auroc"]})
    print(json.dumps({"status": "complete", "output": str(out), "table": table_rows}, indent=2), flush=True)


if __name__ == "__main__":
    main()
