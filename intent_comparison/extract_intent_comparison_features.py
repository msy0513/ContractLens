#!/usr/bin/env python3
"""Extract shared frozen CodeGemma features for all ten intent tasks."""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from intent_comparison_common import (
    CRITICAL_LAYERS, MODEL, OUT, atomic_json, load_jsonl,
    load_stable_units, require_disk, seed_all,
)


MAX_LENGTH = 2048
SEGMENTS = 256


def compress_segments(hidden: torch.Tensor, mask: torch.Tensor):
    batch, length, dim = hidden.shape
    if batch != 1 or length != MAX_LENGTH:
        raise RuntimeError(f"Unexpected hidden shape: {tuple(hidden.shape)}")
    hidden = hidden.reshape(batch, SEGMENTS, MAX_LENGTH // SEGMENTS, dim)
    weights = mask.reshape(batch, SEGMENTS, MAX_LENGTH // SEGMENTS).to(hidden.dtype)
    pooled = (hidden * weights.unsqueeze(-1)).sum(2) / weights.sum(2).clamp_min(1).unsqueeze(-1)
    segment_mask = weights.sum(2) > 0
    return pooled, segment_mask


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(
        MODEL, local_files_only=True, torch_dtype=torch.float16, low_cpu_mem_usage=True,
    )
    model.requires_grad_(False)
    model.eval()
    model.config.use_cache = False
    if any(parameter.requires_grad for parameter in model.parameters()) or model.training:
        raise RuntimeError("CodeGemma freeze check failed")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device


def register_hooks(model, units_by_layer):
    layers = getattr(model, "layers", None)
    if layers is None:
        raise RuntimeError("Expected CodeGemma AutoModel to expose model.layers")
    cache, hooks = {}, []
    for layer in CRITICAL_LAYERS:
        block_index = layer - 1
        unit_index = torch.as_tensor(units_by_layer[layer], dtype=torch.long, device=next(model.parameters()).device)
        def make_hook(layer_id, selected):
            def hook(_module, inputs):
                cache[layer_id] = inputs[0].detach().index_select(-1, selected)
            return hook
        hooks.append(layers[block_index].mlp.down_proj.register_forward_pre_hook(make_hook(layer, unit_index)))
    return cache, hooks


def forward_row(row, tokenizer, model, device, activation_cache, units_by_layer):
    encoded = tokenizer(
        str(row["text"]), truncation=True, padding="max_length",
        max_length=MAX_LENGTH, return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    activation_cache.clear()
    with torch.inference_mode():
        output = model(
            input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"],
            output_hidden_states=True, return_dict=True,
        )
    mask = encoded["attention_mask"].to(torch.float32)
    denominator = mask.sum().clamp_min(1)
    means = []
    for hidden in output.hidden_states:
        mean = (hidden[0].float() * mask[0].unsqueeze(-1)).sum(0) / denominator
        means.append(mean.cpu().numpy())
    segments, segment_mask = compress_segments(output.last_hidden_state, encoded["attention_mask"])
    selected = []
    for layer in CRITICAL_LAYERS:
        if layer not in activation_cache:
            raise RuntimeError(f"Missing MLP activation for Layer {layer}")
        activation = activation_cache[layer][0].float()
        pooled = (activation * mask[0].unsqueeze(-1)).sum(0) / denominator
        if pooled.numel() != len(units_by_layer[layer]):
            raise RuntimeError(f"Stable-unit shape mismatch at Layer {layer}")
        selected.append(pooled.cpu().numpy())
    return (np.stack(means), segments[0].float().cpu().numpy(),
            segment_mask[0].cpu().numpy(), np.concatenate(selected))


def smoke_test(rows):
    tokenizer, model, device = load_model()
    units = load_stable_units()
    cache, hooks = register_hooks(model, units)
    try:
        means, segments, mask, selected = forward_row(rows[0], tokenizer, model, device, cache, units)
        print(json.dumps({"smoke": "passed", "layer_means": list(means.shape),
                          "segments": list(segments.shape), "segment_mask": list(mask.shape),
                          "stable_units": list(selected.shape), "trainable_parameters": 0}, indent=2))
    finally:
        for hook in hooks:
            hook.remove()


def extract(rows, resume: bool):
    require_disk(5.0)
    feature_dir = OUT / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    units_by_layer = load_stable_units()
    unit_offsets, offset = {}, 0
    for layer in CRITICAL_LAYERS:
        unit_offsets[str(layer)] = {"start": offset, "stop": offset + len(units_by_layer[layer]),
                                    "count": len(units_by_layer[layer])}
        offset += len(units_by_layer[layer])
    n = len(rows)
    means_path = feature_dir / "all_layer_means.f16.npy"
    segments_path = feature_dir / "last_layer_segments.f16.npy"
    mask_path = feature_dir / "segment_mask.bool.npy"
    units_path = feature_dir / "stable_mlp_units.f16.npy"
    checkpoint_path = feature_dir / "extraction_checkpoint.json"
    completed = 0
    expected = {
        means_path: (n, 29, 3072), segments_path: (n, SEGMENTS, 3072),
        mask_path: (n, SEGMENTS), units_path: (n, offset),
    }
    if resume and checkpoint_path.exists() and all(path.exists() for path in expected):
        completed = int(json.loads(checkpoint_path.read_text(encoding="utf-8"))["completed_rows"])
        stores = {path: np.lib.format.open_memmap(path, mode="r+") for path in expected}
        for path, shape in expected.items():
            if stores[path].shape != shape:
                raise RuntimeError(f"Feature shape mismatch for {path}: {stores[path].shape} != {shape}")
    else:
        stores = {
            means_path: np.lib.format.open_memmap(means_path, mode="w+", dtype=np.float16, shape=expected[means_path]),
            segments_path: np.lib.format.open_memmap(segments_path, mode="w+", dtype=np.float16, shape=expected[segments_path]),
            mask_path: np.lib.format.open_memmap(mask_path, mode="w+", dtype=np.bool_, shape=expected[mask_path]),
            units_path: np.lib.format.open_memmap(units_path, mode="w+", dtype=np.float16, shape=expected[units_path]),
        }
    tokenizer, model, device = load_model()
    activation_cache, hooks = register_hooks(model, units_by_layer)
    print(json.dumps({"stage": "feature_extraction", "rows": n, "resume_from": completed,
                      "stable_unit_count": offset, "device": str(device), "backbone_frozen": True}), flush=True)
    try:
        for index in range(completed, n):
            means, segments, mask, selected = forward_row(
                rows[index], tokenizer, model, device, activation_cache, units_by_layer,
            )
            stores[means_path][index] = means.astype(np.float16)
            stores[segments_path][index] = segments.astype(np.float16)
            stores[mask_path][index] = mask
            stores[units_path][index] = selected.astype(np.float16)
            if (index + 1) % 10 == 0 or index + 1 == n:
                for store in stores.values():
                    store.flush()
                atomic_json(checkpoint_path, {"completed_rows": index + 1, "total_rows": n})
                print(json.dumps({"stage": "feature_extraction", "processed": index + 1, "total": n}), flush=True)
            if (index + 1) % 100 == 0:
                require_disk(5.0)
    finally:
        for hook in hooks:
            hook.remove()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    atomic_json(feature_dir / "metadata.json", {
        "rows": n, "layer_mean_shape": list(expected[means_path]),
        "segment_shape": list(expected[segments_path]), "max_length": MAX_LENGTH,
        "segments": SEGMENTS, "stable_unit_shape": list(expected[units_path]),
        "stable_unit_offsets": unit_offsets, "stable_units": {str(k): v for k, v in units_by_layer.items()},
        "backbone_frozen": True, "dtype": "float16",
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    seed_all(0)
    rows = load_jsonl(OUT / "selected_union.jsonl")
    if args.smoke:
        smoke_test(rows)
    else:
        extract(rows, args.resume)


if __name__ == "__main__":
    main()
