#!/usr/bin/env python3
"""Extract frozen token-exact mean representations for one or more text fields."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def load_rows(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def windows(tokenizer, text, max_length, overlap, field, row_index):
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
            "field": field,
            "row_index": row_index,
            "ids": prefix + content + suffix,
            "content_offset": len(prefix),
            "content_length": len(content),
            "unique_offset": 0 if start == 0 else min(overlap, len(content)),
        })
        if start + budget >= len(code_ids):
            break
    return len(code_ids), chunks


def open_stores(directory, layer_count, shape, resume, dtype):
    directory.mkdir(parents=True, exist_ok=True)
    files = [directory / f"layer_{layer:02d}.npy" for layer in range(layer_count)]
    if resume and all(path.exists() for path in files):
        stores = [np.lib.format.open_memmap(path, mode="r+") for path in files]
        if any(store.shape != shape or store.dtype != dtype for store in stores):
            raise RuntimeError(f"Unexpected feature store metadata in {directory}")
        return stores
    if resume and any(path.exists() for path in files):
        raise RuntimeError(f"Incomplete resumable stores in {directory}")
    return [
        np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
        for path in files
    ]


def completed_rows(stores):
    values = []
    for store in stores:
        nonzero = np.any(store != 0, axis=1)
        zero = np.flatnonzero(~nonzero)
        prefix = int(zero[0]) if len(zero) else len(nonzero)
        if np.any(nonzero[prefix:]):
            raise RuntimeError("Nonzero rows found after checkpoint boundary")
        values.append(prefix)
    if len(set(values)) != 1:
        raise RuntimeError(f"Inconsistent checkpoints: {values}")
    return values[0]


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fields", nargs="+", default=["text"])
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--overlap", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--row-group-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--storage-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    rows = load_rows(args.data_path)
    output = Path(args.output_dir)
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
    model.config.output_hidden_states = True
    model.config.use_cache = False
    model.to(args.device)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable or model.training:
        raise RuntimeError("Backbone freeze verification failed")

    layer_count = int(model.config.num_hidden_layers) + 1
    hidden_size = int(model.config.hidden_size)
    storage_dtype = np.float16 if args.storage_dtype == "float16" else np.float32
    stores = {
        field: open_stores(
            output / f"features_{field}_mean",
            layer_count,
            (len(rows), hidden_size),
            args.resume,
            storage_dtype,
        )
        for field in args.fields
    }
    checkpoints = {field: completed_rows(field_stores) for field, field_stores in stores.items()}
    if len(set(checkpoints.values())) != 1:
        raise RuntimeError(f"Field checkpoints differ: {checkpoints}")
    completed = next(iter(checkpoints.values()))
    freeze = {
        "passed": True,
        "trainable_parameters": trainable,
        "training_mode": model.training,
        "model_dtype": str(next(model.parameters()).dtype),
        "storage_dtype": str(np.dtype(storage_dtype)),
    }
    print(json.dumps({"freeze_check": freeze, "resume_from": completed, "total": len(rows)}), flush=True)

    processed_windows = 0
    with torch.inference_mode():
        for group_start in range(completed, len(rows), args.row_group_size):
            group_end = min(group_start + args.row_group_size, len(rows))
            sums = {
                row_index: {
                    field: np.zeros((layer_count, hidden_size), dtype=np.float64)
                    for field in args.fields
                }
                for row_index in range(group_start, group_end)
            }
            counts = {row_index: {field: 0 for field in args.fields} for row_index in range(group_start, group_end)}
            expected = {row_index: {} for row_index in range(group_start, group_end)}
            chunks = []
            for row_index in range(group_start, group_end):
                for field in args.fields:
                    expected[row_index][field], field_chunks = windows(
                        tokenizer,
                        str(rows[row_index][field]),
                        args.max_length,
                        args.overlap,
                        field,
                        row_index,
                    )
                    chunks.extend(field_chunks)
            for batch_start in range(0, len(chunks), args.batch_size):
                batch = chunks[batch_start:batch_start + args.batch_size]
                width = max(len(chunk["ids"]) for chunk in batch)
                input_ids = torch.full(
                    (len(batch), width), tokenizer.pad_token_id, dtype=torch.long, device=args.device
                )
                attention_mask = torch.zeros_like(input_ids)
                for batch_index, chunk in enumerate(batch):
                    ids = torch.tensor(chunk["ids"], dtype=torch.long, device=args.device)
                    input_ids[batch_index, :len(ids)] = ids
                    attention_mask[batch_index, :len(ids)] = 1
                hidden_states = model(input_ids=input_ids, attention_mask=attention_mask).hidden_states
                for batch_index, chunk in enumerate(batch):
                    begin = chunk["content_offset"] + chunk["unique_offset"]
                    end = chunk["content_offset"] + chunk["content_length"]
                    if end <= begin:
                        continue
                    row_index = chunk["row_index"]
                    field = chunk["field"]
                    counts[row_index][field] += end - begin
                    for layer, hidden in enumerate(hidden_states):
                        sums[row_index][field][layer] += hidden[batch_index, begin:end].float().sum(0).cpu().numpy()
            for row_index in range(group_start, group_end):
                for field in args.fields:
                    if counts[row_index][field] != expected[row_index][field]:
                        raise RuntimeError(
                            f"Token accounting failed row={row_index} field={field}: "
                            f"{counts[row_index][field]} != {expected[row_index][field]}"
                        )
                    for layer in range(layer_count):
                        stores[field][layer][row_index] = (
                            sums[row_index][field][layer] / counts[row_index][field]
                        ).astype(storage_dtype)
            processed_windows += len(chunks)
            if (group_end - completed) % 20 == 0 or group_end == len(rows):
                for field_stores in stores.values():
                    for store in field_stores:
                        store.flush()
                atomic_json(output / "extraction_checkpoint.json", {
                    "completed_rows": group_end,
                    "total_rows": len(rows),
                    "windows_this_process": processed_windows,
                })
                print(json.dumps({"processed": group_end, "total": len(rows), "windows": processed_windows}), flush=True)

    atomic_json(output / "run_metadata.json", {
        "model_path": args.model_path,
        "data_path": args.data_path,
        "rows": len(rows),
        "fields": args.fields,
        "max_length": args.max_length,
        "overlap": args.overlap,
        "pooling": "mean over every global field token exactly once; overlap supplies context only",
        "freeze_check": freeze,
    })


if __name__ == "__main__":
    main()
