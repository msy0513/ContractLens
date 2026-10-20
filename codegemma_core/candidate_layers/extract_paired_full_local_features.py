#!/usr/bin/env python3
"""Extract frozen full-contract and edit-local CodeGemma representations."""

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


def source_windows(tokenizer, text, max_length, overlap, field):
    code_ids = tokenizer.encode(text, add_special_tokens=False, verbose=False)
    if not code_ids:
        code_ids = [tokenizer.unk_token_id or tokenizer.eos_token_id]
    prefix = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    suffix = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    budget = max_length - len(prefix) - len(suffix)
    step = budget - overlap
    if step <= 0:
        raise ValueError("overlap must be smaller than code-token budget")
    chunks = []
    for start in range(0, len(code_ids), step):
        content = code_ids[start:start + budget]
        chunks.append({
            "field": field,
            "ids": prefix + content + suffix,
            "content_offset": len(prefix),
            "content_length": len(content),
            "unique_offset": 0 if start == 0 else min(overlap, len(content)),
        })
        if start + budget >= len(code_ids):
            break
    return code_ids, chunks


def open_stores(directory, layer_count, shape, resume):
    directory.mkdir(parents=True, exist_ok=True)
    existing = all((directory / f"layer_{layer:02d}.npy").exists() for layer in range(layer_count))
    if resume and existing:
        stores = [np.lib.format.open_memmap(directory / f"layer_{layer:02d}.npy", mode="r+") for layer in range(layer_count)]
        if any(store.shape != shape for store in stores):
            raise RuntimeError(f"Unexpected feature shape in {directory}")
        return stores
    if resume and any(directory.glob("layer_*.npy")):
        raise RuntimeError(f"Incomplete resumable feature set in {directory}")
    return [
        np.lib.format.open_memmap(
            directory / f"layer_{layer:02d}.npy", mode="w+", dtype="float32", shape=shape
        ) for layer in range(layer_count)
    ]


def completed_rows(stores):
    values = []
    for store in stores:
        nonzero = np.any(store != 0, axis=1)
        zero = np.flatnonzero(~nonzero)
        prefix = int(zero[0]) if len(zero) else len(nonzero)
        if np.any(nonzero[prefix:]):
            raise RuntimeError("Nonzero feature rows found after checkpoint")
        values.append(prefix)
    if len(set(values)) != 1:
        raise RuntimeError(f"Inconsistent checkpoints: {values}")
    return values[0]


def write_json_atomic(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--overlap", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    rows = load_rows(args.data_path)
    output = Path(args.output_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(
        args.model_path, local_files_only=True, torch_dtype=torch.float16, low_cpu_mem_usage=True,
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
    shape = (len(rows), hidden_size)
    full_stores = open_stores(output / "features_full_mean", layer_count, shape, args.resume)
    local_stores = open_stores(output / "features_local_mean", layer_count, shape, args.resume)
    full_completed = completed_rows(full_stores)
    local_completed = completed_rows(local_stores)
    if full_completed != local_completed:
        raise RuntimeError(f"Full/local checkpoints differ: {full_completed} vs {local_completed}")
    completed = full_completed
    freeze_check = {
        "passed": True, "trainable_parameters": trainable,
        "training_mode": model.training, "dtype": str(next(model.parameters()).dtype),
    }
    print(json.dumps({"freeze_check": freeze_check, "resume_from": completed, "total": len(rows)}), flush=True)

    windows_processed = 0
    row_group_size = 4
    with torch.inference_mode():
        for group_start in range(completed, len(rows), row_group_size):
            group_end = min(group_start + row_group_size, len(rows))
            token_ids_by_row = {}
            sums_by_row = {}
            counts_by_row = {}
            chunks = []
            for row_index in range(group_start, group_end):
                token_ids_by_row[row_index] = {}
                sums_by_row[row_index] = {
                    field: np.zeros((layer_count, hidden_size), dtype=np.float64)
                    for field in ("text", "focus_text")
                }
                counts_by_row[row_index] = {"text": 0, "focus_text": 0}
                for field in ("text", "focus_text"):
                    ids, field_chunks = source_windows(
                        tokenizer, str(rows[row_index][field]), args.max_length, args.overlap, field
                    )
                    token_ids_by_row[row_index][field] = ids
                    for chunk in field_chunks:
                        chunk["row_index"] = row_index
                    chunks.extend(field_chunks)
            for batch_start in range(0, len(chunks), args.batch_size):
                batch = chunks[batch_start:batch_start + args.batch_size]
                width = max(len(chunk["ids"]) for chunk in batch)
                input_ids = torch.full(
                    (len(batch), width), tokenizer.pad_token_id, dtype=torch.long, device=args.device
                )
                attention_mask = torch.zeros_like(input_ids)
                for item_index, chunk in enumerate(batch):
                    ids = torch.tensor(chunk["ids"], dtype=torch.long, device=args.device)
                    input_ids[item_index, :len(ids)] = ids
                    attention_mask[item_index, :len(ids)] = 1
                hidden_states = model(input_ids=input_ids, attention_mask=attention_mask).hidden_states
                for item_index, chunk in enumerate(batch):
                    begin = chunk["content_offset"] + chunk["unique_offset"]
                    end = chunk["content_offset"] + chunk["content_length"]
                    count = end - begin
                    if count <= 0:
                        continue
                    field = chunk["field"]
                    row_index = chunk["row_index"]
                    counts_by_row[row_index][field] += count
                    for layer, hidden in enumerate(hidden_states):
                        sums_by_row[row_index][field][layer] += hidden[item_index, begin:end].float().sum(0).cpu().numpy()
            for row_index in range(group_start, group_end):
                for field in ("text", "focus_text"):
                    if counts_by_row[row_index][field] != len(token_ids_by_row[row_index][field]):
                        raise RuntimeError(
                            f"Token accounting failed row={row_index} field={field}: "
                            f"{counts_by_row[row_index][field]} != {len(token_ids_by_row[row_index][field])}"
                        )
                for layer in range(layer_count):
                    full_stores[layer][row_index] = (
                        sums_by_row[row_index]["text"][layer] / counts_by_row[row_index]["text"]
                    ).astype(np.float32)
                    local_stores[layer][row_index] = (
                        sums_by_row[row_index]["focus_text"][layer] / counts_by_row[row_index]["focus_text"]
                    ).astype(np.float32)
            windows_processed += len(chunks)
            if (group_end - completed) % 20 == 0 or group_end == len(rows):
                for store in full_stores + local_stores:
                    store.flush()
                write_json_atomic(output / "extraction_checkpoint.json", {
                    "completed_rows": group_end, "total_rows": len(rows),
                    "windows_this_process": windows_processed,
                })
                last_row = group_end - 1
                print(json.dumps({
                    "processed": group_end, "total": len(rows),
                    "full_tokens": len(token_ids_by_row[last_row]["text"]),
                    "local_tokens": len(token_ids_by_row[last_row]["focus_text"]),
                    "windows": len(chunks),
                }), flush=True)

    write_json_atomic(output / "run_metadata.json", {
        "model_path": args.model_path, "data_path": args.data_path, "rows": len(rows),
        "max_length": args.max_length, "overlap": args.overlap,
        "full_pooling": "mean over every global full-source code token exactly once",
        "local_pooling": "mean over every token in annotation-grounded edited functions exactly once",
        "overlap_is_context_only": True,
        "freeze_check": freeze_check,
    })


if __name__ == "__main__":
    main()
