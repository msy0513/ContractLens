#!/usr/bin/env python3
"""Extract full MLP activations for the strict few-shot SmartDNA run.

The CodeGemma backbone is frozen.  Each requested transformer block stores the
mean pre-down-projection MLP activation over every source token exactly once.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from extract_codegemma_token_exact_v3 import windows, load_rows


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layers", nargs="+", type=int, default=[16, 18, 19, 21])
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--overlap", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--row-group-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    rows = load_rows(Path(args.data_path))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(
        args.model_path, local_files_only=True, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.requires_grad_(False)
    model.eval()
    model.config.use_cache = False
    model.to(args.device)
    if any(parameter.requires_grad for parameter in model.parameters()) or model.training:
        raise RuntimeError("CodeGemma freeze verification failed")

    blocks = getattr(model, "layers", None)
    if blocks is None:
        raise RuntimeError("CodeGemma transformer layers were not found")
    hidden = int(model.config.intermediate_size)
    layers = tuple(sorted(set(args.layers)))
    stores = {}
    for layer in layers:
        if layer < 1 or layer > len(blocks):
            raise ValueError(f"Layer {layer} is outside CodeGemma transformer blocks")
        path = output / f"layer_{layer:02d}.f16.npy"
        shape = (len(rows), hidden)
        if args.resume and path.exists():
            store = np.lib.format.open_memmap(path, mode="r+")
            if tuple(store.shape) != shape:
                raise RuntimeError(f"Feature shape mismatch for {path}: {store.shape} != {shape}")
        else:
            store = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=shape)
        stores[layer] = store

    checkpoint_path = output / "extraction_checkpoint.json"
    completed = 0
    if args.resume and checkpoint_path.exists():
        completed = int(json.loads(checkpoint_path.read_text(encoding="utf-8"))["completed_rows"])

    activation_cache = {}
    hooks = []
    for layer in layers:
        block = blocks[layer - 1]
        def make_hook(layer_id):
            def hook(_module, inputs):
                activation_cache[layer_id] = inputs[0].detach()
            return hook
        hooks.append(block.mlp.down_proj.register_forward_pre_hook(make_hook(layer)))

    print(json.dumps({"stage": "mlp_extraction", "rows": len(rows), "layers": list(layers),
                      "intermediate_size": hidden, "resume_from": completed,
                      "backbone_frozen": True}), flush=True)
    try:
        with torch.inference_mode():
            for group_start in range(completed, len(rows), args.row_group_size):
                group_end = min(group_start + args.row_group_size, len(rows))
                sums = {i: {l: np.zeros(hidden, dtype=np.float64) for l in layers}
                        for i in range(group_start, group_end)}
                counts = {i: 0 for i in range(group_start, group_end)}
                chunks = []
                for row_index in range(group_start, group_end):
                    _, row_chunks = windows(tokenizer, str(rows[row_index]["text"]),
                                            args.max_length, args.overlap, "text", row_index)
                    chunks.extend(row_chunks)
                for start in range(0, len(chunks), args.batch_size):
                    batch = chunks[start:start + args.batch_size]
                    width = max(len(c["ids"]) for c in batch)
                    input_ids = torch.full((len(batch), width), tokenizer.pad_token_id,
                                           dtype=torch.long, device=args.device)
                    mask = torch.zeros_like(input_ids)
                    for bi, chunk in enumerate(batch):
                        ids = torch.tensor(chunk["ids"], dtype=torch.long, device=args.device)
                        input_ids[bi, :len(ids)] = ids
                        mask[bi, :len(ids)] = 1
                    activation_cache.clear()
                    model(input_ids=input_ids, attention_mask=mask)
                    for bi, chunk in enumerate(batch):
                        begin = chunk["content_offset"] + chunk["unique_offset"]
                        end = chunk["content_offset"] + chunk["content_length"]
                        if end <= begin:
                            continue
                        row_index = chunk["row_index"]
                        counts[row_index] += end - begin
                        for layer in layers:
                            values = activation_cache[layer][bi, begin:end].float().sum(0)
                            sums[row_index][layer] += values.cpu().numpy()
                for row_index in range(group_start, group_end):
                    if counts[row_index] <= 0:
                        raise RuntimeError(f"No source tokens for row {row_index}")
                    for layer in layers:
                        stores[layer][row_index] = (sums[row_index][layer] / counts[row_index]).astype(np.float16)
                for store in stores.values():
                    store.flush()
                atomic_json(checkpoint_path, {"completed_rows": group_end, "total_rows": len(rows)})
                if group_end % 10 == 0 or group_end == len(rows):
                    print(json.dumps({"stage": "mlp_extraction", "processed": group_end,
                                      "total": len(rows)}), flush=True)
    finally:
        for hook in hooks:
            hook.remove()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    atomic_json(output / "run_metadata.json", {
        "model_path": args.model_path, "data_path": args.data_path,
        "rows": len(rows), "layers": list(layers), "intermediate_size": hidden,
        "max_length": args.max_length, "overlap": args.overlap,
        "pooling": "mean over every source token exactly once",
        "backbone_frozen": True, "storage_dtype": "float16",
    })


if __name__ == "__main__":
    main()
