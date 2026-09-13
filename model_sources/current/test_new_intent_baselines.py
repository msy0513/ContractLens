#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from honeytoken_symbolic import TRANSFER, analyze_bytecode, disassemble, selector_entries
from intent_comparison_common import INTENTS, OUT
from train_new_intent_baselines_seed0 import balanced_pairs, reverse_valid_tokens, task_data


class HoneytokenSymbolicTests(unittest.TestCase):
    def test_modern_push0_and_proxy_rule(self):
        # Six stack arguments followed by DELEGATECALL and STOP.
        result = analyze_bytecode("0x5f5f5f5f5f5ff400", timeout_seconds=2)
        self.assertTrue(result["proxy_token"])
        self.assertGreater(result["instruction_count"], 0)

    def test_dispatcher_and_transfer_restriction(self):
        # transfer selector -> block containing CALLER == SLOAD(...) guard.
        bytecode = "0x63a9059cbb14600a57005b3360005414601457005b00"
        instructions = disassemble(bytecode)
        self.assertEqual(selector_entries(instructions)[TRANSFER], 10)
        result = analyze_bytecode(bytecode, timeout_seconds=2)
        self.assertTrue(result["transfer_restriction"])


class NeuralProtocolTests(unittest.TestCase):
    def test_valid_token_reverse_preserves_padding(self):
        x = torch.arange(10).reshape(2, 5, 1).float()
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.bool)
        output = reverse_valid_tokens(x, mask).squeeze(-1)
        self.assertEqual(output[0].tolist(), [2, 1, 0, 3, 4])
        self.assertEqual(output[1].tolist(), [8, 7, 6, 5, 9])

    def test_pairs_are_balanced_and_never_self_pair(self):
        labels = np.asarray([0, 0, 0, 1, 1, 1])
        anchors, partners, same = balanced_pairs(labels, epoch=0)
        self.assertEqual(int((same == 1).sum()), len(labels))
        self.assertEqual(int((same == 0).sum()), len(labels))
        self.assertTrue(np.all(anchors[same == 1] != partners[same == 1]))
        self.assertTrue(np.all(labels[anchors[same == 1]] == labels[partners[same == 1]]))
        self.assertTrue(np.all(labels[anchors[same == 0]] != labels[partners[same == 0]]))

    def test_all_manifests_match_global_split(self):
        audit = json.loads((OUT / "manifests/global_split_audit.json").read_text())
        self.assertFalse(any(audit["component_overlap"].values()))
        for intent in INTENTS:
            data = task_data(intent)
            self.assertEqual(set(data), {"train", "val", "test"})
            for split in ("train", "val", "test"):
                self.assertEqual(set(np.unique(data[split]["label"])), {0, 1})


if __name__ == "__main__":
    unittest.main()
