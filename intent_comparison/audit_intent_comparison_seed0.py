#!/usr/bin/env python3
"""Re-run final table generation and strict output auditing."""

from intent_comparison_common import seed_all
from train_intent_comparison_seed0 import summarize


if __name__ == "__main__":
    seed_all(0)
    summarize()
