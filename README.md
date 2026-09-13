# ContractLens model release

This repository packages the model implementations used in the ContractLens
smart-contract intent experiments. The primary evaluation protocol is
leave-one-intent-out classification with exclusive multi-label test ownership
and balanced train/validation splits.

## Included model versions

| Model | Effective implementation | Input | Checkpoint status |
|---|---|---|---|
| SmartIntentNN | CodeGemma sequence head, `Seq(v2=False, highlight=True)` | frozen CodeGemma token segments | source and protocol included; head was trained in memory and was not serialized by the original runner |
| SmartIntentV2 | CodeGemma sequence head, `Seq(v2=True)` with 3072-to-768 projection | frozen CodeGemma token segments | source and protocol included; head was trained in memory and was not serialized by the original runner |
| DBESN | `AdaptedDBESN` | frozen CodeGemma token segments | complete-case seed-0 checkpoints included |
| SCMDetector | `AdaptedSCMDetector` | frozen CodeGemma token segments | complete-case seed-0 checkpoints included |
| SmartDNA | stable MLP-unit probe with validation layer selection | frozen CodeGemma MLP activations | deterministic probe implementation included; no standalone classifier file was emitted |

The DBESN and SCMDetector checkpoint files under
`model_artifacts/checkpoints/complete_case_adapted/` belong to the earlier
complete-case training run. The revised exclusive-balanced cross-intent run
reported in `model_artifacts/evaluation/` did not serialize neural-network
weights. This distinction is kept explicit so checkpoints from different
protocols are not accidentally presented as the same experiment.

## Repository layout

```text
model_sources/current/                 Current runners and model definitions
model_sources/SmartIntent-py/          MIT-licensed upstream SmartIntent reference
model_artifacts/checkpoints/           Serialized DBESN/SCMDetector checkpoints
model_artifacts/evaluation/            Manifests, metrics, summaries, and audits
```

## CodeGemma backbone

The experiments use the frozen `google/codegemma-7b` backbone. Its 16 GB of
weights are intentionally not stored in this Git repository because GitHub
rejects files larger than 100 MB. Download the model from its official model
page after accepting Google's Gemma terms, then point the runners at the local
model directory. The small model configuration and tokenizer files may be
stored separately, but they are not a replacement for the weight shards.

## Reproducibility notes

- Intent labels: `Blacklist`, `Mint`, `Honeypot`, `Fee`, `DisableTrading`,
  `MaxSell`, `MaxTX`, `Reflection`, `Reward`, and `Rebase`.
- Train and validation splits contain 320 positive plus 320 negative and 80
  positive plus 80 negative examples, respectively, per fold.
- Test positives are assigned exclusively to one intent; test negatives are
  sampled from the available normal-contract pool.
- Accuracy uses a probability threshold of 0.5. AUROC, AUPRC, Accuracy, and F1
  are recorded in `model_artifacts/evaluation/metrics.csv`.
- Source data and frozen feature caches are excluded from this repository.
  Their paths and shapes are recorded in the evaluation metadata.

The files in this release are model/code artifacts and experiment metadata,
not a claim that the adapted baselines are strict reproductions of every
original paper implementation.
