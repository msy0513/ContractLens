# Model catalog

## Current effective versions

The revised comparison uses five reported methods. The exact implementation
entry point is `model_sources/current/run_cross_type_seed0.py`.

### SmartIntentNN

- Implementation: `Seq(v2=False, centers=..., highlight=True)`.
- Backbone: frozen CodeGemma segment features, 3072 dimensions.
- Head: bidirectional LSTM with 128 hidden units per direction and a linear
  binary output layer.
- Training: binary cross entropy with logits, validation AUROC selection.

### SmartIntentV2

- Implementation: `Seq(v2=True, centers=None)`.
- Backbone: frozen CodeGemma segment features.
- Projection: 3072 to 768, followed by the same bidirectional LSTM family.
- Training: binary cross entropy with logits, validation AUROC selection.

### DBESN

- Definition: `AdaptedDBESN` in
  `model_sources/current/train_new_intent_baselines_seed0.py`.
- Architecture: 3072-to-1024 projection, forward/backward GRU-attention
  encoders, final GRU, and binary output layer.
- Serialized files: `model_artifacts/checkpoints/complete_case_adapted/dbesn/`.

### SCMDetector

- Definition: `AdaptedSCMDetector` in
  `model_sources/current/train_new_intent_baselines_seed0.py`.
- Architecture: 3072-to-300 projection, word-level and line-level BiLSTM with
  masked attention, dropout, and binary output layer.
- Serialized files: `model_artifacts/checkpoints/complete_case_adapted/scmdetector/`.

### SmartDNA

- Definition: `smartdna()` in
  `model_sources/current/run_cross_type_seed0.py`.
- Representation: pre-registered stable MLP units from CodeGemma layers
  16/18/19/21.
- Classifier: standardized logistic probe with validation-selected layer and
  key-unit selection.

## Checkpoint provenance

Every serialized checkpoint is a seed-0 artifact from the complete-case
adapted baseline run. The JSON results that describe that run remain in the
working tree, while the release metadata for the revised exclusive-balanced
run is under `model_artifacts/evaluation/`. The revised runner currently keeps
SmartIntentNN, SmartIntentV2, DBESN, and SCMDetector weights in memory and only
serializes their metrics; rerunning it is required to produce matching revised
weights.
