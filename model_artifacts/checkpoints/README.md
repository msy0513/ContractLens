# Serialized checkpoints

These files are seed-0 checkpoints from the complete-case adapted baseline
run. They are included because each file is below GitHub's 100 MB per-file
limit. The revised exclusive-balanced cross-intent run kept the neural models
in memory and therefore has no corresponding serialized weights.

The checkpoints are loaded with the matching classes in
`model_sources/current/train_new_intent_baselines_seed0.py`:

- `complete_case_adapted/dbesn/<intent>.pt` -> `AdaptedDBESN`
- `complete_case_adapted/scmdetector/<intent>.pt` -> `AdaptedSCMDetector`

The absolute paths embedded in the original result JSON files refer to the
training server. Use the relative paths above after cloning this repository.
