# CodeGemma core experiment scripts

This folder keeps the requested CodeGemma experiment code in three groups:

- `candidate_layers/`: candidate/critical layer discovery and its helper scripts.
- `key_neurons/`: MLP-unit scoring, stable-unit discovery, specificity, and interventions.
- `intent_comparison/`: the ten-intent method comparison and summary/audit scripts.

Files were selected from `/root/autodl-tmp/models/codegemma` first; missing
scripts were copied from `/root/autodl-tmp/models/the first server`. Two helper
dependencies were included even though they were not in the requested list:
`candidate_layers/build_seed0_binary_cf_inv_v2.py` is imported by the V3 dataset
builder, and `intent_comparison/honeytoken_symbolic.py` is imported by the
new-baselines trainer. The candidate-layer report was also retained as protocol
documentation.

This is a code bundle, not a self-contained rerun package. Model weights,
datasets/manifests, activation/feature caches, and trained checkpoints are not
included. Existing scripts use absolute or workspace-relative paths and may
need their paths/dependencies restored before running elsewhere.
