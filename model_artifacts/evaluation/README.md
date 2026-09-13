# Evaluation artifacts

This directory contains the exclusive-balanced leave-one-intent-out manifests,
per-intent metrics, summary, and audit metadata used for the latest comparison.
The source contracts and CodeGemma feature arrays are intentionally excluded
because they are large dataset artifacts rather than model files.

The main table is `metrics.csv`. It reports test AUROC, AUPRC, Accuracy, and F1
for SmartIntentNN, SmartIntentV2, DBESN, SCMDetector, and SmartDNA.
