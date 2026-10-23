# ContractLens: Latent Security Knowledge for Malicious Smart Contract Detection

This repository contains the Python implementation of **ContractLens**, a
mechanistic-probing framework for locating and using latent malicious-contract
knowledge in frozen pretrained code LLMs.

> ContractLens identifies critical layers whose representations are separable
> for malicious and benign contracts, sensitive to malicious-logic removal, and
> stable under behaviour-preserving surface edits. It then selects
> malicious-contract-discriminative units (MCDUs) from those layers and trains
> a lightweight detector on their activations.

## Overview

The pretrained backbone remains frozen throughout probing and detector
training:

```text
Solidity contracts -> Frozen code LLM -> Layer-wise representations
                                      |-- Linear-probe AUROC
                                      |-- Counterfactual logic-removal sensitivity
                                      `-- Surface-edit invariance
                                                |
                         L_critical = L_sep ∩ L_sens ∩ L_stab
                                                |
                              MLP activation contrast
                              Malicious-logic specificity
                              Probe weights
                                                |
                                Tri-factor score + Kneedle
                                                |
                                               MCDUs
                                                |
                                  Lightweight detector
```

The released files form a research code bundle, not a self-contained packaged
application. Model weights, raw datasets, large feature caches, and trained
checkpoints are not included.

## Repository layout

- `candidate_layers/` — SmartIntent conversion, sliding-window feature
  extraction, layer-wise probes, counterfactual pairs, and surface-invariance
  analysis.
- `key_neurons/` — MLP-unit extraction, tri-factor scoring, stable-unit and
  semantic validation, interventions, and neuron-specificity experiments.
- `intent_comparison/` — ten-intent feature preparation, baseline comparison,
  symbolic/honeytoken audit utilities, and result summarization.

## Requirements

Experiments use Python 3.10/3.12, PyTorch 2.x with CUDA, Transformers 4.x,
NumPy, SciPy, pandas, scikit-learn, Matplotlib, and PyArrow for Parquet inputs.
The main experiments used a single NVIDIA Tesla V100 GPU with 32 GB memory.
Exact package versions vary by script; use an isolated environment and install
versions compatible with your checkpoint. Feature extraction scripts expect
local model files (`local_files_only=True`).

## Dataset

The experiments use the public SmartIntent smart-contract intent dataset
(Huang et al., 2025), built from verified Binance Smart Chain (BSC) source
code. After parsing and filtering, the paper reports 32,141 contracts with ten
fine-grained intent labels:

`Blacklist`, `Mint`, `Honeypot`, `Fee`, `DisableTrading`, `MaxSell`, `MaxTX`,
`Reflection`, `Reward`, and `Rebase`.

For binary detection, a contract with at least one target intent is labelled
**malicious**; a contract without any target intent is labelled **benign**.
These are code-level capability labels, not claims that every maliciously
labelled contract has been exploited on-chain. The paper reports 21,166
malicious and 10,975 benign contracts.

The source dump is not redistributed here. Obtain `web3_all.sql.gz` from the
dataset provider and pass its local path to
`candidate_layers/build_smartintent.py`. The script emits JSONL records with
source text, binary and intent labels, contract ID, address, chain, creator, and
a creator/source group key.

### Split protocol

Contracts sharing a creator or source-code group stay in one partition. The
paper uses a 0.64/0.16/0.20 train/validation/test split. Training data fits
probes and supplies neuron statistics; validation data selects layers and
thresholds; the test set is reserved for final evaluation. Test labels must not
be used to construct candidate layers, MCDUs, or thresholds.

## Usage

Run commands from the repository root. Replace example paths with paths on your
machine; most experiment scripts originally used workspace-specific paths.

### 1. Prepare data

```bash
python candidate_layers/build_smartintent.py \
  --dump /path/to/web3_all.sql.gz \
  --output-dir runs/smartintent \
  --binary-per-class 1000 \
  --intent-per-class 500 \
  --seed 0
```

Build the sliding-window dataset used by the CodeGemma experiments:

```bash
python candidate_layers/build_smartintent_sliding_union.py \
  --data-dir runs/smartintent \
  --output-dir runs/sliding
```

### 2. Extract frozen representations

Long contracts are processed with a maximum of 2,048 tokens and 512-token
overlap. Contract representations use mean pooling over non-padding tokens.

```bash
python candidate_layers/extract_codegemma_token_exact_v3.py \
  --model-path /path/to/CodeGemma-7B \
  --data-path runs/sliding/sliding_union.jsonl \
  --output-dir runs/features_token_mean \
  --max-length 2048 --overlap 512 \
  --batch-size 4 --device cuda --resume
```

Paired counterfactual or surface-perturbation representations can be extracted
with:

```bash
python candidate_layers/extract_paired_full_local_features.py \
  --model-path /path/to/CodeGemma-7B \
  --data-path /path/to/paired_manifest.jsonl \
  --output-dir runs/paired_features \
  --max-length 2048 --overlap 512 --device cuda --resume
```

### 3. Localize critical layers

Fit standardized layer-wise logistic-regression probes while keeping the
backbone frozen:

```bash
python candidate_layers/run_layerwise_probes.py \
  --features-dir runs/features_token_mean \
  --data-path runs/sliding/binary.jsonl \
  --output-dir runs/layerwise_probes \
  --seeds 0
```

The separability candidate set is based on validation AUROC `A(l)`:

\[
L_{sep}=\{l:A(l)\geq\tau_{sep}\}.
\]

For malicious-logic sensitivity, compare representation change after removing
the malicious logic with a size-matched non-semantic control:

\[
R_{cf}^{(l)}=\frac{E[d_{cf}^{(l)}]}{E[d_{ctrl}^{(l)}]+\epsilon},\qquad
L_{sens}=\{l:R_{cf}^{(l)}\geq\tau_R,\Delta p^{(l)}\geq\tau_p\}.
\]

For behaviour-preserving edits, measure representation cosine similarity,
prediction consistency, and probability shift:

\[
L_{stab}=\{l:I_{cos}^{(l)}\geq\tau_I,\ C^{(l)}\geq\tau_C,
S_p^{(l)}\leq\tau_S\}.
\]

The final set is the intersection:

\[
L_{critical}=L_{sep}\cap L_{sens}\cap L_{stab}.
\]

The paper configuration uses `delta AUROC = 0.008`, `tau_R = 1.50`,
`tau_I = 0.925`, `tau_p = 0.02`, `tau_C = 0.95`, and `tau_S = 0.04`.

### 4. Select MCDUs

Within the critical layers, the method computes MLP activation contrast,
linear-probe weight, and malicious-logic specificity for each neuron:

\[
a^{contrast}_{l,k}=E_{mal}[a_{l,k}]-E_{ben}[a_{l,k}],
\]

\[
r_{l,k}=\frac{E_{mal}|a_{l,k}(x)-a_{l,k}(x_{cf})|}
{E_{x,T}|a_{l,k}(x)-a_{l,k}(T(x))|+\epsilon},
\qquad
S_{l,k}=|a^{contrast}_{l,k}v_{l,k}|r_{l,k}.
\]

Candidate scores are globally min-max normalized, ranked, and passed to
Kneedle; neurons before the detected knee are retained as
**malicious-contract-discriminative units (MCDUs)**.

Example scripts included in this bundle:

```bash
python key_neurons/run_neuron_specificity_seed0.py \
  --model-path /path/to/CodeGemma-7B \
  --output-dir runs/neuron_specificity \
  --random-repeats 30 --seed 0 --device cuda --resume

python key_neurons/stage10_stable_unit_discovery.py \
  --stage9 runs/stage9_output \
  --output runs/stable_units \
  --bootstrap-repeats 100 --random-repeats 50 --seed 0
```

### 5. Evaluate MCDU detection and controls

The selected MCDU activations are concatenated across critical layers and used
to train a lightweight logistic-regression detector. Specificity experiments
compare MCDUs with `Random-In` and `Random-Ex` sets; intervention and ablation
scripts evaluate the contribution of selected units.

```bash
python key_neurons/stage11_unit_semantic_validation.py --output runs/unit_semantics
python key_neurons/stage12_true_mlp_intervention.py --output runs/intervention
python key_neurons/stage14_codegemma_unit_ablation.py --output runs/unit_ablation
```

### 6. Compare malicious intents

The `intent_comparison/` scripts provide seed-0 ten-intent feature preparation,
baseline training, and auditing:

```bash
python intent_comparison/build_intent_comparison_seed0.py --stage all
python intent_comparison/extract_intent_comparison_features.py
python intent_comparison/train_intent_comparison_seed0.py --stage train
python intent_comparison/train_new_intent_baselines_seed0.py --method all --stage train
```

## Evaluation

The primary metrics are AUROC and accuracy; F1 and AUPRC are also reported,
especially for imbalanced and unseen-intent settings. Unseen-intent evaluation
holds one intent out of localization and training and uses it only for testing.
Neuron-specificity baselines repeat random neuron selection (30 repetitions in
the paper) while keeping the MCDU set fixed.

## Reproducibility notes

- Set random seeds explicitly; `--seed 0` follows the released seed-0 runs.
- Reuse the same group-aware split manifest within a trial.
- The pretrained model is frozen; only linear probes and lightweight
  downstream classifiers are optimized.
- Use `--resume` for long feature extraction jobs and check disk space first.
- Scripts may need path edits when moved outside the original workspace.

## Citation

```bibtex
@inproceedings{miao2027contractlens,
  title     = {ContractLens: Latent Security Knowledge for Malicious Smart Contract Detection},
  author    = {Miao, Shuyi and Huang, Xinyi and Qiu, Wangjie and Shen, Fei and He, Jinchun and Zheng, Zhiming and Chua, Tat-Seng},
  booktitle = {International Conference on Learning Representations},
  year      = {2027}
}
```

## Responsible use

This project is intended for research and defensive smart-contract analysis.
Check the licenses of the pretrained models and dataset before redistribution
or commercial use. A malicious label indicates a labelled unsafe capability;
it is not by itself proof of an on-chain exploit or legal finding.
