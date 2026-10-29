# Small multimodal neural network

Independent entrypoint: `scripts/train_multimodal_nn.py`. It reads the same
precomputed embeddings and financial files as stacking. It does not change
stacking code, artifacts, dependencies, or the running nested search. PyTorch is
already installed in the project environment through sentence-transformers.

## Run on this laptop

```bash
# Default: cross-entropy plus confidence Brier, early stopping, CPU/CUDA auto.
uv run scripts/train_multimodal_nn.py --device cpu

# Compare four loss functions, keeping architecture and initial seed fixed.
uv run scripts/train_multimodal_nn.py --device cpu --loss all --output models/nn_loss_comparison

# Alternative surrogate objective.
uv run scripts/train_multimodal_nn.py --device cpu --loss ce_score --output models/nn_score

# Smoke test only: two epochs, separate artifacts, no test evaluation.
uv run scripts/train_multimodal_nn.py --device cpu --epochs 2 --output models/nn_smoke

# Resume a matching run; increasing --epochs extends a budget-limited run.
uv run scripts/train_multimodal_nn.py --device cpu

# Evaluate only after development choices are settled.
uv run scripts/train_multimodal_nn.py --device cpu --evaluate-test

# Optional raw/temperature comparison and calibration plots.
uv run scripts/train_multimodal_nn.py --device cpu --evaluate-test --compare-calibration
uv run scripts/train_multimodal_nn.py --device cpu --evaluate-test --calibration-plots
```

Resume commands must retain the original loss, architecture and preprocessing
arguments, plus `--output` if customized. `--retrain` creates a new run without
deleting prior runs. CPU threads default to two, batch size to 256, maximum epochs
to 60, and early-stopping patience to eight. No large hyperparameter search is
started. `--loss all` trains four candidates, not a Cartesian parameter grid.

## Architecture

```text
768-dimensional embedding
  -> optional input normalization (default per-row L2)
  -> Linear(768,128) -> LayerNorm -> ReLU -> Dropout(0.2)
  -> Linear(128,64) -> ReLU
                                                    \
                                                     concat(80)
                                                    /   -> Linear(80,64)
3 financial ratios + 6 missing/infinity flags           -> ReLU -> Dropout(0.2)
+ learned country embedding (8 dimensions)              -> Linear(64,25)
  -> Linear(17,32) -> LayerNorm -> ReLU                  -> logits
  -> Linear(32,16) -> ReLU                              -> softmax for probabilities
```

Dimensions adjust to the loaded data. On the current data this is **115,689
trainable parameters**, about 0.44 MiB of FP32 parameters. Optimizer, gradients,
activations and the CUDA runtime use additional memory. Text embeddings stay
fixed; the sentence-transformer is not fine-tuned. Training uses AdamW, weight
decay 0.001, gradient clipping, and FP32. There is no mixed precision requirement
or dependence on tensor cores. LayerNorm supports small batches without batch
normalization statistics.

## Financial processing and missingness

Preprocessing is fitted **only on neural training rows**, then saved and reused:

1. Treat `+inf` and `-inf` ratios as missing. Preserve separate missingness and
   infinity flags for every ratio; a missing flag includes infinities.
2. By default clip finite ratios at training quantiles 0.005/0.995. `--no-clip`
   disables clipping. No company is removed.
3. Fill from a country median when at least 10 finite training observations exist
   for that country/feature; otherwise use the global training median. Missing or
   unseen countries also use global medians. Completely empty features use zero.
   `--imputation global` uses only global medians. The medians are computed from
   the clipped training values when clipping is enabled.
4. Apply `sign(x) * log1p(abs(x))`, which supports negative ratios, then standardize
   the three ratios using training statistics. Keep indicator flags as 0/1.
5. Country is a small learned embedding. Missing country has its own trainable
   category; an unseen country uses a zero vector. Country vocabulary is learned
   only from training rows.

Embedding input options are `--embedding-normalization l2` (default), `standard`
(train-fitted StandardScaler), and `none`. Numeric inputs never contain NaNs;
MLPs cannot use NaNs as XGBoost can. Industry labels never enter imputation.
Missing financial features do not cause otherwise useful text rows to be dropped.

## Loss functions

The network returns **logits**. Cross-entropy receives logits directly; softmax
is applied when a probability-based loss or evaluation needs probabilities.
[PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)

Let `p` be the class-probability vector, `c=max(p)`, and `correct` indicate whether
`argmax(p)` equals the true class. The project's confidence Brier term is:

```text
B_conf = mean((c - correct)^2)
score  = weighted_F1 * (1 - B_conf)
```

| `--loss` | Training objective |
|---|---|
| `ce` | Cross-entropy baseline |
| `ce_brier` (default) | Cross-entropy + `brier_weight * B_conf` |
| `ce_score` | Cross-entropy + `score_weight * (1 - soft_weighted_F1 * (1 - B_conf))` |
| `multiclass_brier` | Mean per-row sum of squared errors against one-hot labels |
| `confidence_brier` | `B_conf` alone; explicitly experimental |
| `all` | Compare the first four choices |

`--brier-weight` and `--score-weight` default to one. The multiclass Brier loss is
different from the selected-class confidence Brier used by the combined metric.

The exact F1 score uses discrete argmax decisions, so ordinary backpropagation
cannot optimize it directly. The surrogate replaces hard class counts with
probability sums:

```text
soft_TP[k] = sum_i(p[i,k] * one_hot_y[i,k])
soft_F1[k] = 2*soft_TP[k] / (sum_i p[i,k] + sum_i one_hot_y[i,k])
soft_weighted_F1 = sum_k(true_class_fraction[k] * soft_F1[k])
```

This is a minibatch approximation, not the exact dataset F1. The confidence Brier
term uses the exact hard correctness indicator as a detached target; gradients
flow through confidence. It is piecewise differentiable and changes its target
at argmax boundaries. Pure confidence Brier can favor weak, low-confidence
predictions without learning good class decisions, so `all` excludes that mode
and the default retains cross-entropy.

**Epoch and loss selection always use the actual combined metric**, with hard
predictions and weighted F1 on the development partition. `--min-delta` controls
the minimum early-stopping improvement (default 0.0001). No training surrogate
value is substituted for the evaluation score.

## Splits, calibration and evaluation

Splits are stratified and saved before fitting:

| Partition | Rows |
|---|---|
| Training | 80% of original training data: 19,531 currently |
| Development | 20% of original training data: 4,883 currently |
| Calibration fit | 25% of original validation data: 678 currently |
| Calibration selection | 25% of original validation data: 678 currently |
| Test | Remaining 50% of original validation: 1,357 currently |

The test membership matches the stacking scripts when seed is the same. Do not
use these observed test rows repeatedly to choose architectures/losses; this is
already an observed benchmark, not a fresh final generalization estimate.

The development partition selects epochs and, with `--loss all`, the loss. The
winning network stays fitted on training rows; there is no post-selection refit
on development rows. Early stopping avoids choosing an arbitrary epoch budget.

Temperature is a positive scalar applied to logits before softmax. By default
it minimizes **confidence Brier on calibration-fit rows**. Because positive
temperature preserves argmax, F1 is fixed and minimizing this term also maximizes
the combined score on those rows. `--temperature-objective log_loss` provides the
conventional cross-entropy alternative. Temperature search is bounded to
`exp(-4)..exp(4)` and keeps `T=1` if it cannot improve the fit objective.

A separate calibration-selection partition then chooses raw versus temperature
predictions by the combined score. Ties keep raw. The selected prediction rule
is frozen before optional test evaluation. Default evaluation prints F1,
confidence Brier, combined score, accuracy and mean confidence; plots are opt-in.
`--compare-calibration` is diagnostic and never changes which rule is selected.

## Checkpoints and inference

Artifacts are under `models/multimodal_nn/<run-id>/` by default:

- Input/configuration/source fingerprints; split indices and company IDs.
- Saved preprocessing and label encoding, architecture and runtime metadata.
- Each candidate's latest model/optimizer/RNG/early-stopping state and best weights.
- Epoch histories, development scores, selected model, calibration choice and
  optional test scores.

Each completed epoch is checkpointed atomically. An interrupted epoch restarts;
completed epochs resume with optimizer and RNG state. CPU replay is tested to
match uninterrupted training. `--epochs` can increase without invalidating the
run, but a run already stopped by patience remains stopped. Changes to loss
weights, learning rate, patience, data, preprocessing or architecture require a
new output or `--retrain`. Device/CPU-thread changes are allowed; CPU and CUDA
need not produce identical numerical results. A matching core PyTorch version
can use different CPU/CUDA wheel builds. Other recorded library/source versions
must match. Calibration is recomputed after resuming so new weights cannot reuse
stale calibration.

```python
from pathlib import Path
import json
from src.multimodal_nn import predict_nn

root = Path("models/multimodal_nn")
run = root / json.loads((root / "manifest.json").read_text())["run"]
probabilities, industries = predict_nn(run, embeddings, financial_rows, device="cpu")
# Financial rows must contain country_code and the three numeric ratios, aligned
# with the embedding rows. Preprocessing is automatic; do not impute beforehand.
# Default uses the development-selected raw/temperature rule.
# calibrated=False/True forces raw/temperature for diagnostics.
```

Only load trusted checkpoints and preprocessing artifacts.

## GTX 1060 setup on the home PC

The GTX 1060 is a Pascal GPU. **CUDA 13 PyTorch wheels do not support Pascal**;
use a compatible CUDA 12.6 wheel, with PyTorch below 2.15. Official packaging
notes identify 2.14 as the last release shipping CUDA 12.6 wheels.
[PyTorch CUDA support matrix](https://github.com/pytorch/pytorch/blob/main/RELEASE.md),
[CUDA 12.6 retirement notice](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432)

Use an isolated environment so the CUDA choice does not change other experiments:

```bash
uv venv --python 3.11 .venv-nn
uv pip install --python .venv-nn/bin/python 'torch>=2.0,<2.15' --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv-nn/bin/python numpy pandas 'scikit-learn>=1.8' scipy xgboost joblib threadpoolctl matplotlib

.venv-nn/bin/python scripts/train_multimodal_nn.py --device cuda --loss all
```

On Windows, use `.venv-nn/Scripts/python.exe` for the interpreter paths. A compatible
NVIDIA driver is still required. Invoke this environment's Python directly to
retain the selected CUDA wheel. The script performs a real CUDA arithmetic and
backward probe; `--device cuda` fails clearly if it cannot execute on the GPU,
while `auto` falls back to CPU. Only CPU execution has been tested on this laptop.

## Experiment outcome

Keep this standalone MLP as the neural-network comparison. The saved baseline
reached about **64.1% confidence_score** on the existing test split. Wider networks,
learning-rate schedules, weight averaging and ensembles were explored, but the
observed gains were small and uncertain. The experimental variants were discarded;
the original train/development/calibration/test protocol and financial cleaning
remain in this implementation.
