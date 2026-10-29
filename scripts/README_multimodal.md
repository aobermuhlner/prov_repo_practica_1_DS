# Resumable multimodal training

The script tunes individual branches, tunes a logistic-regression combiner, then
selects raw or temperature-calibrated probabilities **before** evaluating test
rows. Every decision optimizes weighted F1 × (1 − confidence Brier).

## Use your existing fitted branches first

The completed protocol-1 run can supply its fitted branches. This reruns only
meta-model fitting/selection and calibration, with convergence diagnostics and
scaled meta-features. It keeps the original run intact:

```bash
uv run scripts/train_multimodal.py \
  --strategy holdout \
  --reuse-base-from models/multimodal_search/3b7f0efaadf7 \
  --output models/multimodal_meta_v2
```

Repeat the same command to resume. Add `--evaluate-test` once the development
choices are settled. Keep `--strategy`, `--reuse-base-from`, and `--output` when
resuming. Input hashes, library versions, class splits, seed and search settings
are checked before reuse. The source must be a completed holdout run. Historical
branch search diagnostics are not recreated; convergence tracking applies to
new fits. `--reuse-base-from` cannot be combined with `--retrain`.

## Fresh OOF workflow

```bash
# Default: nested OOF stacking, final full-training refits, calibration selection.
uv run scripts/train_multimodal.py

# Resume; optionally stop after a stage.
uv run scripts/train_multimodal.py --through base
uv run scripts/train_multimodal.py --through meta
uv run scripts/train_multimodal.py

# Explicit final evaluation: print development-selected models' scores.
uv run scripts/train_multimodal.py --evaluate-test

# Optional diagnostic comparisons and reliability plots.
uv run scripts/train_multimodal.py --evaluate-test --compare-calibration
uv run scripts/train_multimodal.py --evaluate-test --calibration-plots

# Fresh run; retain old run directories.
uv run scripts/train_multimodal.py --retrain

# Optional extra branch; use a separate configuration/output.
uv run scripts/train_multimodal.py --embedding-hgb --output models/multimodal_hgb_v2
```

Default output is `models/multimodal_search_v2/`, so the original artifacts remain
untouched. `--jobs 4` limits threads; searches and folds run sequentially. Use
`--quick` only for smoke testing: it tests one candidate per branch with five
trees. Changing data, source, dependencies or training settings requires a new
output or `--retrain`. Stage/evaluation flags can change when resuming.

## Splits and fit order

All partitions are saved before fitting. Company IDs are unique and disjoint
between the original training and validation files. The test membership is
identical to protocol 1; previously observed test rows never move into training.

| Partition | Default OOF | Holdout / branch reuse |
|---|---|---|
| Base development pool | 86% of original training | 65% of original training |
| Separate meta-fit rows | OOF predictions on base development pool | 21% of original training |
| Meta selection | Remaining 14% of original training | Remaining 14% of original training |
| Calibration fit | 25% of original validation | Same |
| Calibration selection | 25% of original validation | Same |
| Test | Remaining 50% of original validation | Same |

The default flow is:

1. Tune each branch by inner CV on the base development pool. Fit the winners
   on that pool to predict the untouched meta-selection partition.
2. Generate OOF predictions on the base development pool using outer CV.
   **Repeat hyperparameter selection entirely within each outer training fold.**
   Neither fitting nor parameter selection sees the outer-held-out row.
3. Train candidate combiners on these nested OOF predictions. Select using the
   separate meta-selection partition. Its rows have not entered branch tuning,
   branch fitting, or candidate combiner fitting.
4. Refit the chosen combiner on both the OOF data and meta-selection predictions.
   Refit the chosen development-pool branch configurations on **100% of original
   training rows**. No further parameter search occurs during these final refits.
5. Freeze final branches and combiner. Fit temperature calibrators on calibration
   fit rows. Compare raw and temperature predictions on calibration-selection
   rows using the combined metric. Ties keep raw probabilities. Preserve the
   selected calibrator as fitted; do not refit it using selection rows.
6. Save the selected inference rules. Evaluate the test partition only when
   `--evaluate-test` is passed. Optional test comparisons never change selection.

As in ordinary stacking, final branches use more rows than the models that
produced the meta-training probabilities; the resulting distribution shift is
a modeling tradeoff. Nested tuning prevents the additional leakage that would
occur if outer-held-out labels influenced branch hyperparameters.

Holdout mode keeps the original branch/meta partitions and does not perform
nested OOF or final full-training branch refits. It is useful for an inexpensive
meta/calibration update using existing branches.

Meta-selection scores are used to choose configurations, not presented as
unbiased final estimates. Calibration-selection scores likewise guide a choice.
Because the existing test has already been inspected, use genuinely new data
for a final generalization claim. The script does not invent a fresh test set
by reshuffling previously observed rows.

## Searches and convergence

| Branch | Candidates | Parameters searched |
|---|---:|---|
| Cosine KNN on embeddings | 12 | Scaling on/off; neighbors 11/19/35; uniform/distance weights |
| Logistic regression on embeddings | 8 | Scaling on/off; `C` 0.01/0.1/1/10 |
| Tabular XGBoost | 12 | Country/global/native missing handling; indicators on/off; depth 3/5 |
| Tabular random forest | 8 | Country/global medians; clipping on/off; minimum leaf size 2/8 |
| Optional PCA + embedding histogram boosting | 4 | PCA fixed at 32 dimensions; leaves 7/15; L2 1/10 |

Default nested OOF with three inner and three outer folds entails **480 branch
CV fits, 16 winner refits, and four final full-training refits**. This costs about
four times as many branch fits as the original holdout workflow. It is not a
cheap rerun; all completed trials and individual folds are checkpointed. A stopped
fit restarts, while completed folds are reused.

Country medians require 10 observed values per feature, otherwise global medians
are used. Missing/unseen countries also use global medians; completely empty
features fall back to zero. Missingness and infinity flags precede imputation.
Optional clipping uses training-fold finite quantiles 0.005/0.995. Native XGBoost
handling leaves numeric NaNs. Scaling, clipping, imputation and PCA are learned
only within their training fold.

The combiner tests five branch sets:

- KNN + logistic regression.
- All branches.
- All except random forest.
- All except XGBoost.
- KNN probabilities only.

Each tests probabilities and clipped log-probabilities, with `C` 0.01/0.1/1/10:
**40 meta candidates**. Every candidate scales its meta-features using training
rows only. Inputs contain every class probability from each selected branch:
four models and 25 classes yield 100 columns. No embedding/probability PCA or
financial feature interactions are added. A uniform soft-vote development
baseline is saved separately.

New logistic-regression fits record solver, iteration limit, iterations used,
convergence status and warning text. A warning triggers bounded retries: LBFGS
with at least 6,000 iterations, then Newton-CG with 2,000 iterations. Retries keep
the same features, regularization and objective. A trial that still fails to
converge is recorded and excluded from selection. A final refit that fails stops
the run instead of saving an unconverged winner. Other fitting errors also stop
the run, retaining completed checkpoints.

Temperature scaling minimizes multiclass log loss, which differs from our
confidence Brier objective. This is why calibration is now a development-selected
option. The stack always consumes **raw branch probabilities**, consistently
through fitting, calibration and inference; independently selected branch
calibrators do not change the stack's inputs.

## Saved artifacts and inference

Each run has a manifest with input/configuration fingerprints, split indices and
company IDs, label encoding, branch searches, nested-fold indices, cached
probabilities, meta trials and convergence diagnostics, final branches,
calibrators, `calibration_selection.json`, a complete inference bundle, and
optional test-score CSVs. Run directories are retained under `models/` and ignored
by git. Only load joblib artifacts you trust.

```python
from pathlib import Path
import json
import joblib
from src.multimodal_training import predict_bundle

root = Path("models/multimodal_search_v2")
run_id = json.loads((root / "manifest.json").read_text())["run"]
bundle = joblib.load(root / run_id / "bundle.joblib")

# embeddings: finite 2D matrix. financial_rows: matching rows with country_code,
# net_profit_margin, ebitda_margin, asset_turnover.
probabilities, industries = predict_bundle(bundle, embeddings, financial_rows)
# Default uses the raw/temperature rule selected on development data.
# calibrated=False forces raw; calibrated=True forces temperature for diagnostics.
# Probability columns follow bundle["label_encoder"].classes_.
```

`evaluate_model(..., calibration_plots=False)` prints scores only. Set the flag
to `True` for a confidence table and reliability plot. The confidence Brier value
measures confidence in correctness, rather than full multiclass Brier.
