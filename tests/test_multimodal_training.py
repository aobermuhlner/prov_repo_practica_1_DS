"""Checks for partition isolation and fold-local financial preprocessing."""
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from src.multimodal_training import (
    FinancialCleaner, NUMERIC, create_splits, fit_checked, checked_oof,
    nested_oof, select_calibration, FitDidNotConverge,
)


class FinancialCleaningTests(unittest.TestCase):
    def setUp(self):
        self.training = pd.DataFrame({
            "country_code": ["CH"] * 12 + ["US"] * 2,
            **{col: [2.] * 12 + [100., np.nan] for col in NUMERIC},
        })

    def test_country_support_unseen_country_and_no_fit_on_transform(self):
        cleaner = FinancialCleaner().fit(self.training)
        validation = pd.DataFrame({
            "country_code": ["CH", "US", "XX", None],
            **{col: [np.inf, -np.inf, np.nan, np.nan] for col in NUMERIC},
        })
        result = cleaner.transform(validation)
        np.testing.assert_allclose(result[NUMERIC], 2.)
        self.assertTrue((result.net_profit_margin_missing == 1).all())
        self.assertEqual(result.net_profit_margin_infinite.tolist(), [1, 1, 0, 0])
        # Transforming a huge value must not change learned imputation medians.
        extreme = self.training.copy()
        extreme[NUMERIC] = 9999.
        cleaner.transform(extreme)
        pd.testing.assert_frame_equal(result, cleaner.transform(validation))

    def test_native_missingness_and_training_only_clipping(self):
        cleaner = FinancialCleaner(imputation="native", clip=True).fit(self.training)
        validation = self.training.iloc[:2].copy()
        validation.loc[validation.index[0], NUMERIC] = np.inf
        validation.loc[validation.index[1], NUMERIC] = 1e10
        result = cleaner.transform(validation)
        self.assertTrue(result.iloc[0][NUMERIC].isna().all())
        np.testing.assert_allclose(result.iloc[1][NUMERIC].to_numpy(float), cleaner.upper_)

    def test_all_missing_fallback_is_finite(self):
        self.training[NUMERIC] = np.nan
        result = FinancialCleaner().fit_transform(self.training)
        self.assertTrue(np.isfinite(result[NUMERIC]).all().all())
        np.testing.assert_allclose(result[NUMERIC], 0.)


class SplitTests(unittest.TestCase):
    def test_reproducibility_disjointness_and_coverage(self):
        y = np.tile(np.arange(3), 200)
        first = create_splits(y, 450, 42)
        second = create_splits(y, 450, 42)
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])
            self.assertEqual(len(np.unique(y[first[name]])), 3)
        all_rows = np.concatenate(list(first.values()))
        np.testing.assert_array_equal(np.sort(all_rows), np.arange(len(y)))
        self.assertTrue((first['base'] < 450).all())
        self.assertTrue((first['calibration'] >= 450).all())
        self.assertTrue((first['test'] >= 450).all())


    def test_test_membership_is_preserved_from_original_protocol(self):
        from sklearn.model_selection import train_test_split
        y = np.tile(np.arange(3), 200)
        _, original_test = train_test_split(
            np.arange(450, 600), test_size=0.5, stratify=y[450:], random_state=42,
        )
        for strategy in ["holdout", "oof"]:
            splits = create_splits(y, 450, 42, strategy)
            np.testing.assert_array_equal(splits['test'], original_test)
            self.assertFalse(set(splits['calibration']) & set(splits['calibration_select']))


class ConvergenceTests(unittest.TestCase):
    def test_iteration_limit_retries_are_recorded(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(120, 5))
        y = (X[:, 0] > 0).astype(int)
        fitted, diagnostic = fit_checked(LogisticRegression(max_iter=1), X, y)
        self.assertTrue(diagnostic['converged'])
        self.assertFalse(diagnostic['attempts'][0]['converged'])
        self.assertGreater(len(diagnostic['attempts']), 1)
        self.assertLess(fitted.n_iter_[0], fitted.max_iter)

    def test_failed_retries_are_not_silently_accepted(self):
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        def never_converges(model, X, y):
            warnings.warn('forced iteration limit', ConvergenceWarning)
            return model
        with patch.object(LogisticRegression, 'fit', never_converges):
            with self.assertRaises(FitDidNotConverge) as caught:
                fit_checked(LogisticRegression(max_iter=1), np.zeros((4, 2)), np.array([0, 1, 0, 1]))
        self.assertFalse(caught.exception.diagnostics['converged'])
        self.assertEqual(len(caught.exception.diagnostics['attempts']), 3)


class OOFTests(unittest.TestCase):
    def test_nested_hyperparameter_search_never_sees_prediction_rows(self):
        X = np.arange(60).reshape(-1, 1)
        y = np.tile([0, 1], 30)
        heldouts = []
        class ExclusionCheckingModel:
            def __init__(self, train_ids):
                self.train_ids = set(train_ids)
            def predict_proba(self, X_valid):
                ids = X_valid[:, 0]
                self_test.assertFalse(self.train_ids & set(ids))
                heldouts.extend(ids)
                return np.full((len(ids), 2), 0.5)
        self_test = self
        def search(name, model, grid, X_train, y_train, directory, folds, seed):
            return ExclusionCheckingModel(X_train[:, 0])
        with tempfile.TemporaryDirectory() as tmp, patch('src.multimodal_training.tune_branch', search):
            proba = nested_oof({'emb_fake': (None, None)}, np.arange(60), X, None, y, Path(tmp), 3, 42)
        np.testing.assert_array_equal(np.sort(heldouts), np.arange(60))
        self.assertEqual(proba['emb_fake'].shape, (60, 2))

    def test_completed_oof_folds_are_reused(self):
        X = np.arange(60).reshape(-1, 1)
        y = np.tile([0, 1], 30)
        with tempfile.TemporaryDirectory() as tmp:
            first, _ = checked_oof(KNeighborsClassifier(3), X, y, 3, 42, Path(tmp))
            with patch('src.multimodal_training.fit_checked', side_effect=AssertionError('unexpected refit')):
                second, _ = checked_oof(KNeighborsClassifier(3), X, y, 3, 42, Path(tmp))
        np.testing.assert_array_equal(first, second)


class CalibrationSelectionTests(unittest.TestCase):
    def test_raw_can_win_and_temperature_can_win(self):
        class FixedModel:
            classes_ = np.array([0, 1])
            def __init__(self, confidence):
                self.confidence = confidence
            def predict_proba(self, X):
                p = self.confidence
                return np.array([[p, 1-p], [1-p, p]])
        stronger, weaker = FixedModel(0.9), FixedModel(0.6)
        y = np.array([0, 1])
        self.assertEqual(select_calibration(stronger, weaker, None, y)['method'], 'raw')
        self.assertEqual(select_calibration(weaker, stronger, None, y)['method'], 'temperature')
        self.assertEqual(select_calibration(stronger, stronger, None, y)['method'], 'raw')


if __name__ == '__main__':
    unittest.main()
