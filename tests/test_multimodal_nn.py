"""Neural losses, preprocessing isolation, resumability and calibration checks."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax

from src.multimodal_nn import (
    LOSSES, NeuralPreprocessor, calibration_choice, fit_temperature, loss_components,
    neural_splits, train_candidate, training_loss,
)
from src.multimodal_training import NUMERIC, create_splits, metrics


class NeuralLossTests(unittest.TestCase):
    def test_all_losses_produce_finite_nonzero_gradients(self):
        for name in [*LOSSES, 'confidence_brier']:
            torch.manual_seed(42)
            logits = torch.randn(12, 4, requires_grad=True)
            loss = training_loss(logits, torch.arange(12) % 4, name)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(logits.grad).all())
            self.assertGreater(logits.grad.abs().sum().item(), 0.)

    def test_confidence_brier_matches_the_reported_metric(self):
        logits = torch.tensor([[2., 0., -1.], [-1., 1., 3.], [2., 0., 1.]])
        y = torch.tensor([0, 1, 2])
        actual = loss_components(logits, y)['confidence_brier'].item()
        expected = metrics(y.numpy(), logits.softmax(1).numpy(), np.arange(3))['confidence_brier']
        self.assertAlmostEqual(actual, expected, places=6)


class NeuralPreprocessingTests(unittest.TestCase):
    def test_missing_unseen_countries_and_finite_features(self):
        embedding = np.arange(56, dtype=np.float32).reshape(14, 4)
        train = pd.DataFrame({'country_code': ['CH'] * 12 + ['US'] * 2,
                              **{col: [2.] * 12 + [100., np.nan] for col in NUMERIC}})
        processor = NeuralPreprocessor().fit(embedding, train)
        validation = pd.DataFrame({'country_code': ['CH', None, 'NEW'],
                                   **{col: [np.inf, -np.inf, np.nan] for col in NUMERIC}})
        mean_before = processor.ratio_scaler_.mean_.copy()
        emb, numeric, country = processor.transform(embedding[:3], validation)
        self.assertTrue(np.isfinite(emb).all() and np.isfinite(numeric).all())
        self.assertEqual(numeric.shape, (3, 9))
        self.assertGreaterEqual(country[0], 2)
        self.assertEqual(country[1:].tolist(), [1, 0])
        np.testing.assert_array_equal(mean_before, processor.ratio_scaler_.mean_)
        np.testing.assert_allclose(np.linalg.norm(emb, axis=1), 1., atol=1e-6)

    def test_partitions_are_disjoint_and_original_test_is_preserved(self):
        y = np.tile(np.arange(3), 200)
        splits = neural_splits(y, 450, 42)
        np.testing.assert_array_equal(np.sort(np.concatenate(list(splits.values()))), np.arange(600))
        np.testing.assert_array_equal(splits['test'], create_splits(y, 450, 42)['test'])


class NeuralCalibrationTests(unittest.TestCase):
    def test_brier_temperature_preserves_labels_and_improves_fit_objective(self):
        logits = np.array([[8., 0.], [0., 8.], [8., 0.], [0., 8.]])
        y = np.array([0, 1, 1, 0])
        temperature = fit_temperature(logits, y)
        np.testing.assert_array_equal(logits.argmax(1), (logits / temperature).argmax(1))
        before = metrics(y, softmax(logits, axis=1), np.arange(2))['confidence_brier']
        after = metrics(y, softmax(logits / temperature, axis=1), np.arange(2))['confidence_brier']
        self.assertLessEqual(after, before)

    def test_selection_can_keep_raw_and_ties_keep_raw(self):
        logits = np.array([[3., 0.], [0., 3.]])
        y = np.array([0, 1])
        self.assertEqual(calibration_choice(logits, y, 10.)[0], 'raw')
        self.assertEqual(calibration_choice(logits, y, 0.1)[0], 'temperature')
        self.assertEqual(calibration_choice(logits, y, 1.)[0], 'raw')


class NeuralResumeTests(unittest.TestCase):
    def test_epoch_resume_matches_uninterrupted_cpu_training(self):
        torch.set_num_threads(1)
        rng = np.random.default_rng(42)
        arrays = (rng.normal(size=(60, 8)).astype(np.float32),
                  rng.normal(size=(60, 9)).astype(np.float32), np.arange(60, dtype=np.int64) % 6)
        y = np.arange(60, dtype=np.int64) % 3
        splits = {'train': np.arange(45), 'development': np.arange(45, 60)}
        config = dict(embedding_dim=8, numeric_dim=9, n_countries=6, n_classes=3,
                      embedding_hidden=16, dropout=0.2)
        args = SimpleNamespace(seed=42, lr=1e-3, weight_decay=1e-3, epochs=1, patience=8,
                               batch_size=16, brier_weight=1., score_weight=1., min_delta=1e-4)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            resumed, continuous = Path(tmp) / 'resumed', Path(tmp) / 'continuous'
            train_candidate('ce_brier', config, arrays, y, splits, args, resumed, torch.device('cpu'))
            args.epochs = 2
            train_candidate('ce_brier', config, arrays, y, splits, args, resumed, torch.device('cpu'))
            train_candidate('ce_brier', config, arrays, y, splits, args, continuous, torch.device('cpu'))
            first = torch.load(resumed / 'ce_brier/latest.pt', weights_only=True)
            second = torch.load(continuous / 'ce_brier/latest.pt', weights_only=True)
        for key in first['model']:
            torch.testing.assert_close(first['model'][key], second['model'][key], rtol=0, atol=0)
        self.assertEqual(first['history'], second['history'])


if __name__ == '__main__':
    unittest.main()
