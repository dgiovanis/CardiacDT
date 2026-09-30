"""Synthetic regression tests; no private research arrays are needed."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from cardiac_dt.data import classify_arrhythmia, load_data, validate_pair
from cardiac_dt.plom import (
    ConditionalSecondOrderMoment,
    ISDESolver,
    MinMaxScaler,
    PCA_PLom,
)
from cardiac_dt.workflow import Config, conditional_validation, fit_generate, run


def synthetic_data():
    rng = np.random.default_rng(21)
    inputs = rng.uniform(0.5, 1.5, (60, 16))
    t = np.linspace(0, 1, 100)
    outputs = inputs[:, :1] * np.sin(2 * np.pi * t) + inputs[:, 1:2] * np.cos(
        4 * np.pi * t
    )
    outputs[:, -50:] = 0
    return inputs, outputs


class DataTests(unittest.TestCase):
    def test_tail_classification_matches_notebook_loop(self):
        x = np.zeros((5, 80))
        x[1] = 0.8
        x[2] = np.tile([-1, 1], 40)
        x[3, :30] = 20  # Early activity does not affect the tail rule.
        x[4] = 0.69
        expected = {"arrhythmia": [], "nonarrhythmia": []}
        for i, row in enumerate(x):
            tail = row[-50:]
            name = (
                "nonarrhythmia"
                if np.std(tail) < 0.1 and abs(np.mean(tail)) < 0.7
                else "arrhythmia"
            )
            expected[name].append(i)
        for key, actual in classify_arrhythmia(x).items():
            np.testing.assert_array_equal(actual, expected[key])

    def test_shape_and_finite_checks(self):
        x, y = synthetic_data()
        for bad_x, bad_y in (
            (x.T, y),
            (x[:-1], y),
            (x, y * np.nan),
            (x.astype(complex), y),
        ):
            with self.assertRaises(ValueError):
                validate_pair(bad_x, bad_y)

    def test_pickle_arrays_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            np.save(Path(tmp) / "tof_input_data.npy", np.array([{}], dtype=object))
            np.save(Path(tmp) / "tof_output_data.npy", np.ones((2, 4)))
            with self.assertRaises(ValueError):
                load_data(tmp)


class NumericalTests(unittest.TestCase):
    def test_scaling_and_whitened_pca_round_trip(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(4, 40))
        x[0] = 3  # Constant input is handled by scaling and rank reduction.
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(x)
        pca = PCA_PLom(error_PCA=1e-10, verbose=False).fit(scaled)
        eta = pca.transform_plom(scaled)
        np.testing.assert_allclose(np.cov(eta), np.eye(pca.nu), atol=1e-12)
        np.testing.assert_allclose(
            scaler.inverse_transform(pca.inverse_transform_plom(eta)), x, atol=1e-12
        )

    def test_conditional_moments_against_explicit_kernel_formula(self):
        w = np.array([[-2.0, -1.0, 1.0, 2.0]])
        q = np.array([[1.0, 2.0, 4.0, 5.0], [3.0, 3.0, 3.0, 3.0]])
        model = ConditionalSecondOrderMoment(bandwidth=0.5).fit(q, w)
        mean, second = model.predict(np.array([0.2]))
        weights = np.exp(-0.5 * ((w[0] - 0.2) / w.std(ddof=1) / 0.5) ** 2)
        weights /= weights.sum()
        expected_mean = q @ weights
        scale = q.std(axis=1, ddof=1)
        scale[scale == 0] = (
            1  # Preserve the source estimator's constant feature convention.
        )
        expected_second = (q**2) @ weights + scale**2 * 0.5**2
        np.testing.assert_allclose(mean, expected_mean)
        np.testing.assert_allclose(second, expected_second)

    def test_isde_serial_and_parallel_agree(self):
        eta = np.random.default_rng(1).normal(size=(3, 12))
        params = dict(
            nu=3,
            n_d=12,
            MatReta_d=eta,
            nbMC=2,
            M0transient=3,
            mode="full",
            random_state=1,
            verbose=False,
        )
        serial, _ = ISDESolver(**params).solve(n_jobs=1)
        parallel, _ = ISDESolver(**params).solve(n_jobs=2)
        np.testing.assert_array_equal(serial, parallel)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs, cls.outputs = synthetic_data()
        cls.config = Config(
            n_eigenpairs=10,
            intrinsic_dim=3,
            lifting_eigenpairs=8,
            n_mc=2,
            transient_steps=3,
        )
        cls.result = fit_generate(cls.inputs, cls.outputs, cls.config)

    def test_sample_dimensions_and_validation(self):
        self.assertEqual(self.result["inputs"].shape, (120, 16))
        self.assertEqual(self.result["outputs"].shape, (120, 100))
        val = conditional_validation(
            self.result, self.result["inputs"][:3], self.outputs[:3]
        )
        self.assertEqual(val["mean"].shape, (3, 100))
        self.assertTrue(np.isfinite(val["mean"]).all())
        self.assertTrue((val["std"] >= 0).all())
        self.assertEqual(
            len(set(self.result["train_indices"]) & set(self.result["test_indices"])), 0
        )

    def test_repeatability_and_rng_isolation(self):
        np.random.seed(123)
        before = np.random.get_state()
        other = fit_generate(self.inputs, self.outputs, self.config)
        after = np.random.get_state()
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        np.testing.assert_allclose(
            other["outputs"], self.result["outputs"], rtol=1e-10, atol=1e-10
        )

    def test_missing_class_and_invalid_configuration(self):
        with self.assertRaisesRegex(ValueError, "sample count"):
            fit_generate(
                self.inputs, self.outputs, replace(self.config, case="arrhythmia")
            )
        with self.assertRaises(ValueError):
            Config(n_mc=0)
        with self.assertRaises(ValueError):
            Config(intrinsic_dim=38)

    def test_saved_artifacts_and_overwrite_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.save(root / "tof_input_data.npy", self.inputs)
            np.save(root / "tof_output_data.npy", self.outputs)
            np.save(
                root / "input_for_validation_not_arrythmia.npy",
                self.result["inputs"][:3],
            )
            np.save(root / "output_for_validation_not_arrythmia.npy", self.outputs[:3])
            dest = root / "results"
            run(root, dest, self.config, validate=True, plots=False)
            with np.load(dest / "samples.npz", allow_pickle=False) as arrays:
                self.assertEqual(arrays["inputs"].shape, (120, 16))
            info = json.loads((dest / "run.json").read_text())
            self.assertIn("validation_rmse_mean", info["metrics"])
            self.assertTrue((dest / "validation.npz").is_file())
            with self.assertRaisesRegex(ValueError, "not empty"):
                run(root, dest, self.config, plots=False)


if __name__ == "__main__":
    unittest.main()
