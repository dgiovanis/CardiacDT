"""Protect paper-specific choices from silent regression."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from datafold.pcfold import GaussianKernel

from cardiac_dt.data import load_data
from cardiac_dt.reproduce import compare_value, nearest_distances
from cardiac_dt.workflow import apply_floor, conditional_validation


class PaperContractTests(unittest.TestCase):
    def test_reference_source_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / "input_for_validation_arrythmia.npy", np.ones((2, 16)))
            np.save(root / "output_for_validation_arrythmia.npy", np.full((2, 3), 4.0))
            np.save(root / "manifold_arrhythmia_sims.npy", np.full((2, 3), 99.0))
            _, reference = load_data(root, "arrhythmia")
            np.testing.assert_array_equal(reference, np.full((2, 3), 4.0))

    def test_floor_has_no_upper_ceiling_and_does_not_mutate(self):
        raw = np.array([-150.0, -90.0, 100.0])
        np.testing.assert_array_equal(apply_floor(raw, -90), [-90, -90, 100])
        np.testing.assert_array_equal(raw, [-150, -90, 100])
        np.testing.assert_array_equal(apply_floor(raw, None), raw)

    def test_validation_metrics_use_saved_floored_bands(self):
        inputs = np.repeat(np.linspace(-1, 1, 20)[:, None], 16, axis=1)
        responses = np.c_[np.linspace(-120, 20, 20), np.linspace(-20, 40, 20)]
        result = {"inputs": inputs, "outputs": apply_floor(responses, -90)}
        references = np.array([[-130.0, 10.0], [-80.0, 30.0]])
        val = conditional_validation(result, inputs[[4, 14]], references)
        np.testing.assert_array_equal(val["reference"], [[-90, 10], [-80, 30]])
        self.assertTrue((val["lower"] >= -90).all())
        np.testing.assert_allclose(
            val["rmse_per_sample"],
            np.sqrt(np.mean((val["reference"] - val["mean"]) ** 2, axis=1)),
        )
        np.testing.assert_array_equal(
            val["coverage_per_sample"],
            np.mean(
                (val["reference"] >= val["lower"]) & (val["reference"] <= val["upper"]),
                axis=1,
            ),
        )

    def test_published_rounding_tolerance_detects_mismatch(self):
        self.assertTrue(compare_value("RMSE", 32.4375, 32.44, 0.005)["passed"])
        self.assertFalse(compare_value("RMSE", 47.2033, 32.44, 0.005)["passed"])
        self.assertFalse(compare_value("RMSE", float("nan"), 32.44, 0.005)["passed"])

    def test_duplicate_detection_uses_euclidean_distance(self):
        actual = nearest_distances(
            np.array([[0.0, 0.0], [3.0, 4.0], [0.0, 2.0]]),
            np.array([[0.0, 0.0]]),
            chunk_size=1,
        )
        np.testing.assert_array_equal(actual, [0.0, 5.0, 2.0])

    def test_actual_datafold_kernel_bandwidth_convention(self):
        # The supplied PDF writes different denominator conventions. Keep the
        # actual implementation explicit until the manuscript is reconciled.
        points = np.array([[0.0, 0.0], [3.0, 4.0]])
        kernel = GaussianKernel(epsilon=5)(points)
        self.assertAlmostEqual(kernel[0, 1], np.exp(-25 / (2 * 5)))


if __name__ == "__main__":
    unittest.main()
