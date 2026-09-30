"""Explicit, pickle-free loading and validation of research arrays."""

from pathlib import Path

import numpy as np

VALIDATION_FILES = {
    "arrhythmia": (
        "input_for_validation_arrythmia.npy",
        "output_for_validation_arrythmia.npy",
    ),
    "nonarrhythmia": (
        "input_for_validation_not_arrythmia.npy",
        "output_for_validation_not_arrythmia.npy",
    ),
}


def validate_pair(inputs, outputs):
    """Require finite, samples-first arrays with 16 cardiac input parameters."""
    for name, values in (("inputs", inputs), ("outputs", outputs)):
        if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
            raise ValueError(f"{name} must be a 2D array with at least two samples")
        if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
            raise ValueError(f"{name} must contain real numbers")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains NaN or infinite values")
    if inputs.shape[0] != outputs.shape[0] or inputs.shape[1] != 16:
        raise ValueError("Expected matching sample counts and exactly 16 input columns")
    return inputs, outputs


def load_data(directory, case=None):
    """Load training data, or revision reference data for a given class."""
    names = (
        ("tof_input_data.npy", "tof_output_data.npy")
        if case is None
        else VALIDATION_FILES[case]
    )
    arrays = [np.load(Path(directory) / name, allow_pickle=False) for name in names]
    return validate_pair(*arrays)


def classify_arrhythmia(samples, tail_length=50, flat_tol=0.1, zero_tol=0.7):
    """Apply the notebook's tail heuristic; this is not a clinical classifier.

    Non-VT: the final 50 values have std < 0.1 and abs(mean) < 0.7.
    All other traces are assigned VT. Thresholds use the supplied data units.
    """
    samples = np.asarray(samples)
    if samples.ndim != 2 or samples.shape[1] < 1 or not np.isfinite(samples).all():
        raise ValueError("Expected finite samples-first response traces")
    if tail_length < 1 or flat_tol <= 0 or zero_tol <= 0:
        raise ValueError("Tail length and tolerances must be positive")
    tail = samples[:, -tail_length:]
    non_vt = (np.std(tail, axis=1) < flat_tol) & (
        np.abs(np.mean(tail, axis=1)) < zero_tol
    )
    return {
        "arrhythmia": np.flatnonzero(~non_vt),
        "nonarrhythmia": np.flatnonzero(non_vt),
    }
