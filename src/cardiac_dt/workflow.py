"""Notebook-derived diffusion maps, lifting, PLoM, and validation pipeline."""

import json
import platform
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
from datafold.dynfold import (
    DiffusionMaps,
    GeometricHarmonicsInterpolator,
    LocalRegressionSelection,
)
from datafold.pcfold import GaussianKernel, PCManifold
from scipy.spatial.distance import pdist
from sklearn.model_selection import train_test_split

from .data import classify_arrhythmia, load_data, validate_pair
from .plom import (
    ConditionalSampler,
    ConditionalSecondOrderMoment,
    ISDESolver,
    MinMaxScaler,
    PCA_PLom,
)


@dataclass(frozen=True)
class Config:
    """Revised-paper settings, with serial execution for portability."""

    case: str = "nonarrhythmia"
    n_eigenpairs: int = 38
    intrinsic_dim: int = 15
    diffusion_scale: float = 30.0
    selection_scale: float = 6.0
    lifting_eigenpairs: int = 30
    lifting_scale: float = 50.0
    train_size: float = 0.8
    split_seed: int = 7
    pca_error: float = 1e-5
    n_mc: int = 100
    transient_steps: int = 50
    f0: float = 0.01
    time_step_coefficient: float = 5500.0
    seed: int = 1
    n_jobs: int = 1
    duration_ms: float = 2999.0
    voltage_floor: float | None = -90.0

    def __post_init__(self):
        if self.case not in ("arrhythmia", "nonarrhythmia"):
            raise ValueError("case must be arrhythmia or nonarrhythmia")
        for name in (
            "n_eigenpairs",
            "intrinsic_dim",
            "lifting_eigenpairs",
            "n_mc",
            "transient_steps",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 2 <= self.intrinsic_dim < self.n_eigenpairs - 1:
            raise ValueError("Require 2 <= intrinsic_dim < n_eigenpairs - 1")
        if not 0 < self.train_size < 1 or not 0 < self.pca_error < 1:
            raise ValueError("train_size and pca_error must be between zero and one")
        for name in (
            "diffusion_scale",
            "selection_scale",
            "lifting_scale",
            "f0",
            "time_step_coefficient",
            "duration_ms",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.voltage_floor is not None and not np.isfinite(self.voltage_floor):
            raise ValueError("voltage_floor must be finite or None")
        if self.n_jobs != -1 and self.n_jobs < 1:
            raise ValueError("n_jobs must be -1 or a positive integer")


def fit_generate(inputs, outputs, config=Config()):
    """Fit the joint manifold and return generated samples and diagnostics.

    The embedding sees the entire class, as in the notebook. The split tests
    geometric-harmonics reconstruction within that embedding, not end-to-end
    generalization to unseen simulations. The revised-paper analysis applies
    a -90 mV response floor after lifting; set voltage_floor=None to disable it.
    """
    validate_pair(inputs, outputs)
    selected = classify_arrhythmia(outputs)[config.case]
    joint = np.concatenate((inputs[selected], outputs[selected]), axis=1)
    n_train = int(config.train_size * len(joint))
    if config.n_eigenpairs >= len(joint):
        raise ValueError(
            f"n_eigenpairs must be below the class sample count ({len(joint)})"
        )
    if config.lifting_eigenpairs >= n_train:
        raise ValueError(
            f"lifting_eigenpairs must be below the training count ({n_train})"
        )

    # The original eigensolver may use NumPy's global RNG. Scope its seed and
    # the legacy ISDE seed to this call, restoring the caller's state afterward.
    state = np.random.get_state()
    try:
        np.random.seed(config.seed)
        cloud = PCManifold(joint)
        cloud.optimize_parameters()
        epsilon = float(cloud.kernel.epsilon * config.diffusion_scale)
        dmap = DiffusionMaps(
            kernel=GaussianKernel(
                epsilon=epsilon, distance=dict(cut_off=cloud.cut_off)
            ),
            n_eigenpairs=config.n_eigenpairs,
            is_stochastic=True,
            time_exponent=0,
            alpha=1,
        ).fit(cloud)
        selection = LocalRegressionSelection(
            intrinsic_dim=config.intrinsic_dim,
            n_subsample=len(joint),
            strategy="dim",
            eps_med_scale=config.selection_scale,
        ).fit(dmap.eigenvectors_)
        coordinates = dmap.eigenvectors_[:, selection.evec_indices_]
        train, test = train_test_split(
            np.arange(len(joint)),
            train_size=config.train_size,
            random_state=config.split_seed,
        )
        lifting_epsilon = float(
            np.median(pdist(coordinates[train])) ** 2 * config.lifting_scale
        )
        if not np.isfinite(lifting_epsilon) or lifting_epsilon <= 0:
            raise ValueError(
                "Degenerate diffusion coordinates: lifting bandwidth is not positive"
            )
        lifting = GeometricHarmonicsInterpolator(
            GaussianKernel(epsilon=lifting_epsilon),
            n_eigenpairs=config.lifting_eigenpairs,
            is_stochastic=False,
            alpha=1,
        ).fit(coordinates[train], joint[train])
        predicted_train = lifting.predict(coordinates[train])
        predicted_test = lifting.predict(coordinates[test])
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(coordinates.T)
        pca = PCA_PLom(error_PCA=config.pca_error, verbose=False).fit(scaled)
        if np.any(pca.eigenvalues <= 0):
            raise ValueError(
                "PLoM whitening requires positive retained PCA eigenvalues"
            )
        solver = ISDESolver(
            nu=pca.nu,
            n_d=len(joint),
            MatReta_d=pca.transform_plom(scaled),
            f0=config.f0,
            nbMC=config.n_mc,
            coeffDeltar=config.time_step_coefficient,
            M0transient=config.transient_steps,
            mode="full",
            random_state=config.seed,
            verbose=False,
        )
        generated, _ = solver.solve(n_jobs=config.n_jobs)
        reconstructed = scaler.inverse_transform(pca.inverse_transform_plom(generated))
        lifted = lifting.predict(reconstructed.T)
    finally:
        np.random.set_state(state)
    if not np.isfinite(lifted).all():
        raise ValueError("Generated samples contain non-finite values")
    metrics = {
        "class_samples": len(joint),
        "generated_samples": len(lifted),
        "pca_dimension": int(pca.nu),
        "diffusion_epsilon_base": float(cloud.kernel.epsilon),
        "diffusion_epsilon": epsilon,
        "diffusion_exponent": 0,
        "stochastic_step": float(solver.Deltar),
        "conditional_sampling_bandwidth": float(
            3 * ConditionalSampler(W=lifted[:, :16]).silverman_bandwidth()
        ),
        "floored_response_values": int(
            np.count_nonzero(lifted[:, 16:] < config.voltage_floor)
        )
        if config.voltage_floor is not None
        else 0,
        "lifting_epsilon": lifting_epsilon,
        "lifting_train_mse": float(np.mean((joint[train] - predicted_train) ** 2)),
        "lifting_test_mse": float(np.mean((joint[test] - predicted_test) ** 2)),
        "lifting_test_squared_error_std": float(
            np.std((joint[test] - predicted_test) ** 2)
        ),
    }
    return {
        "inputs": lifted[:, :16],
        "outputs": apply_floor(lifted[:, 16:], config.voltage_floor),
        "raw_outputs": lifted[:, 16:],
        "generated_coordinates": reconstructed.T,
        "lifting_eigenvalues": lifting.eigenvalues_,
        "source_indices": selected,
        "train_indices": selected[train],
        "test_indices": selected[test],
        "eigenvalues": dmap.eigenvalues_,
        "selection_residuals": selection.residuals_,
        "selected_eigenvectors": selection.evec_indices_,
        "coordinates": coordinates,
        "metrics": metrics,
    }


def apply_floor(values, floor):
    """Apply only the lower bound used by the revision analysis (no upper clip)."""
    return np.asarray(values) if floor is None else np.maximum(values, floor)


def conditional_validation(result, inputs, outputs, voltage_floor=-90.0):
    """Evaluate moments against each supplied reference row.

    File selection follows the revision analysis, not verified simulator-export
    provenance. Raw and floored reference arrays are both retained.
    """
    validate_pair(inputs, outputs)
    if outputs.shape[1] != result["outputs"].shape[1]:
        raise ValueError(
            "Validation and training responses must have the same time grid"
        )
    model = ConditionalSecondOrderMoment().fit(result["outputs"].T, result["inputs"].T)
    moments = [model.predict(row) for row in inputs]
    means = np.array([pair[0] for pair in moments])
    std = np.sqrt(np.maximum(np.array([pair[1] for pair in moments]) - means**2, 0.0))
    means = apply_floor(means, voltage_floor)
    reference = apply_floor(outputs, voltage_floor)
    lower = apply_floor(means - std, voltage_floor)
    upper = apply_floor(means + std, voltage_floor)
    return {
        "inputs": inputs,
        "mean": means,
        "std": std,
        "reference": reference,
        "raw_reference": outputs,
        "lower": lower,
        "upper": upper,
        "bandwidth": np.asarray(model.sx),
        "rmse_per_sample": np.sqrt(np.mean((reference - means) ** 2, axis=1)),
        "coverage_per_sample": np.mean(
            (reference >= lower) & (reference <= upper), axis=1
        ),
    }


def run(data_dir, output_dir, config=Config(), *, validate=False, plots=True):
    """Run a reproducible experiment into a new or empty output directory."""
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Output directory is not empty; choose a new run directory")
    inputs, outputs = load_data(data_dir)
    validation_data = load_data(data_dir, config.case) if validate else None
    if validation_data is not None and validation_data[1].shape[1] != outputs.shape[1]:
        raise ValueError(
            "Validation and training responses must have the same time grid"
        )
    result = fit_generate(inputs, outputs, config)
    validation = (
        conditional_validation(
            result, *validation_data, voltage_floor=config.voltage_floor
        )
        if validate
        else None
    )
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination / "samples.npz",
        **{k: v for k, v in result.items() if k != "metrics"},
    )
    if validation is not None:
        np.savez_compressed(destination / "validation.npz", **validation)
        result["metrics"]["validation_rmse_mean"] = float(
            validation["rmse_per_sample"].mean()
        )
        result["metrics"]["validation_coverage_mean"] = float(
            validation["coverage_per_sample"].mean()
        )
        result["metrics"]["conditional_moment_bandwidth"] = float(
            validation["bandwidth"]
        )
    metadata = {
        "config": asdict(config),
        "metrics": result["metrics"],
        "python": platform.python_version(),
        "versions": {
            name: version(name)
            for name in (
                "cardiac-dt",
                "numpy",
                "scipy",
                "datafold",
                "scikit-learn",
                "matplotlib",
            )
        },
        "input_shapes": {"inputs": list(inputs.shape), "outputs": list(outputs.shape)},
    }
    (destination / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if plots:
        from .plots import save_figures

        save_figures(inputs, outputs, result, config, destination, validation)
    return result
