"""Publication-style diagnostics without interactive windows or global styles."""

import numpy as np
from scipy.stats import gaussian_kde

from .data import classify_arrhythmia

INPUT_NAMES = [
    rf"$I_{{\mathrm{{{current}}},{tissue}}}$"
    for tissue in ("h", "f")
    for current in ("Na", "CaL", "to", "K1", "Ks", "Kr")
] + [rf"$\sigma_{{\mathrm{{{name}}}}}$" for name in ("HL", "FL", "HT", "FT")]


def kde_bootstrap_band(values, grid, n_boot=200, random_state=0):
    """Notebook bootstrap KDE mean and std, skipping singular resamples."""
    values = np.asarray(values)
    rng = np.random.default_rng(random_state)
    densities = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        if np.ptp(sample) > 0:
            try:
                densities.append(gaussian_kde(sample)(grid))
            except np.linalg.LinAlgError:
                continue
    if not densities:
        raise ValueError("Cannot estimate a KDE from constant or singular samples")
    return np.mean(densities, axis=0), np.std(densities, axis=0)


def _density(ax, values, grid, color, label, seed):
    if len(values) < 2 or np.ptp(values) == 0:
        ax.axvline(values[0], color=color, label=label)
    else:
        mean, std = kde_bootstrap_band(values, grid, random_state=seed)
        ax.plot(grid, mean, color=color, label=label)
        ax.fill_between(grid, mean - std, mean + std, color=color, alpha=0.2)


def save_figures(inputs, outputs, result, config, directory, validation=None):
    """Save the notebook's main diagnostic figure families as PNG files."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    def figure(rows=1, cols=1, size=(8, 5), **kwargs):
        fig = Figure(figsize=size, layout="constrained")
        FigureCanvasAgg(fig)
        return fig, fig.subplots(rows, cols, squeeze=False, **kwargs)

    def save(fig, name):
        fig.savefig(directory / name, dpi=200)
        fig.clear()

    def density_grid(first, second, labels, name):
        fig, axes = figure(4, 4, (14, 12))
        for k, ax in enumerate(axes.flat):
            lo, hi = (
                min(first[:, k].min(), second[:, k].min()),
                max(first[:, k].max(), second[:, k].max()),
            )
            grid = np.linspace(lo, hi if hi > lo else lo + 1, 300)
            _density(ax, first[:, k], grid, "royalblue", labels[0], 123)
            _density(ax, second[:, k], grid, "firebrick", labels[1], 456)
            ax.set_xlabel(INPUT_NAMES[k])
        axes[0, 0].legend()
        save(fig, name)

    classes = classify_arrhythmia(outputs)
    if all(len(v) for v in classes.values()):
        density_grid(
            inputs[classes["arrhythmia"]],
            inputs[classes["nonarrhythmia"]],
            ("VT", "Non-VT"),
            "input_densities.png",
        )
    selected = result["source_indices"]
    density_grid(
        inputs[selected],
        result["inputs"],
        ("Reference", "Augmented"),
        "generated_input_densities.png",
    )
    time = np.linspace(0, config.duration_ms, outputs.shape[1], endpoint=False)
    fig, axes = figure(1, 2, (13, 5), sharey=True)
    for ax, (case, indices) in zip(axes.flat, classes.items()):
        if len(indices):
            traces = outputs[indices]
            ax.plot(time, traces.T, color="black", linewidth=0.1)
            mean = traces.mean(axis=0)
            ci = traces.std(axis=0)
            ax.plot(time, mean, "--", color="royalblue", label="Mean")
            ax.fill_between(
                time,
                mean - ci,
                mean + ci,
                alpha=0.25,
                label="Mean ±1 standard deviation",
            )
        ax.set(xlabel="Time (ms)", ylabel="Voltage (mV)", title=case)
    axes[0, 0].legend()
    save(fig, "reference_responses.png")

    fig, axes = figure(1, 2, (11, 4))
    axes[0, 0].semilogy(result["eigenvalues"], "o--")
    axes[0, 0].set(xlabel="Eigenvector index", ylabel="Eigenvalue")
    colors = [
        "firebrick" if k in result["selected_eigenvectors"] else "royalblue"
        for k in range(config.n_eigenpairs)
    ]
    axes[0, 1].bar(
        np.arange(config.n_eigenpairs), result["selection_residuals"], color=colors
    )
    axes[0, 1].set(xlabel="Eigenvector index", ylabel="Local regression residual")
    save(fig, "diffusion_diagnostics.png")

    fig, axes = figure()
    ax = axes[0, 0]
    for traces, label, color in (
        (outputs[selected], "Reference", "black"),
        (result["outputs"], "Augmented", "royalblue"),
    ):
        mean, std = traces.mean(axis=0), traces.std(axis=0)
        ax.plot(time, mean, label=label, color=color)
        ax.fill_between(time, mean - std, mean + std, color=color, alpha=0.15)
    ax.set(
        xlabel="Time (ms)",
        ylabel="Voltage (mV)",
        title="Means and ±1 sample standard deviation",
    )
    ax.legend()
    save(fig, "generated_responses.png")

    fig, axes = figure(2, 2, (10, 8))
    indices = [
        int(np.argmin(np.abs(time - target)))
        for target in (70, 140, 340, 680)
        if target <= time[-1]
    ]
    for ax, idx in zip(axes.flat, indices):
        for values, label, color in (
            (outputs[selected, idx], "Reference", "black"),
            (result["outputs"][:, idx], "Augmented", "royalblue"),
        ):
            if np.ptp(values) > 0:
                grid = np.linspace(values.min(), values.max(), 300)
                ax.plot(
                    grid,
                    gaussian_kde(values)(grid),
                    color=color,
                    label=label,
                    linestyle="--" if label == "Augmented" else "-",
                )
            else:
                ax.axvline(values[0], color=color, label=label)
        ax.set(xlabel="Voltage (mV)", ylabel="Density", title=f"t = {time[idx]:.2f} ms")
    for ax in list(axes.flat)[len(indices) :]:
        ax.set_visible(False)
    if indices:
        axes[0, 0].legend()
    save(fig, "response_densities.png")

    if validation is not None:
        indices = [i for i in (1, 6, 5, 9) if i < len(validation["mean"])]
        if not indices:
            indices = [0]
        fig, axes = figure(len(indices), 1, (10, 3 * len(indices)), sharex=True)
        for ax, idx in zip(axes.flat, indices):
            mean, std = validation["mean"][idx], validation["std"][idx]
            ax.plot(
                time, validation["reference"][idx], color="black", label="Reference"
            )
            ax.plot(time, mean, color="royalblue", label="Conditional mean")
            ax.fill_between(
                time,
                validation["lower"][idx],
                validation["upper"][idx],
                alpha=0.25,
                label="Floored mean ±1 std",
            )
            ax.set(
                xlabel="Time (ms)",
                ylabel="Voltage (mV)",
                title=f"Validation sample {idx}",
            )
        axes[0, 0].legend()
        save(fig, "conditional_validation.png")
