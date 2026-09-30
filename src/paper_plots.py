"""Cross-regime paper diagnostics and deterministic conditional examples.

These recreate diagnostic content, not the original page layouts or unknown
figure-specific query selections. Example indices and RNG seeds are saved.
"""

import json

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .plom import ConditionalSampler
from .workflow import Config, apply_floor, conditional_validation


def save_paper_figures(inputs, outputs, results, directory):
    def figure(rows=1, cols=1, size=(12, 5), **kwargs):
        fig = Figure(figsize=size, layout="constrained")
        FigureCanvasAgg(fig)
        return fig, fig.subplots(rows, cols, squeeze=False, **kwargs)

    def save(fig, name, parent=directory):
        fig.savefig(parent / f"{name}.png", dpi=200)
        fig.savefig(parent / f"{name}.pdf")
        fig.clear()

    cases = list(results)
    labels = {"arrhythmia": "VT", "nonarrhythmia": "Non-VT"}
    time = np.linspace(0, Config().duration_ms, outputs.shape[1], endpoint=False)
    for mode in ("spectrum", "selection", "latent_spectrum"):
        fig, axes = figure(1, 2)
        for ax, case in zip(axes.flat, cases):
            result = results[case]
            if mode == "spectrum":
                ax.semilogy(result["eigenvalues"], "o--")
                ax.set(xlabel="Eigenvalue index", ylabel="Ambient eigenvalue")
            elif mode == "selection":
                idx = np.arange(len(result["selection_residuals"]))
                ax.bar(
                    idx,
                    result["selection_residuals"],
                    color=[
                        "firebrick"
                        if i in result["selected_eigenvectors"]
                        else "royalblue"
                        for i in idx
                    ],
                )
                ax.set(xlabel="Eigenvector index", ylabel="Local regression residual")
            else:
                spectrum = result["lifting_eigenvalues"]
                ax.semilogy(spectrum / spectrum[0], "o--")
                ax.set(xlabel="Latent-kernel mode", ylabel="Normalized eigenvalue")
            ax.set_title(labels[case])
        save(
            fig,
            {
                "spectrum": "figure16_ambient_spectrum",
                "selection": "figure17_coordinate_selection",
                "latent_spectrum": "figure19_latent_spectrum",
            }[mode],
        )

    fig, axes = figure(1, 2)
    for ax, metric, title in zip(
        axes.flat,
        ("diffusion_epsilon_base", "lifting_epsilon"),
        ("Ambient optimized base bandwidth", "Latent bandwidth: 50 × median distance²"),
    ):
        ax.bar(
            [labels[c] for c in cases], [results[c]["metrics"][metric] for c in cases]
        )
        ax.set_title(title)
    save(fig, "figure18_bandwidths")

    fig, axes = figure(1, 3, (16, 5))
    for ax, block in zip(axes.flat, ("W", "R", "joint")):
        for case in cases:
            with np.load(directory / case / "nearest_distances.npz") as data:
                ax.hist(
                    data[block], bins=40, density=True, alpha=0.5, label=labels[case]
                )
        ax.set(
            title=block, xlabel="Nearest original-training distance", ylabel="Density"
        )
        ax.legend()
    save(fig, "figure05_sample_diversity")

    manifest = {}
    for case in cases:
        result, parent = results[case], directory / case
        selected = result["source_indices"]
        example_rows = np.random.default_rng(2024).choice(
            len(selected), size=4, replace=False
        )
        original_rows = selected[example_rows]
        validation = conditional_validation(
            result, inputs[original_rows], outputs[original_rows]
        )
        fig, axes = figure(2, 2, (12, 8), sharex=True, sharey=True)
        for ax, i, original_row in zip(axes.flat, range(4), original_rows):
            ax.plot(
                time, validation["mean"][i], color="royalblue", label="Conditional mean"
            )
            ax.fill_between(
                time,
                validation["lower"][i],
                validation["upper"][i],
                alpha=0.25,
                label="Floored mean ±1 std",
            )
            ax.set(
                title=f"{labels[case]} input row {original_row}",
                xlabel="Time (ms)",
                ylabel="Voltage (mV)",
            )
        axes[0, 0].legend()
        save(fig, "conditional_training_moments", parent)

        # Q is explicitly the generated diffusion coordinates. The legacy
        # notebook defaulted Q=R.T and mislabeled response panels as latent.
        sampler = ConditionalSampler(
            W=result["inputs"],
            R=result["outputs"],
            Q=result["generated_coordinates"].T,
            random_state=1,
        )
        bandwidth = 3 * sampler.silverman_bandwidth()
        fig, axes = figure(2, 2, (12, 8), sharex=True, sharey=True)
        sample_indices = []
        for ax, original_row in zip(axes.flat, original_rows):
            samples, indices, _ = sampler.conditional_samples(
                inputs[original_row],
                bandwidth,
                n_samples=3,
                random_state=1000 + int(original_row),
                return_info=True,
            )
            sample_indices.append(indices.tolist())
            ax.plot(
                time,
                apply_floor(outputs[original_row], -90),
                color="black",
                label="Reference",
            )
            ax.plot(time, samples.T, "--", color="firebrick", alpha=0.7)
            ax.set(
                title=f"{labels[case]} input row {original_row}",
                xlabel="Time (ms)",
                ylabel="Voltage (mV)",
            )
        axes[0, 0].plot([], [], "--", color="firebrick", label="Conditional samples")
        axes[0, 0].legend()
        save(fig, "conditional_training_samples", parent)

        query = inputs[selected[0]]
        responses, response_indices, _ = sampler.conditional_samples(
            query, bandwidth, n_samples=30, random_state=1, return_info=True
        )
        q_samples = sampler.sample_q_given_w(
            query, bandwidth, n_draws=30, random_state=1
        )
        q_gaussian = sampler.sample_q_given_w_local_gaussian(
            query, bandwidth, n_draws=30, random_state=1
        )
        q_mean, q_cov = (
            sampler.conditional_mean_q(query, bandwidth),
            sampler.conditional_cov_q(query, bandwidth),
        )
        fig, axes = figure(2, 2, (12, 8))
        axes[0, 0].plot(time, responses.T, color="royalblue", alpha=0.15)
        axes[0, 0].plot(
            time, sampler.conditional_expectation(query, bandwidth), color="black"
        )
        axes[0, 0].set(
            title="Conditional response ensemble",
            xlabel="Time (ms)",
            ylabel="Voltage (mV)",
        )
        axes[0, 1].scatter(q_samples[0], q_samples[1], label="Empirical")
        axes[0, 1].scatter(
            q_gaussian[0], q_gaussian[1], marker="x", label="Local Gaussian"
        )
        axes[0, 1].set(xlabel="Diffusion coordinate 1", ylabel="Diffusion coordinate 2")
        axes[0, 1].legend()
        axes[1, 0].errorbar(
            np.arange(len(q_mean)), q_mean, yerr=np.sqrt(np.diag(q_cov)), fmt="o"
        )
        axes[1, 0].set(xlabel="Diffusion coordinate", ylabel="Conditional mean ± std")
        im = axes[1, 1].imshow(q_cov, cmap="coolwarm")
        axes[1, 1].set(
            title="Conditional diffusion-coordinate covariance",
            xlabel="Coordinate",
            ylabel="Coordinate",
        )
        fig.colorbar(im, ax=axes[1, 1])
        save(fig, "conditional_response_and_latent", parent)
        np.savez_compressed(
            parent / "conditional_examples.npz",
            query=query,
            response_samples=responses,
            response_sample_indices=response_indices,
            latent_samples=q_samples,
            latent_gaussian_samples=q_gaussian,
            latent_mean=q_mean,
            latent_covariance=q_cov,
        )
        manifest[case] = {
            "training_row_indices": original_rows.tolist(),
            "sample_indices": sample_indices,
            "selection_seed": 2024,
            "conditional_seed": 1,
            "fixed_query_training_row": int(selected[0]),
            "conditional_sampling_metric": "raw input Euclidean distance (legacy implementation)",
            "effective_sample_size": float(
                sampler.effective_sample_size(query, bandwidth)
            ),
            "snapshot_targets_ms": [70, 140, 340, 680],
            "snapshot_actual_ms": [
                float(time[np.argmin(abs(time - t))]) for t in [70, 140, 340, 680]
            ],
        }
    (directory / "figure_examples.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
