"""Run both paper regimes and verify Tables 1/3 and the duplication diagnostic."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

from .data import load_data
from .reference import DATA_SHA256, PAPER_TABLE1, PAPER_TABLE3, RETAINED_INDICES
from .workflow import Config, run


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compare_value(label, actual, expected, tolerance=0):
    """Absolute tolerances reflect the last reported digit, not fitted acceptance."""
    return {
        "check": label,
        "actual": float(actual),
        "expected": float(expected),
        "absolute_tolerance": tolerance,
        "passed": bool(np.isfinite(actual) and abs(actual - expected) <= tolerance),
    }


def nearest_distances(generated, training, chunk_size=256):
    """Compute raw Euclidean nearest-neighbor distances without a huge matrix."""
    return np.concatenate(
        [
            cdist(generated[i : i + chunk_size], training).min(axis=1)
            for i in range(0, len(generated), chunk_size)
        ]
    )


def reproduce(data_dir, output_dir, *, plots=True, n_jobs=1):
    """Reproduce published numerical claims; never substitute saved predictions."""
    source, destination = Path(data_dir), Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Output directory must be new or empty")
    inputs, outputs = load_data(source)
    # Require every revision-reference file before fitting anything.
    for case in PAPER_TABLE3:
        load_data(source, case)
    destination.mkdir(parents=True, exist_ok=True)
    checks = []
    for filename, expected_hash in DATA_SHA256.items():
        digest = hashlib.sha256((source / filename).read_bytes()).hexdigest()
        checks.append(
            {
                "check": f"data:{filename}",
                "actual": digest,
                "expected": expected_hash,
                "passed": digest == expected_hash,
            }
        )
    results, summaries, per_case, diversity = {}, [], [], []
    for case in PAPER_TABLE3:
        config = Config(case=case, n_jobs=n_jobs)
        result = run(source, destination / case, config, validate=True, plots=plots)
        results[case] = result
        metrics = result["metrics"]
        with np.load(
            destination / case / "validation.npz", allow_pickle=False
        ) as archive:
            validation = dict(archive)
        for i, (rmse, coverage) in enumerate(
            zip(validation["rmse_per_sample"], validation["coverage_per_sample"]), 1
        ):
            per_case.append(
                {
                    "regime": case,
                    "case": i,
                    "rmse_mV": float(rmse),
                    "coverage": float(coverage),
                }
            )
        summaries.append(
            {
                "group": case,
                "n": len(validation["mean"]),
                "rmse_mV": metrics["validation_rmse_mean"],
                "coverage": metrics["validation_coverage_mean"],
            }
        )
        actual_table3 = dict(
            metrics,
            candidate_eigenpairs=config.n_eigenpairs,
            intrinsic_dimension=config.intrinsic_dim,
            lifting_modes=len(result["lifting_eigenvalues"]),
            damping=config.f0,
            transient_steps=config.transient_steps,
            monte_carlo_trajectories=config.n_mc,
        )
        for name, (expected, tolerance) in PAPER_TABLE3[case].items():
            checks.append(
                compare_value(
                    f"{case}:{name}", actual_table3[name], expected, tolerance
                )
            )
        checks.append(
            {
                "check": f"{case}:retained_indices",
                "actual": result["selected_eigenvectors"].tolist(),
                "expected": RETAINED_INDICES[case],
                "passed": result["selected_eigenvectors"].tolist()
                == RETAINED_INDICES[case],
            }
        )
        original = np.c_[
            inputs[result["source_indices"]], outputs[result["source_indices"]]
        ]
        generated = np.c_[result["inputs"], result["outputs"]]
        distances = {}
        for block, sl in (
            ("W", slice(0, 16)),
            ("R", slice(16, None)),
            ("joint", slice(None)),
        ):
            d = nearest_distances(generated[:, sl], original[:, sl])
            distances[block] = d
            duplicates = int(np.count_nonzero(d <= 1e-12))
            diversity.append(
                {
                    "regime": case,
                    "block": block,
                    "duplicates": duplicates,
                    "tolerance": 1e-12,
                    "minimum": float(d.min()),
                    "median": float(np.median(d)),
                }
            )
            checks.append(compare_value(f"{case}:duplicates:{block}", duplicates, 0))
        np.savez_compressed(destination / case / "nearest_distances.npz", **distances)
    summaries.append(
        {
            "group": "all",
            "n": len(per_case),
            "rmse_mV": float(np.mean([r["rmse_mV"] for r in per_case])),
            "coverage": float(np.mean([r["coverage"] for r in per_case])),
        }
    )
    for row in summaries:
        expected = PAPER_TABLE1[row["group"]]
        for metric, tolerance in (("n", 0), ("rmse_mV", 0.005), ("coverage", 0.0005)):
            checks.append(
                compare_value(
                    f"Table1:{row['group']}:{metric}",
                    row[metric],
                    expected[metric],
                    tolerance,
                )
            )
    write_csv(destination / "table1.csv", summaries)
    write_csv(destination / "simulator_cases.csv", per_case)
    write_csv(destination / "diversity.csv", diversity)
    write_csv(
        destination / "table3.csv",
        [
            c
            for c in checks
            if "absolute_tolerance" in c and not c["check"].startswith("Table1")
        ],
    )
    report = {
        "numerical_reproduction_passed": all(c["passed"] for c in checks),
        "full_paper_consistency": False,
        "simulator_reference_provenance_verified": False,
        "input_column_schema_verified": False,
        "scope": "Tables 1/3, coordinate indices, sample counts, and duplicate diagnostic",
        "remaining_manuscript_issues": "See docs/PAPER_AUDIT.md: reference-array provenance, input-column labels, kernel notation, undocumented flooring, conditioning metric, time/latent-panel provenance, and unavailable upstream simulation inputs.",
        "checks": checks,
    }
    (destination / "reproduction_check.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    if plots:
        from .paper_plots import save_paper_figures

        save_paper_figures(inputs, outputs, results, destination)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        report = reproduce(
            args.data_dir, args.output_dir, plots=not args.no_plots, n_jobs=args.n_jobs
        )
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f"error: {error}\n")
    if not report["numerical_reproduction_passed"]:
        failed = [item["check"] for item in report["checks"] if not item["passed"]]
        parser.exit(1, f"Paper numerical checks FAILED: {', '.join(failed)}\n")
    print(
        f"PASS: {len(report['checks'])} paper numerical checks. Results: {args.output_dir}"
    )
    print(
        "This is numerical reproduction, not full manuscript consistency; see docs/PAPER_AUDIT.md."
    )


if __name__ == "__main__":
    main()
