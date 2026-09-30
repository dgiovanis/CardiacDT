"""Command-line entry point; no computation happens on import."""

import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Cardiac diffusion maps / PLoM research workflow"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing the documented NPY files",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="New or empty results directory"
    )
    parser.add_argument(
        "--case", choices=("arrhythmia", "nonarrhythmia"), default="nonarrhythmia"
    )
    parser.add_argument(
        "--n-mc", type=int, default=100, help="Monte Carlo trajectories (default: 100)"
    )
    parser.add_argument("--transient-steps", type=int, default=50)
    parser.add_argument(
        "--n-jobs", type=int, default=1, help="Worker processes; -1 uses all CPUs"
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Load the revision reference arrays (see the paper audit)",
    )
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    from .workflow import Config, run

    config_args = {
        key: getattr(args, key)
        for key in ("case", "n_mc", "transient_steps", "n_jobs", "seed")
    }
    try:
        result = run(
            args.data_dir,
            args.output_dir,
            Config(**config_args),
            validate=args.validate,
            plots=not args.no_plots,
        )
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(
        f"Saved {result['metrics']['generated_samples']} generated samples to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
