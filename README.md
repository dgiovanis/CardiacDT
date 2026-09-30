# Probabilistic cardiac digital twins

Research code for **Probabilistic Cardiac Digital Twins for Patient-Specific
Modeling**, by Dimitris G. Giovanis, Kelly Zhang, Justin Tso, Mauro Maggioni,
Ioannis G. Kevrekidis, and Natalia Trayanova.

## Installation

Use Python 3.10 or 3.11. From this repository directory:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The tested Python 3.11 environment is pinned in `requirements-tested.txt`;
install it before `pip install -e .` for those exact dependency versions.
Python 3.12 is excluded because the required scikit-learn release failed to
install in that environment. Optional notebook support:
`python -m pip install -e '.[notebook]'` followed by `jupyter lab`.

## Reproduce the numerical paper tables

Obtain the six documented arrays separately; see [data/README.md](data/README.md).
No research data or pickle files are distributed here. From the original local
repository location, `../data` points to the existing arrays.

```bash
cardiac-dt-reproduce --data-dir ../data --output-dir results/paper
```

Equivalently, use `python -m cardiac_dt.reproduce`. This command:

- Fits both regimes and generates 3900 VT and 6100 non-VT realizations.
- Checks the six array hashes, Table 3 values, retained coordinate indices,
  sample counts, and zero exact duplicates at tolerance `1e-12`.
- Recomputes all 20 reference comparisons and checks Table 1 to its published
  precision. Expected values are assertions, never inputs to model fitting.
- Writes `table1.csv`, `table3.csv`, `simulator_cases.csv`, `diversity.csv`,
  `reproduction_check.json`, per-regime arrays/metadata, and diagnostic figures.
- Returns a nonzero exit status if any numerical/data-identity check fails.
  A numerical pass does not clear the manuscript issues recorded separately
  as `full_paper_consistency: false` in the report.

Use `--no-plots` for numerical checks only and `--n-jobs` for multiple Monte
Carlo worker processes. Use a new or empty output directory on every run.

| Group | Reproduced RMSE (mV) | Reproduced temporal coverage |
| --- | ---: | ---: |
| All 20 cases | 22.523037 | 0.945275 |
| 10 VT cases | 32.437507 | 0.907388 |
| 10 non-VT cases | 12.608567 | 0.983162 |


## Individual runs and Python API

```bash
cardiac-dt --data-dir ../data --output-dir results/non-vt --validate
cardiac-dt --data-dir ../data --output-dir results/vt --case arrhythmia --validate
```

```python
from cardiac_dt.workflow import Config, run

result = run("../data", "results/custom", Config(case="nonarrhythmia"), validate=True)
```

The default configuration follows the revision analysis: 38 candidate diffusion
eigenpairs, 15 selected coordinates, ambient bandwidth multiplier 30, selection
scale 6, 30 GH modes, latent bandwidth multiplier 50, 80/20 lifting split with
seed 7, PCA tolerance `1e-5`, and full-space ISDE with 100 trajectories,
50 steps, damping 0.01, time-step coefficient 5500, and seed 1. Serial execution
is the default. Protect multiprocess Python scripts with
`if __name__ == '__main__':`; keep notebook runs serial.

The embedding sees the whole selected class before the lifting split. Lifting
MSE is a transductive reconstruction diagnostic mixing parameter and voltage
units, not end-to-end generalization error. 

## Outputs and figure scope

Each regime produces `samples.npz` with inputs, floored/raw outputs, original
row indices, diffusion coordinates, generated coordinates, and both spectra;
`validation.npz` with inputs, raw/floored references, conditional moments,
actual band endpoints, per-row RMSE and coverage; and `run.json` with parameters,
dependency versions, input shapes, and computed metrics.

Plots cover parameter marginals, mean response variability, response snapshots,
conditional comparisons, both spectra, coordinate selection, bandwidths, and
sample diversity. Extra conditional examples save all query/sample indices
and seeds in `figure_examples.json`. They are deterministic diagnostic
counterparts, not assertions of exact original-panel replication.

Response variability bands show one standard deviation, matching Figure 3.
Snapshot plots choose the closest grid samples to 70, 140, 340, and 680 ms,
following page 8, and record the actual times. The inherited `[0,2999)` ms
grid and original figure cases need confirmation. Full-field cardiac simulation,
geometric reconstruction, and cell-population results (Figures 1, 14, 15)
are outside the supplied notebook's scope. See [the figure map](docs/FIGURES.md).

## Tests and provenance

```bash
python -m unittest discover -s tests -v
```


