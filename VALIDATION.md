# Verification record

Verified locally on 2026-09-30 with Python 3.11.16, macOS arm64, NumPy 1.26.4,
SciPy 1.11.4, scikit-learn 1.2.2, and Datafold 2.0.1. Full dependency versions
are in `requirements-tested.txt`. Python 3.10 is configured in CI but was not
run locally. The source revision report names Datafold 2.0.2; the public pinned
2.0.1 environment nevertheless reproduces the reported numbers below.

## Final numerical checks

- **16 synthetic tests passed**, including direct formula checks for the
  conditional estimator, scaling/PCA, deterministic serial/parallel ISDE,
  RNG isolation, shape validation, explicit reference selection, flooring,
  saved coverage, duplicate distances, and actual kernel parameter convention.
- **53 paper numerical/data-identity checks passed** using fresh full runs of
  both regimes. Checks cover Tables 1 and 3, six source-array hashes, retained
  coordinates, sample counts, and the Figure 5 zero-duplicate claim.
- The original PLoM numerical classes/functions remain unchanged apart from
  formatting/import cleanup. The first refactor matched notebook samples;
  the final defaults additionally reproduce the revision's -90 mV floor and
  reference-array choice. Notebook equivalence is not used as a substitute
  for checking the paper.
- Both full runs generated figures, moments, raw/floored data, metadata, and
  diagnostic arrays. New conditional panels use seeded examples and actual
  diffusion coordinates where labeled latent. Representative panels were
  visually checked; exact original-panel reproduction is not claimed.
- Ruff checks and formatting checks pass. The reproduction CLI exits nonzero
  on a failed numerical comparison, and reports manuscript consistency
  separately from the numerical pass.

| Class | Training rows | Generated rows | RMSE (mV) | Temporal coverage |
| --- | ---: | ---: | ---: | ---: |
| VT | 39 | 3900 | 32.4375066661 | 0.9073883162 |
| non-VT | 61 | 6100 | 12.6085673890 | 0.9831615120 |
| All | 100 | 10000 | 22.5230370275 | 0.9452749141 |

These are means over ten comparison rows in each regime, using the revision's
reference arrays and floored bands. They agree with Table 1 at its published
precision. The generated data and reference arrays are not embedded in the
public repository. Saved summary evidence is in `docs/reproduction_check.json`
and `docs/reproduced_table1.csv`.

## Limitations that a numerical pass does not clear

See [docs/PAPER_AUDIT.md](docs/PAPER_AUDIT.md). Reference-file simulator
provenance, input-column labels, kernel notation, undocumented flooring,
conditioning metric/variance descriptions, and figure time/latent definitions
require reconciliation. Upstream imaging and EP simulation figures cannot be
recreated from this notebook package. The paper's complete scientific results
are therefore **not** certified free of inconsistency.
