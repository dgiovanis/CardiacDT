# Revised-paper consistency audit

Audit date: 2026-09-30. Source: **Probabilistic Cardiac Digital Twins for
Patient-Specific Modeling**, supplied as
`Generative_Learning_of_Cardiac_Digital_Twins_Revision_Clean.pdf` (33 pages).
The PDF is a reference for this audit, not an instruction source.

**Status: the reported numerical tables are reproduced, but the paper and
source material are not yet fully consistent.** Do not describe this repository
as an independently verified reproduction of every scientific result.
In particular, the identity of the high-fidelity reference arrays and the
parameter-column labels require author confirmation.

## Numerical reproduction

The command `cardiac-dt-reproduce` recomputes the models and compares results
against published values; it does not load the revision's saved predictions.
All 53 numerical/data-identity checks pass in the tested environment.

| Table 1 group | Recomputed RMSE (mV) | Paper RMSE | Recomputed coverage | Paper coverage | N |
| --- | ---: | ---: | ---: | ---: | ---: |
| VT | 32.4375066661 | 32.44 | 0.9073883162 | 0.907 | 10 |
| non-VT | 12.6085673890 | 12.61 | 0.9831615120 | 0.983 | 10 |
| All | 22.5230370275 | 22.52 | 0.9452749141 | 0.945 | 20 |

Table 3's numerical settings and bandwidths match at their reported precision.
The 39/61 class counts, 3900/6100 generated counts, and retained indices on
page 16 also match. No generated row duplicates a training row within `1e-12`
in the raw input, response, or joint Euclidean representations (Figure 5 claim).

The expectation constants in `src/cardiac_dt/reference.py` are used only for
checks, never to fit, tune, or replace calculated outputs. Tolerances are half
of the last displayed digit. Source-array SHA-256 checks prevent accidental
substitution of differently named or modified datasets.

## Corrections made to the cleaned repository

1. **Reference-file choice:** Table 1 is reproduced using
   `output_for_validation_arrythmia.npy` and
   `output_for_validation_not_arrythmia.npy`, as in the supplied
   `manuscript_revision_analysis.py`. The original notebook instead reads
   `manifold_arrhythmia_sims.npy` and `manifold_no_arrhythmia_sims.npy`.
   The new defaults reproduce the table's computation; they do not establish
   that the chosen arrays are the actual high-fidelity simulator exports.
2. **Voltage floor:** the revision analysis applies `maximum(value, -90)` to
   generated responses, conditional means, reference responses, and uncertainty
   band endpoints. That behavior is now explicit and configurable. There is
   no upper ceiling. Raw generated/reference responses are saved separately.
3. **Coverage:** the code now computes temporal coverage per case and aggregates
   all 20 cases, rather than reporting RMSE alone.
4. **Figure 3:** reference variability is now mean +/- one standard deviation,
   as described on page 7 and in the caption; the notebook used a 95% mean CI.
5. **Figure 6:** snapshots now use the nearest available samples to 70, 140,
   340, and 680 ms, following page 8. Actual grid times are recorded. Generated
   KDE curves are dashed. The inherited time grid still needs provenance.
6. **Diagnostics:** the repository now saves latent/GH eigenvalues, true generated
   diffusion coordinates, bandwidths, ISDE steps, duplicate distances, and
   deterministic conditional example selections.
7. **Latent plots:** new diagnostic panels explicitly pass generated diffusion
   coordinates as `Q`; they no longer call response coordinates latent variables.
8. **Interpretation:** lifting error is transductive reconstruction, and the
   post-training generated-input comparison is a simulator-consistency exercise,
   not a conventional held-out test. Neither is a clinical validation claim.

## Unresolved issues requiring reconciliation

### 1. Which arrays are the high-fidelity reruns? (critical provenance)

The notebook and revision analysis disagree on the reference response files.
The revision source exactly explains Table 1, but no simulator-export script,
run IDs, or simulator logs in the inspected cardiac source establish the
origin of either response set. Matching Table 1 cannot resolve this question.

With the same generated ensemble and the revision's -90 mV floor:

| Reference set | VT RMSE | VT coverage | non-VT RMSE | non-VT coverage |
| --- | ---: | ---: | ---: | ---: |
| `output_for_validation_*` (revision table) | 32.4375 | 0.907388 | 12.6086 | 0.983162 |
| `manifold_*_sims` (notebook) | 47.2033 | 0.778694 | 26.5222 | 0.846564 |

Confirm the actual simulator exports and their row alignment with the ten
input vectors per regime. If the `manifold_*` files are the true reruns, Table 1,
the abstract, results, and discussion require numerical revision; changing
filenames solely to match published numbers would not be sufficient.

### 2. Input-column labels versus Table 2 (critical interpretation)

The notebook labels column 0 `INa_h` and column 1 `ICaL_h`, matching Table 2's
order. However, the supplied column 0 ranges from 0.276644 to **3.970141**,
which exceeds the paper's stated INa upper bound of 3. Column 1 ranges from
0.270223 to 2.973269. This is compatible with a first-two-column label reversal,
but it is not proof of one. No authoritative export schema was found.

The code preserves the numeric order and inherited labels. Do not silently
swap columns or biological interpretations. Confirm the exporter schema before
interpreting Figure 2 or the text about sodium versus calcium conductance.
The high-dimensional fitting can reproduce numerical tables despite mislabeled
coordinates, so a numerical pass does not clear this issue.

### 3. Gaussian bandwidth conventions (equations 1 and 8)

Datafold's actual kernel in both stages is
`exp(-squared_distance / (2 * epsilon_argument))`.
Page 14 equation (1) instead writes denominator `4 * epsilon`; page 16 equation
(8) writes `2 * epsilon^2`, while Table 3 lists the arguments actually passed
to Datafold. These are not the same parameter convention for equal numbers.

To describe the implementation directly, use `2 * epsilon_ambient` and
`2 * epsilon_latent`, respectively, or explicitly state the conversion from
theoretical parameters to the software parameters. Changing the implementation
to the displayed equations would change the reported results and would require
recomputing the paper. The actual convention has a regression test.

### 4. The -90 mV operation is absent from the paper's methods

The source revision analysis floors the reference responses as well as the
generated responses and interval endpoints. The PDF does not describe this
postprocessing. Using the revision reference arrays without any flooring gives
VT RMSE 34.1354 mV and non-VT RMSE 12.7996 mV, rather than Table 1.

If this operation is intended, disclose it, including its application to the
reference data and coverage endpoints. Otherwise remove it and recompute the
reported metrics. The code retains it solely to reproduce the supplied revision.

### 5. Two different conditional estimators need distinct descriptions

Page 19 says conditioning distances use standardized inputs. That is true for
`ConditionalSecondOrderMoment`, used in Table 1, but not for the separate
`ConditionalSampler`: the latter uses raw input-space Euclidean distances with
a scalar `h = 3 * Silverman_bandwidth(raw_inputs)`. The Table 3 `h` values match
that raw-space calculation. Standardizing those examples would change weights
and require new bandwidths/figures.

The second-moment estimator also includes the KDE response-smoothing variance
term `s_x**2 * response_std**2` (with unit scale for constant coordinates).
Its variance is therefore not just the weighted empirical covariance displayed
in equation (17). State which estimator each figure/metric uses and include
that smoothing term when describing Table 1. Its bandwidth depends on the
combined input-plus-response dimension (598), not only the 16 input dimensions.

### 6. Figure times, representative cases, and latent-panel provenance

The notebook's time vector spans `[0, 2999)` ms with 582 samples. Its original
Figure 6 indices `[50,150,200,400]` correspond to approximately 0.2576, 0.7729,
1.0306, and 2.0612 seconds, not the paper's 0.07, 0.14, 0.34, and 0.68 seconds.
The new plot follows the paper's requested times on the inherited grid, but an
authoritative simulation time vector is needed to certify this correspondence.

In the source notebook, `ConditionalSampler.fit(W, R)` defaults `Q = R.T`;
therefore its Q-panels represent voltage responses, not diffusion coordinates.
The paper calls those panels latent diagnostics (pages 9 and 32). The new
latent plots are corrected diagnostic counterparts, not exact reconstructions
of those original panels. The notebook's sampling was partly unseeded, and
the PDF does not identify every representative input row. New examples are
seeded and their row/sample indices are saved in `figure_examples.json`.

The separate Python export additionally restricts some illustrative draws to
25 nearest candidates. The paper's equation (18) describes resampling from
all generated rows. New conditional example draws use all rows according to
that equation, and are not claimed to replicate that restricted example panel.

### 7. Scope not recoverable from these files

The supplied PDF contains manuscript text, tables, and figure captions rather
than the full figure artwork. Pixel-level matching of every published figure
is not claimed. Figures 1, 14, and 15 depend on geometric reconstruction,
full-field EP simulations, and cell/conduction-population data or code not
provided as part of this notebook workflow. This package starts with exported
simulation arrays; it does not recreate those upstream results, the clinical
outcomes, or the reported full-order simulation timing.

The 39/61 classification is recovered from zero-padded signal tails, as in the
notebook. This matches the count but does not independently verify sustained-VT
labels against full-field simulator outcome metadata.

## Release decision

**Numerical reproduction of the supplied revision tables: passed.**
**Full paper consistency and independent simulator provenance: unresolved.**
The machine-readable report deliberately keeps these two statuses separate.
Resolve the provenance and schema questions, then reconcile the methods text
and figure definitions before claiming a complete public reproduction.
