# Figure reproduction scope

The paper command emits reproducible diagnostic counterparts. The supplied
PDF contains captions, not the full artwork; original layouts and every query
selection cannot be verified from it. Paths below are relative to a run's
output directory; `{case}` is `arrhythmia` or `nonarrhythmia`.

| Paper figure | Generated file(s) | Qualification |
| --- | --- | --- |
| 1 | Not available | Full-field VT simulation snapshots are outside this code |
| 2 | `{case}/input_densities.png` | Bootstrap KDE; inherited column labels need confirmation |
| 3 | `{case}/reference_responses.png` | Mean +/- population std; both regimes shown |
| 4 | `{case}/generated_responses.png` | Reference/generated means and dispersion |
| 5 | `figure05_sample_diversity.png` and PDF | Raw Euclidean nearest-training distances; numerical claim checked |
| 6 | `{case}/response_densities.png` | Nearest samples to the paper's stated times; grid provenance pending |
| 7 | `{case}/conditional_training_moments.png` and PDF | Four explicitly recorded training queries; not identified original panels |
| 8, 9 | `{case}/conditional_training_samples.png` and PDF | Full weighted categorical draws, not the export's nearest-25 restriction |
| 10, 11 | `{case}/conditional_response_and_latent.png` and PDF | Actual diffusion coordinates supplied as Q; legacy Q was the response |
| 12, 13 | `{case}/conditional_validation.png` | Four reference rows [1,6,5,9], zero-based; provenance unresolved |
| 14 | Not available | Imaging/geometric reconstruction pipeline not supplied |
| 15 | Not available | Cell/conduction population simulation inputs not supplied |
| 16 | `figure16_ambient_spectrum.png` and PDF | 38 eigenpairs in each regime |
| 17 | `figure17_coordinate_selection.png` and PDF | Fixed 15-coordinate selection, retained indices checked |
| 18 | `figure18_bandwidths.png` and PDF | Computed ambient base and latent bandwidths |
| 19 | `figure19_latent_spectrum.png` and PDF | All 30 computed GH eigenvalues |

`figure_examples.json` records selected training rows, sample indices, seeds,
metric convention, effective sample size, and actual snapshot times. Local
Gaussian latent draws are moment diagnostics, not additional simulated responses.
