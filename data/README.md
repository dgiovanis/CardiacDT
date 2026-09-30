# Required arrays and unresolved provenance

Supply finite real NumPy arrays with samples in rows. Files are loaded with
`allow_pickle=False`; no pickle is required. Arrays are not distributed here.

The paper-reproduction command deliberately uses the following files because
these are the inputs of the revision analysis that produced Table 1:

| File | Shape | Computational role |
| --- | --- | --- |
| `tof_input_data.npy` | `(100, 16)` | Training parameters |
| `tof_output_data.npy` | `(100, 582)` | Training response traces |
| `input_for_validation_arrythmia.npy` | `(10, 16)` | VT comparison inputs |
| `output_for_validation_arrythmia.npy` | `(10, 582)` | VT reference array used for Table 1 |
| `input_for_validation_not_arrythmia.npy` | `(10, 16)` | Non-VT comparison inputs |
| `output_for_validation_not_arrythmia.npy` | `(10, 582)` | Non-VT reference array used for Table 1 |

**The actual simulator origin of the last two response arrays remains
unverified.** The original notebook instead uses `manifold_arrhythmia_sims.npy`
and `manifold_no_arrhythmia_sims.npy`, which produce different comparison
metrics. Confirm the simulator-export provenance and row alignment before
calling either set independent high-fidelity ground truth. Merely matching the
paper's reported numbers does not establish that provenance.

The historical `arrythmia` spelling is preserved in filenames. A single-case
run needs its two comparison arrays only when `--validate` is requested.
The full paper command always requires all six files and checks their hashes.
No fallback between reference sets occurs silently.

Input labels inherited from the notebook, in numeric column order:

| Zero-based column | Inherited label |
| ---: | --- |
| 0 | `INa_h` |
| 1 | `ICaL_h` |
| 2 | `Ito_h` |
| 3 | `IK1_h` |
| 4 | `IKs_h` |
| 5 | `IKr_h` |
| 6 | `INa_f` |
| 7 | `ICaL_f` |
| 8 | `Ito_f` |
| 9 | `IK1_f` |
| 10 | `IKs_f` |
| 11 | `IKr_f` |
| 12 | `sigma_HL` |
| 13 | `sigma_FL` |
| 14 | `sigma_HT` |
| 15 | `sigma_FT` |

**Labels require confirmation:** column 0 reaches 3.970141, exceeding the
paper's INa upper bound of 3; column 1 stays below 3. This suggests a possible
label/order issue but is not enough evidence to reorder the data. The package
preserves numerical columns and records the discrepancy in the paper audit.

The inherited response-time convention is 582 points on `[0,2999)` ms; no
explicit simulator-export time vector is supplied. The tail classifier assumes
zero padding of terminated non-VT signals. Signal units, time grid, outcome
labels, and preprocessing should be confirmed from simulation metadata.
The revision's -90 mV floor is applied downstream, not while loading these arrays.

Before distributing data, supply an authorized access location and reuse terms.
The source hashes identify the tested files; they do not grant distribution
rights or prove simulation provenance.
