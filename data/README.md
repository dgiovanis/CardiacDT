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

