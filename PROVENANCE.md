# Code and paper provenance

The final package uses the supplied `ddplom_heart.ipynb` for the joint-manifold
workflow, `plom_library.py` for the bundled numerical routines, and
`manuscript_revision_analysis.py` for the reference-array selection, voltage
floor, and metrics that generated the revised paper's tables.
The supplied revised PDF is the authority for the reported numbers and captions.
None of these original files was modified.

| Source | Final counterpart |
| --- | --- |
| Notebook cells 0-5 | Explicit imports, array loading, tail classifier |
| Notebook cells 6-9 | Bootstrap KDE and reference response figures |
| Notebook cells 10-22 | `workflow.Config` and `workflow.fit_generate` |
| Notebook cells 23-24, 29 | Generated-response/density diagnostics |
| Revision `run_regime` | Lower floor, reference-file choice, RMSE and coverage |
| Revision diversity/hyperparameter analysis | `reproduce.py` numerical checks and diagnostics |
| Notebook cells 30-31 | Conditional example samplers; Q explicitly set to diffusion coordinates |
| Revised PDF Tables 1/3 and page 16 | `reference.py` assertions (not fitting inputs) |

All seven top-level numerical definitions in `plom.py` retain the original
abstract syntax trees. Import cleanup and formatting do not alter their
formulas. The wrappers add validation, deterministic execution, explicit paths,
configuration, serial defaults, raw/floored result storage, and scientific
provenance warnings. The NumPy RNG is scoped and restored around the workflow.

Differences from the original notebook include the revision's -90 mV lower
floor and `output_for_validation_*` reference selection, Figure 3 standard
deviation bands, Figure 6 requested snapshot times, and actual latent variables
in latent-labeled diagnostics. New illustrative conditional draws follow the
full categorical distribution, not the separate export's nearest-25 restriction.
Figure layouts and example selections are not asserted identical to the paper.

The original unmodified notebook was numerically matched in the first cleanup;
that was not sufficient for paper reproduction. Version 0.2.0 instead checks
the revision's reported numbers directly. It explicitly distinguishes this
numerical match from unverified simulator provenance and unresolved manuscript
consistency. See `docs/PAPER_AUDIT.md` for evidence and required reconciliations.

No license was present in the supplied research sources. Ownership and reuse
terms must be established by the authors before adding a distribution license.

## Source SHA-256

- `ddplom_heart.ipynb`: `8f595e70fef92a1d8223d8960435a914bb0f03bbd4ff34fd3d85bdcf88a30886`

- `plom_library.py`: `dc2e4325b27e15f0d0cc5a669045a229acb3edc801bded73317096a2edb68aad`

- `manuscript_revision_analysis.py`: `51db985d053ab6996f666a08ada790cee4ac4f94553d948c9bf10f3df2fd0be7`

- `Generative_Learning_of_Cardiac_Digital_Twins_Revision_Clean.pdf`: `de48079edcb9d772d77c4cdea8caab7a4ba71242ce5618ba1a803b02d6f5a669`
