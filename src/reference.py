"""Published values transcribed from the supplied revised PDF (Tables 1/3).

Data hashes identify the local arrays used by the revision analysis; no arrays
are embedded. Values are used only for assertions, never for model fitting.
"""

DATA_SHA256 = {
    "tof_input_data.npy": "2f908402f438b314afa2e64bb874f79dde2d7c1b5ee8f82a8e25ff30c494e65f",
    "tof_output_data.npy": "68444fcf21dd6c5b30f94483830126c89271324f9f63b1632dd4ef6d0e49b26b",
    "input_for_validation_arrythmia.npy": "5796854f82ac46b25d851404cdadf95adba627ddc04bf2d83b8211efb4ca0b65",
    "output_for_validation_arrythmia.npy": "ae0f5c9a8358996a01faf6d60ed6bd5529a84e14a527f463e6f61bc4209fcad7",
    "input_for_validation_not_arrythmia.npy": "05f63ce5349adcf82f9ee666d1707c46d615f247a551dd6ba4520102fe1a2db5",
    "output_for_validation_not_arrythmia.npy": "c40f067a9825997e09fe693a15039a75e979ee8ab5ee1cf364524d7eda560fe6",
}

PAPER_TABLE1 = {
    "arrhythmia": {"n": 10, "rmse_mV": 32.44, "coverage": 0.907},
    "nonarrhythmia": {"n": 10, "rmse_mV": 12.61, "coverage": 0.983},
    "all": {"n": 20, "rmse_mV": 22.52, "coverage": 0.945},
}

PAPER_TABLE3 = {
    "arrhythmia": {
        "class_samples": (39, 0),
        "generated_samples": (3900, 0),
        "diffusion_epsilon_base": (200342.0, 0.5),
        "diffusion_epsilon": (6010260.0, 5),
        "candidate_eigenpairs": (38, 0),
        "diffusion_exponent": (0, 0),
        "intrinsic_dimension": (15, 0),
        "lifting_epsilon": (39.31996, 5e-06),
        "lifting_modes": (30, 0),
        "damping": (0.01, 0),
        "transient_steps": (50, 0),
        "monte_carlo_trajectories": (100, 0),
        "stochastic_step": (0.000699322, 5e-10),
        "conditional_sampling_bandwidth": (2.02283, 5e-06),
        "conditional_moment_bandwidth": (0.97818, 5e-06),
    },
    "nonarrhythmia": {
        "class_samples": (61, 0),
        "generated_samples": (6100, 0),
        "diffusion_epsilon_base": (99067.7, 0.05),
        "diffusion_epsilon": (2972030.0, 5),
        "candidate_eigenpairs": (38, 0),
        "diffusion_exponent": (0, 0),
        "intrinsic_dimension": (15, 0),
        "lifting_epsilon": (26.0131, 5e-06),
        "lifting_modes": (30, 0),
        "damping": (0.01, 0),
        "transient_steps": (50, 0),
        "monte_carlo_trajectories": (100, 0),
        "stochastic_step": (0.000686943, 5e-10),
        "conditional_sampling_bandwidth": (1.09951, 5e-06),
        "conditional_moment_bandwidth": (0.97746, 5e-06),
    },
}

RETAINED_INDICES = {
    "arrhythmia": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    "nonarrhythmia": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 19, 20, 21],
}
