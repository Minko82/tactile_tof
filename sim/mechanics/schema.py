"""Versioned mechanics-output and estimator metadata."""

MECHANICS_OUTPUT_SCHEMA_VERSION = 2
CONTACT_METRIC_MODEL = "vbd_penalty_reconstruction"
CONTACT_FORCE_ESTIMATOR_VERSION = 1

SIMULATION_CAPABILITY = "normal_indentation_mechanics_mvp"
SHEAR_VALIDATED = False
SLIP_VALIDATED = False

DEPRECATED_ALIASES = {
    "contact_area_m2": "approx_contact_area_m2",
    "normal_force_n": "estimated_axial_reaction_n",
    "tangential_force_n": "estimated_transverse_reaction_n",
    "slip_velocity_m_s": "estimated_tangential_relative_velocity_m_s",
    "maximum_displacement_m": "maximum_displacement_from_equilibrated_baseline_m",
}
