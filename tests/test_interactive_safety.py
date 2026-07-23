from __future__ import annotations

import numpy as np

from sim.mechanics.interactive_safety import (
    ProbePose,
    commanded_indentation,
    evaluate_candidate_safety,
)


def _evaluate(
    relative_j,
    *,
    force_n: float = 0.0,
    indentation_m: float = 0.0,
):
    return evaluate_candidate_safety(
        relative_j=relative_j,
        estimated_force_magnitude_n=force_n,
        commanded_indentation_m=indentation_m,
        circuit_breaker_minimum_j=0.15,
        stop_minimum_j=0.20,
        warning_minimum_j=0.30,
        warning_estimated_force_n=1.0,
        stop_estimated_force_n=2.0,
        warning_commanded_indentation_m=0.00060,
        maximum_commanded_indentation_m=0.00075,
    )


def test_commanded_indentation_uses_probe_support_point_and_contact_reference():
    local_vertices = np.asarray(
        [
            [x, y, z]
            for x in (-0.0125, 0.0125)
            for y in (-0.009, 0.009)
            for z in (-0.006, 0.006)
        ],
        dtype=np.float64,
    )
    probe = {"type": "rounded_block"}
    contact_location = np.zeros(3)
    direction = np.asarray([0.0, 0.0, -1.0])

    clearance_pose = ProbePose([0.0, 0.0, 0.016], [0.0, 0.0, 0.0, 1.0])
    contact_pose = ProbePose([0.0, 0.0, 0.006], [0.0, 0.0, 0.0, 1.0])
    limit_pose = ProbePose([0.0, 0.0, 0.00525], [0.0, 0.0, 0.0, 1.0])

    assert np.isclose(
        commanded_indentation(
            clearance_pose,
            probe=probe,
            contact_location_m=contact_location,
            contact_direction=direction,
            mesh_vertices_m=local_vertices,
        ),
        -0.010,
    )
    assert np.isclose(
        commanded_indentation(
            contact_pose,
            probe=probe,
            contact_location_m=contact_location,
            contact_direction=direction,
            mesh_vertices_m=local_vertices,
        ),
        0.0,
    )
    assert np.isclose(
        commanded_indentation(
            limit_pose,
            probe=probe,
            contact_location_m=contact_location,
            contact_direction=direction,
            mesh_vertices_m=local_vertices,
        ),
        0.00075,
    )


def test_safety_warning_can_report_primary_and_secondary_limits_together():
    evaluation = _evaluate(
        [0.95, 0.25, 0.90],
        force_n=1.2,
        indentation_m=0.00065,
    )

    assert not evaluation.fatal
    assert not evaluation.stopped
    assert set(evaluation.warning_reasons) == {
        "minimum_relative_tet_volume",
        "estimated_force",
        "commanded_indentation",
    }
    np.testing.assert_array_equal(evaluation.affected_tet_indices, [1])


def test_tet_stop_precedes_secondary_stop_but_remains_recoverable():
    evaluation = _evaluate(
        [0.95, 0.19, 0.90],
        force_n=3.0,
        indentation_m=0.001,
    )

    assert not evaluation.fatal
    assert evaluation.stop_reason == "minimum_relative_tet_volume"
    np.testing.assert_array_equal(evaluation.affected_tet_indices, [1])


def test_existing_j_circuit_breaker_remains_fatal_at_point_one_five():
    evaluation = _evaluate([0.95, 0.149, 0.90])

    assert evaluation.fatal
    assert evaluation.fatal_reason == "tet_volume_circuit_breaker"
    assert evaluation.stop_reason is None
    np.testing.assert_array_equal(evaluation.affected_tet_indices, [1])


def test_commanded_indentation_and_force_are_secondary_recoverable_stops():
    indentation = _evaluate([0.95, 0.80], indentation_m=0.000751)
    force = _evaluate([0.95, 0.80], force_n=2.01)

    assert indentation.stop_reason == "commanded_indentation_limit"
    assert force.stop_reason == "estimated_force_limit"
    assert not indentation.fatal
    assert not force.fatal
