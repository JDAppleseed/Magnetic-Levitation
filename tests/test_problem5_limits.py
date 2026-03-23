from __future__ import annotations

import numpy as np

from analysis.runtime import load_system_parameters
from physics.affine_force_model import AffineFaceMagneticForceModel
from physics.decay_analysis import analyze_decay_limit, critical_domain_size
from physics.earnshaw_demos import analyze_fixed_input_potential
from physics.force_feasibility import evaluate_hover_feasibility
from physics.magnetic_field_model import CubeGeometry, build_face_actuators


def test_problem5_decay_limit(system):
    decay_constant = system.actuators[0].affine_gain
    alpha = system.actuators[0].decay_power
    u_max = system.actuators[0].u_max
    critical = critical_domain_size(system.ball.mass, system.cube.gravity, u_max, decay_constant, alpha)
    side_length = 1.05 * critical
    cube = CubeGeometry(side_length=side_length, gravity=system.cube.gravity)
    actuators = build_face_actuators(
        side_length,
        system.actuators[0].u_min,
        u_max,
        decay_constant,
        alpha,
        0.0,
        system.actuators[0].dipole_strength,
    )
    model = AffineFaceMagneticForceModel(cube, actuators)
    center = np.array([side_length / 2.0, side_length / 2.0, side_length / 2.0], dtype=float)
    hover = evaluate_hover_feasibility(model, center, system.ball.mass, system.cube.gravity, mode="bounded_ls")
    summary = analyze_decay_limit(side_length, system.ball.mass, system.cube.gravity, u_max, decay_constant, alpha)
    assert not summary.center_hover_feasible
    assert not hover.allocation.feasible


def test_problem5_earnshaw_obstruction(system, physical_models):
    fixed_model, _ = physical_models
    lower, upper = system.cube.admissible_bounds(system.ball.radius)
    u_star = np.ones(6, dtype=float)
    summary = analyze_fixed_input_potential(
        lambda x: fixed_model.scalar_potential(x, u_star),
        lower + 0.05,
        upper - 0.05,
        samples_per_axis=4,
    )
    assert summary.strict_minimum_count == 0
    assert summary.max_abs_trace < 1.0e-5
