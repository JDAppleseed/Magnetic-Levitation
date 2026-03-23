from __future__ import annotations

import numpy as np

from analysis.reachable_set import summarize_reachable_set
from physics.force_feasibility import evaluate_hover_feasibility, hover_force


def test_hover_feasibility_matches_problem1(affine_model, system, sim_defaults):
    hover = evaluate_hover_feasibility(
        affine_model,
        sim_defaults.hover_point,
        system.ball.mass,
        system.cube.gravity,
        mode="bounded_ls",
    )
    assert hover.allocation.feasible
    assert hover.allocation.residual_norm < 1.0e-6
    lower, upper = affine_model.actuator_bounds()
    assert np.all(hover.allocation.u >= lower - 1.0e-9)
    assert np.all(hover.allocation.u <= upper + 1.0e-9)


def test_reachable_force_summary_has_full_rank_at_hover_point(affine_model, sim_defaults):
    summary = summarize_reachable_set(affine_model, sim_defaults.hover_point)
    assert summary.rank == 3
    assert np.all(summary.singular_values > 0.0)


def test_hover_force_matches_gravity(system):
    required = hover_force(system.ball.mass, system.cube.gravity)
    assert np.allclose(required, np.array([0.0, 0.0, system.ball.mass * system.cube.gravity]))

