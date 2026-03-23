from __future__ import annotations

from analysis.linearization import analytic_linearization, finite_difference_linearization, operator_norm_difference
from control.stabilizability_checks import evaluate_stabilizability
from physics.force_feasibility import evaluate_hover_feasibility


def test_problem2_linearization_matches_finite_difference(affine_model, system, sim_defaults):
    hover = evaluate_hover_feasibility(
        affine_model,
        sim_defaults.hover_point,
        system.ball.mass,
        system.cube.gravity,
        mode="bounded_ls",
    )
    analytic = analytic_linearization(affine_model, sim_defaults.hover_point, hover.allocation.u, system.ball.mass)
    fd = finite_difference_linearization(affine_model, sim_defaults.hover_point, hover.allocation.u, system.ball.mass)
    assert operator_norm_difference(analytic.A_x, fd.A_x) < 1.0e-4
    assert operator_norm_difference(analytic.B_x, fd.B_x) < 1.0e-4


def test_problem2_stabilizability_report_is_full_rank(affine_model, system, sim_defaults):
    hover = evaluate_hover_feasibility(
        affine_model,
        sim_defaults.hover_point,
        system.ball.mass,
        system.cube.gravity,
        mode="bounded_ls",
    )
    linearization = analytic_linearization(affine_model, sim_defaults.hover_point, hover.allocation.u, system.ball.mass)
    report = evaluate_stabilizability(linearization.A_x, linearization.B_x, linearization.A, linearization.B)
    assert report.input_rank == 3
    assert report.full_translational_authority
    assert report.stabilizable

