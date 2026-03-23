from __future__ import annotations

import numpy as np

from analysis.runtime import build_backend, load_controller_config, load_system_parameters
from control.physical_mode_guard import assess_induced_hover, guard_induced_command, induced_start_allowed
from control.pid import TwoLayerController
from physics.force_feasibility import evaluate_hover_feasibility
from physics.induced_dipole_force import InducedDipoleForceModel
from physics.magnetic_field_model import FaceDipoleFieldModel
from sim.state import RigidBodyState


def _induced_model(system):
    field = FaceDipoleFieldModel(system.cube, system.actuators)
    return InducedDipoleForceModel(field, system.coupling.induced_alpha)


def test_induced_force_finite_difference_stability():
    system = load_system_parameters(physical_preset="demo_calibrated_physical")
    model = _induced_model(system)
    x = np.array([0.5, 0.5, 0.55], dtype=float)
    hover = evaluate_hover_feasibility(model, x, system.ball.mass, system.cube.gravity, mode="nonlinear")
    assert hover.allocation.feasible
    analytic_force = model.force(x, hover.allocation.u)
    for h in (1.0e-6, 2.0e-6, 5.0e-6, 1.0e-5, 2.0e-5, 5.0e-5, 1.0e-4):
        fd_force = model.force_from_gradient_fd(x, hover.allocation.u, h)
        relative_error = np.linalg.norm(fd_force - analytic_force) / np.linalg.norm(analytic_force)
        assert relative_error < 5.0e-5


def test_induced_mode_infeasibility_gating():
    system = load_system_parameters(physical_preset="conservative_physical")
    model = _induced_model(system)
    x = np.array([0.5, 0.5, 0.55], dtype=float)
    assessment = assess_induced_hover(
        model,
        x,
        system.ball.mass,
        system.cube.gravity,
        system.ball.radius,
        system.induced_mode,
    )
    allowed, status = induced_start_allowed(assessment, exploratory_enabled=False)
    exploratory_allowed, exploratory_status = induced_start_allowed(assessment, exploratory_enabled=True)
    assert not allowed
    assert status == "target hover infeasible"
    assert exploratory_allowed
    assert exploratory_status == "guarded_exploratory"


def test_induced_mode_bounded_force_in_exploratory():
    system = load_system_parameters(physical_preset="demo_calibrated_physical")
    model = _induced_model(system)
    _, upper = model.actuator_bounds()
    result = guard_induced_command(
        model,
        np.array([0.5, 0.5, 0.08], dtype=float),
        np.zeros(3, dtype=float),
        proposed_u=upper,
        previous_u=np.zeros_like(upper),
        dt=system.backend.dt_by_mode["induced_dipole"],
        ball_radius=system.ball.radius,
        safety=system.induced_mode,
        exploratory_enabled=True,
    )
    assert result.aborted
    assert not result.allowed
    assert np.linalg.norm(result.achieved_force) <= system.induced_mode.max_force_norm


def test_induced_mode_admissible_region_normal_run():
    system = load_system_parameters(physical_preset="demo_calibrated_physical")
    model = _induced_model(system)
    backend = build_backend(system, enable_contact=True)
    controller_cfg = load_controller_config()
    controller = TwoLayerController(
        model,
        system.ball.mass,
        system.cube.gravity,
        controller_cfg.gains,
        allocator_mode="nonlinear",
        integral_limit=controller_cfg.integral_limit,
    )
    target = np.array([0.5, 0.5, 0.55], dtype=float)
    state0 = RigidBodyState(target + np.array([0.02, 0.0, 0.01], dtype=float), np.zeros(3, dtype=float), 0.0)
    log = backend.simulate(
        state0,
        model,
        duration=1.0,
        dt=system.backend.dt_by_mode["induced_dipole"],
        controller=controller,
        target_provider=lambda _t: target,
    )
    assert all(log.inside_admissible_flags)
    assert max(log.stage_projection_counts) == 0
    assert min(log.boundary_distances) > system.induced_mode.min_boundary_margin
    assert np.max(np.linalg.norm(np.asarray(log.achieved_forces), axis=1)) < system.induced_mode.max_force_norm


def test_fixed_dipole_regression_hover_feasible():
    system = load_system_parameters(physical_preset="demo_calibrated_physical")
    field = FaceDipoleFieldModel(system.cube, system.actuators)
    from physics.dipole_force import FixedDipoleForceModel

    model = FixedDipoleForceModel(field, system.coupling.fixed_dipole_moment)
    hover = evaluate_hover_feasibility(
        model,
        np.array([0.5, 0.5, 0.55], dtype=float),
        system.ball.mass,
        system.cube.gravity,
        mode="bounded_ls",
    )
    assert hover.allocation.feasible
    assert hover.allocation.residual_norm < 1.0e-6
