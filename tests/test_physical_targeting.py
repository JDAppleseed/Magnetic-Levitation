from __future__ import annotations

import numpy as np

from control.physical_targeting import (
    EXECUTION_BLOCKED,
    EXECUTION_DIRECT_TRAJECTORY,
    EXECUTION_FALLBACK_REGULATION,
    EXECUTION_GUARDED_FALLBACK,
    EXECUTION_HOLD,
    ENDPOINT_BLOCKED,
    ENDPOINT_EXACT_FEASIBLE,
    ENDPOINT_MARGINAL,
    PATH_BLOCKED,
    PATH_MARGINAL,
    START_ALLOWED,
    START_ALLOWED_CAUTION,
    START_BLOCKED,
    START_UNSAFE_ABORT,
    PhysicalTargetState,
    classify_physical_plan,
    plan_physical_target_request,
)
from physics.dipole_force import FixedDipoleForceModel
from physics.induced_dipole_force import InducedDipoleForceModel
from physics.magnetic_field_model import FaceDipoleFieldModel, load_system_parameters


def _fixed_model(system):
    field = FaceDipoleFieldModel(system.cube, system.actuators)
    return FixedDipoleForceModel(field, system.coupling.fixed_dipole_moment)


def _induced_model(system):
    field = FaceDipoleFieldModel(system.cube, system.actuators)
    return InducedDipoleForceModel(field, system.coupling.induced_alpha)


def _plan(system, model, source, target, exploratory_enabled=False, duration=4.0):
    return plan_physical_target_request(
        model,
        np.asarray(source, dtype=float),
        np.asarray(target, dtype=float),
        system.ball.mass,
        system.cube.gravity,
        trajectory_duration=duration,
        ball_radius=system.ball.radius,
        induced_safety=system.induced_mode,
        status_thresholds=system.physical_status,
        sample_count=system.physical_status.path_sample_count,
        exploratory_enabled=exploratory_enabled,
        mode_name=model.__class__.__name__,
    )


def test_fixed_dipole_exact_feasible_classification():
    system = load_system_parameters(physical_preset="demo_calibrated_physical")
    plan = _plan(system, _fixed_model(system), [0.5, 0.5, 0.55], [0.5, 0.5, 0.55])
    assert plan.assessment.target_status == ENDPOINT_EXACT_FEASIBLE
    assert plan.assessment.state == START_ALLOWED
    assert plan.assessment.execution_mode == EXECUTION_HOLD
    assert plan.accepted
    assert plan.assessment.start_allowed
    assert plan.endpoint.residual_norm < system.physical_status.exact_residual_norm


def test_fixed_dipole_marginal_feasible_classification():
    system = load_system_parameters(physical_preset="demo_calibrated_physical")
    plan = _plan(system, _fixed_model(system), [0.2, 0.5, 0.5], [0.2, 0.5, 0.5])
    assert plan.assessment.target_status == ENDPOINT_MARGINAL
    assert plan.assessment.state == START_ALLOWED_CAUTION
    assert plan.assessment.execution_mode == EXECUTION_HOLD
    assert plan.accepted
    assert plan.endpoint.margin_estimate < system.physical_status.exact_margin
    assert plan.endpoint.margin_estimate >= system.physical_status.marginal_margin


def test_fixed_dipole_infeasible_target_is_blocked():
    system = load_system_parameters(physical_preset="demo_calibrated_physical")
    plan = _plan(system, _fixed_model(system), [0.5, 0.5, 0.55], [0.5, 0.5, 0.8])
    assert plan.assessment.target_status == ENDPOINT_BLOCKED
    assert plan.assessment.state == START_BLOCKED
    assert plan.assessment.execution_mode == EXECUTION_BLOCKED
    assert not plan.accepted
    assert not plan.assessment.start_allowed


def test_endpoint_path_split_for_fixed_dipole_transfer():
    system = load_system_parameters(physical_preset="demo_calibrated_physical")
    plan = _plan(system, _fixed_model(system), [0.2, 0.5, 0.5], [0.8, 0.5, 0.5], duration=0.5)
    assert plan.path is not None
    assert plan.assessment.target_status == ENDPOINT_MARGINAL
    assert plan.assessment.path_status == PATH_BLOCKED
    assert plan.assessment.state == START_ALLOWED_CAUTION
    assert plan.assessment.execution_mode == EXECUTION_FALLBACK_REGULATION
    assert plan.accepted
    assert plan.path.first_failure_time is not None


def test_fixed_dipole_nominal_transfer_still_uses_trajectory():
    system = load_system_parameters(physical_preset="demo_calibrated_physical")
    plan = _plan(system, _fixed_model(system), [0.5, 0.5, 0.55], [0.5, 0.5, 0.6], duration=0.5)
    assert plan.assessment.target_status == ENDPOINT_MARGINAL
    assert plan.assessment.path_status == PATH_MARGINAL
    assert plan.assessment.state == START_ALLOWED_CAUTION
    assert plan.assessment.execution_mode == EXECUTION_DIRECT_TRAJECTORY
    assert plan.accepted


def test_induced_path_blocked_uses_guarded_fallback():
    system = load_system_parameters(physical_preset="demo_calibrated_physical")
    plan = _plan(
        system,
        _induced_model(system),
        [0.2, 0.5, 0.5],
        [0.8, 0.5, 0.5],
        exploratory_enabled=False,
        duration=0.5,
    )
    assert plan.assessment.target_status == ENDPOINT_EXACT_FEASIBLE
    assert plan.assessment.path_status == PATH_BLOCKED
    assert plan.assessment.state == START_ALLOWED_CAUTION
    assert plan.assessment.execution_mode == EXECUTION_GUARDED_FALLBACK
    assert plan.accepted

    override_plan = _plan(
        system,
        _induced_model(system),
        [0.2, 0.5, 0.5],
        [0.8, 0.5, 0.5],
        exploratory_enabled=True,
        duration=0.5,
    )
    assert override_plan.assessment.state == START_ALLOWED_CAUTION
    assert override_plan.assessment.execution_mode == EXECUTION_GUARDED_FALLBACK
    assert override_plan.accepted
    assert override_plan.assessment.start_allowed


def test_last_feasible_target_fallback_preserves_hold_target():
    system = load_system_parameters(physical_preset="demo_calibrated_physical")
    model = _fixed_model(system)
    state = PhysicalTargetState()
    accepted = _plan(system, model, [0.5, 0.5, 0.55], [0.5, 0.5, 0.55])
    blocked = _plan(system, model, [0.5, 0.5, 0.55], [0.5, 0.5, 0.8])

    assert state.apply_request(accepted, simulation_active=False)
    assert state.pending_plan is accepted
    state.activate_pending()
    assert np.allclose(state.hold_target, accepted.requested_target)
    assert state.last_feasible_target is None

    assert not state.apply_request(blocked, simulation_active=False)
    assert state.pending_plan is None
    assert np.allclose(state.hold_target, accepted.requested_target)
    assert state.last_feasible_target is None
    assert state.preview_status == blocked.assessment.target_status
    assert state.warning == blocked.assessment.message


def test_target_selection_uses_same_assessment_for_status_and_gating():
    system = load_system_parameters(physical_preset="demo_calibrated_physical")
    plan = _plan(system, _fixed_model(system), [0.5, 0.5, 0.55], [0.5, 0.5, 0.8])
    state = PhysicalTargetState()
    state.apply_request(plan, simulation_active=False)
    assert state.preview_status == plan.assessment.target_status
    assert state.warning == plan.assessment.message
    assert plan.accepted == plan.assessment.start_allowed


def test_runtime_unsafe_abort_state_disables_restart():
    system = load_system_parameters(physical_preset="demo_calibrated_physical")
    plan = _plan(system, _induced_model(system), [0.5, 0.5, 0.55], [0.5, 0.5, 0.55], exploratory_enabled=True)
    assessment = classify_physical_plan(
        plan.endpoint,
        plan.path,
        system.physical_status,
        mode_name="Induced Dipole",
        exploratory_enabled=True,
        exploratory_supported=True,
        runtime_unsafe_reason="force cap exceeded",
    )
    assert assessment.state == START_UNSAFE_ABORT
    assert not assessment.start_allowed


def test_fixed_dipole_never_requires_induced_override_logic():
    system = load_system_parameters(physical_preset="demo_calibrated_physical")
    plan = _plan(system, _fixed_model(system), [0.2, 0.5, 0.5], [0.8, 0.5, 0.5], exploratory_enabled=True, duration=0.5)
    assert plan.assessment.path_status == PATH_BLOCKED
    assert plan.assessment.state == START_ALLOWED_CAUTION
    assert plan.assessment.execution_mode == EXECUTION_FALLBACK_REGULATION
    assert not plan.assessment.start_requires_override
