"""Safety and feasibility helpers for physical magnetic modes."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from numpy.typing import NDArray

from physics.force_feasibility import HoverFeasibility, evaluate_hover_feasibility
from physics.induced_dipole_force import InducedDipoleDiagnostics, InducedDipoleForceModel
from physics.magnetic_field_model import InducedModeParameters


Vector = NDArray[np.float64]


@dataclass(frozen=True)
class InducedHoverAssessment:
    """Hover-feasibility and numerical-stability summary for induced mode."""

    hover: HoverFeasibility
    diagnostics: InducedDipoleDiagnostics
    force_to_weight_ratio: float
    stable: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class GuardResult:
    """Per-step guard result for induced mode."""

    allowed: bool
    aborted: bool
    status: str
    reasons: tuple[str, ...]
    u: Vector
    achieved_force: Vector
    diagnostics: InducedDipoleDiagnostics


def induced_start_allowed(assessment: InducedHoverAssessment, exploratory_enabled: bool) -> tuple[bool, str]:
    """Return whether induced closed-loop start is permitted under the current policy."""

    if assessment.hover.allocation.feasible and assessment.stable:
        return True, "ready"
    if exploratory_enabled:
        return True, "guarded_exploratory"
    if not assessment.hover.allocation.feasible:
        return False, "target hover infeasible"
    return False, "operating point outside stable regime"


def corner_sampled_max_force_norm(force_model: object, x: Vector) -> float:
    """Return a corner-sampled estimate of the maximum achievable force norm."""

    lower, upper = force_model.actuator_bounds()
    corners = product(*[(lo, hi) for lo, hi in zip(lower, upper)])
    max_norm = 0.0
    point = np.asarray(x, dtype=float)
    for corner in corners:
        force = np.asarray(force_model.force(point, np.asarray(corner, dtype=float)), dtype=float)
        max_norm = max(max_norm, float(np.linalg.norm(force)))
    return max_norm


def assess_induced_hover(
    force_model: InducedDipoleForceModel,
    x: Vector,
    mass: float,
    gravity: float,
    ball_radius: float,
    safety: InducedModeParameters,
    mode: str = "nonlinear",
) -> InducedHoverAssessment:
    """Assess whether induced-dipole hover is feasible and numerically well-conditioned."""

    hover = evaluate_hover_feasibility(force_model, x, mass, gravity, mode=mode)
    diagnostics = force_model.diagnostics(x, hover.allocation.u, ball_radius)
    force_to_weight_ratio = corner_sampled_max_force_norm(force_model, x) / max(mass * gravity, 1.0e-12)
    reasons: list[str] = []
    if not hover.allocation.feasible:
        reasons.append("target hover infeasible")
    if diagnostics.inside_admissible_region is False:
        reasons.append("state outside admissible region")
    if diagnostics.min_distance_to_boundary is not None and diagnostics.min_distance_to_boundary < safety.min_boundary_margin:
        reasons.append("too close to admissible boundary")
    if diagnostics.grad_norm_sq_norm > safety.max_gradient_norm:
        reasons.append("gradient norm exceeds safety threshold")
    stable = len(reasons) == 0
    return InducedHoverAssessment(
        hover=hover,
        diagnostics=diagnostics,
        force_to_weight_ratio=force_to_weight_ratio,
        stable=stable,
        reasons=tuple(reasons),
    )


def guard_induced_command(
    force_model: InducedDipoleForceModel,
    position: Vector,
    velocity: Vector,
    proposed_u: Vector,
    previous_u: Vector,
    dt: float,
    ball_radius: float,
    safety: InducedModeParameters,
    exploratory_enabled: bool,
) -> GuardResult:
    """Apply explicit induced-mode safety checks and optional exploratory slew-rate limits."""

    lower, upper = force_model.actuator_bounds()
    proposed = np.clip(np.asarray(proposed_u, dtype=float), lower, upper)
    prior = np.asarray(previous_u, dtype=float)
    if exploratory_enabled:
        max_delta = safety.actuator_slew_rate_limit * dt
        proposed = np.clip(proposed, prior - max_delta, prior + max_delta)
        proposed = np.clip(proposed, lower, upper)

    diagnostics = force_model.diagnostics(position, proposed, ball_radius)
    achieved_force = np.asarray(force_model.force(position, proposed), dtype=float)
    reasons: list[str] = []
    if diagnostics.inside_admissible_region is False:
        reasons.append("state outside admissible region")
    if diagnostics.min_distance_to_boundary is not None and diagnostics.min_distance_to_boundary < safety.min_boundary_margin:
        reasons.append("boundary margin violated")
    if np.linalg.norm(velocity) > safety.max_velocity_norm:
        reasons.append("velocity norm exceeds safety threshold")
    if diagnostics.force_norm > safety.max_force_norm:
        reasons.append("force norm exceeds safety threshold")
    if diagnostics.grad_norm_sq_norm > safety.max_gradient_norm:
        reasons.append("gradient norm exceeds safety threshold")
    if reasons:
        return GuardResult(
            allowed=False,
            aborted=True,
            status="unsafe_abort",
            reasons=tuple(reasons),
            u=np.zeros_like(proposed),
            achieved_force=np.zeros(3, dtype=float),
            diagnostics=diagnostics,
        )
    status = "exploratory_guarded" if exploratory_enabled and not np.allclose(proposed, proposed_u) else "ok"
    return GuardResult(
        allowed=True,
        aborted=False,
        status=status,
        reasons=tuple(),
        u=proposed,
        achieved_force=achieved_force,
        diagnostics=diagnostics,
    )
