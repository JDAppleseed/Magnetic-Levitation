"""Force-space actuator allocation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares, lsq_linear


Vector = NDArray[np.float64]
Matrix = NDArray[np.float64]


@dataclass(frozen=True)
class AllocationResult:
    """Detailed force-allocation result."""

    requested_force: Vector
    achieved_force: Vector
    u: Vector
    residual: Vector
    residual_norm: float
    feasible: bool
    saturated: bool
    saturation_fraction: float
    status: str


def _allocation_result(
    requested_force: Vector,
    achieved_force: Vector,
    u: Vector,
    lower: Vector,
    upper: Vector,
    status: str,
    feasible_tol: float = 1.0e-6,
) -> AllocationResult:
    residual = achieved_force - requested_force
    residual_norm = float(np.linalg.norm(residual))
    span = np.maximum(upper - lower, 1.0e-12)
    distance_to_bounds = np.minimum((u - lower) / span, (upper - u) / span)
    normalized_margin = float(np.clip(np.min(distance_to_bounds), 0.0, 0.5))
    saturation_fraction = 1.0 - 2.0 * normalized_margin
    saturated = bool(np.any(np.isclose(u, lower, atol=1.0e-7)) or np.any(np.isclose(u, upper, atol=1.0e-7)))
    return AllocationResult(
        requested_force=requested_force,
        achieved_force=achieved_force,
        u=u,
        residual=residual,
        residual_norm=residual_norm,
        feasible=residual_norm <= feasible_tol,
        saturated=saturated,
        saturation_fraction=saturation_fraction,
        status=status,
    )


def allocate_affine_pseudoinverse(force_model: object, x: Vector, requested_force: Vector) -> AllocationResult:
    """Unconstrained minimum-norm allocator for the affine model."""

    G = np.asarray(force_model.force_matrix(x), dtype=float)
    lower, upper = force_model.actuator_bounds()
    u = np.linalg.pinv(G) @ np.asarray(requested_force, dtype=float)
    achieved_force = force_model.force(x, u)
    return _allocation_result(requested_force, achieved_force, u, lower, upper, "pseudoinverse")


def allocate_affine_bounded(force_model: object, x: Vector, requested_force: Vector) -> AllocationResult:
    """Bounded least-squares allocator for the affine model."""

    G = np.asarray(force_model.force_matrix(x), dtype=float)
    lower, upper = force_model.actuator_bounds()
    solve = lsq_linear(G, requested_force, bounds=(lower, upper), lsmr_tol="auto", verbose=0)
    u = solve.x.astype(float)
    achieved_force = force_model.force(x, u)
    return _allocation_result(requested_force, achieved_force, u, lower, upper, f"bounded_ls:{solve.status}")


def allocate_physical_nonlinear(
    force_model: object,
    x: Vector,
    requested_force: Vector,
    u_seed: Vector | None = None,
) -> AllocationResult:
    """Local bounded nonlinear least-squares allocator for a physical model."""

    lower, upper = force_model.actuator_bounds()
    if u_seed is None:
        u0 = 0.5 * (lower + upper)
    else:
        u0 = np.clip(np.asarray(u_seed, dtype=float), lower, upper)

    def residual(u: Vector) -> Vector:
        return force_model.force(x, u) - requested_force

    def jacobian(u: Vector) -> Matrix:
        return np.asarray(force_model.input_jacobian(x, u), dtype=float)

    solve = least_squares(
        residual,
        u0,
        jac=jacobian,
        bounds=(lower, upper),
        xtol=1.0e-10,
        ftol=1.0e-10,
        gtol=1.0e-10,
        max_nfev=50,
    )
    u = solve.x.astype(float)
    achieved_force = force_model.force(x, u)
    return _allocation_result(requested_force, achieved_force, u, lower, upper, f"nonlinear_ls:{solve.status}")


def allocate_physical_local_linearization(
    force_model: object,
    x: Vector,
    requested_force: Vector,
    u_seed: Vector | None = None,
    iterations: int = 4,
    regularization: float = 1.0e-7,
) -> AllocationResult:
    """Fast local nonlinear allocator using repeated bounded Jacobian solves."""

    lower, upper = force_model.actuator_bounds()
    if u_seed is None:
        u = 0.5 * (lower + upper)
    else:
        u = np.clip(np.asarray(u_seed, dtype=float), lower, upper)

    status = "local_linearized"
    for iteration in range(iterations):
        achieved_force = force_model.force(x, u)
        residual = np.asarray(requested_force, dtype=float) - achieved_force
        residual_norm = float(np.linalg.norm(residual))
        if residual_norm <= 1.0e-6:
            status = f"local_linearized:{iteration}"
            break
        jacobian = np.asarray(force_model.input_jacobian(x, u), dtype=float)
        augmented_matrix = np.vstack(
            [
                jacobian,
                np.sqrt(regularization) * np.eye(jacobian.shape[1], dtype=float),
            ]
        )
        augmented_target = np.concatenate([residual, np.zeros(jacobian.shape[1], dtype=float)])
        delta_lower = lower - u
        delta_upper = upper - u
        solve = lsq_linear(augmented_matrix, augmented_target, bounds=(delta_lower, delta_upper), lsmr_tol="auto")
        step = solve.x.astype(float)
        if np.linalg.norm(step) <= 1.0e-10:
            status = f"local_linearized_stalled:{iteration}"
            break
        best_u = u.copy()
        best_residual_norm = residual_norm
        for scale in (1.0, 0.5, 0.25, 0.1):
            candidate = np.clip(u + scale * step, lower, upper)
            candidate_force = force_model.force(x, candidate)
            candidate_residual_norm = float(np.linalg.norm(np.asarray(requested_force, dtype=float) - candidate_force))
            if candidate_residual_norm + 1.0e-10 < best_residual_norm:
                best_u = candidate
                best_residual_norm = candidate_residual_norm
        if np.allclose(best_u, u, atol=1.0e-12):
            status = f"local_linearized_stalled:{iteration}"
            break
        u = best_u
        status = f"local_linearized:{iteration}"

    achieved_force = force_model.force(x, u)
    return _allocation_result(requested_force, achieved_force, u, lower, upper, status)


def allocate_force_request(
    force_model: object,
    x: Vector,
    requested_force: Vector,
    mode: str = "bounded_ls",
    u_seed: Vector | None = None,
) -> AllocationResult:
    """Dispatch to an allocator compatible with the force model."""

    if hasattr(force_model, "force_matrix"):
        if mode == "pseudoinverse":
            return allocate_affine_pseudoinverse(force_model, x, requested_force)
        return allocate_affine_bounded(force_model, x, requested_force)
    if mode == "nonlinear_local":
        return allocate_physical_local_linearization(force_model, x, requested_force, u_seed=u_seed)
    return allocate_physical_nonlinear(force_model, x, requested_force, u_seed=u_seed)
