"""Induced-dipole magnetic force model based on ``grad ||B||^2``."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from physics.dipole_force import DifferenceSettings
from physics.magnetic_field_model import FaceDipoleFieldModel


Vector = NDArray[np.float64]
Matrix = NDArray[np.float64]


@dataclass(frozen=True)
class InducedDipoleDiagnostics:
    """Local field and force diagnostics for induced-dipole mode."""

    field_norm: float
    grad_norm_sq_norm: float
    force_norm: float
    finite_difference_eps: float
    min_distance_to_boundary: float | None
    inside_admissible_region: bool | None


class InducedDipoleForceModel:
    """Force model ``F = (alpha / 2) * grad ||B||^2 = alpha * (dB/dx)^T B``."""

    def __init__(
        self,
        field_model: FaceDipoleFieldModel,
        alpha: float,
        diff: DifferenceSettings | None = None,
    ) -> None:
        self.field_model = field_model
        self.alpha = float(alpha)
        self.diff = DifferenceSettings() if diff is None else diff

    def actuator_bounds(self) -> tuple[Vector, Vector]:
        return self.field_model.actuator_bounds()

    def potential(self, x: Vector, u: Vector) -> float:
        field = self.field_model.field(x, u)
        return -0.5 * self.alpha * float(np.dot(field, field))

    def grad_field_energy(self, x: Vector, u: Vector) -> Vector:
        """Return ``grad ||B||^2`` at ``(x, u)``."""

        field = self.field_model.field(x, u)
        field_jacobian = self.field_model.field_jacobian(x, u)
        return 2.0 * field_jacobian.T @ field

    def force(self, x: Vector, u: Vector) -> Vector:
        return 0.5 * self.alpha * self.grad_field_energy(x, u)

    def force_from_gradient_fd(self, x: Vector, u: Vector, position_eps: float | None = None) -> Vector:
        """Legacy central-difference estimate used for sensitivity diagnostics."""

        eps = self.diff.position_eps if position_eps is None else float(position_eps)
        point = np.asarray(x, dtype=float)
        gradient = np.zeros(3, dtype=float)
        for axis in range(3):
            delta = np.zeros(3, dtype=float)
            delta[axis] = eps
            phi_plus = -self.potential(point + delta, u)
            phi_minus = -self.potential(point - delta, u)
            gradient[axis] = (phi_plus - phi_minus) / (2.0 * eps)
        return gradient

    def input_jacobian(self, x: Vector, u: Vector) -> Matrix:
        basis, basis_jacobians = self.field_model.field_basis_and_jacobians(x)
        u_vec = np.asarray(u, dtype=float)
        field = basis @ u_vec
        field_jacobian = np.tensordot(u_vec, np.asarray(basis_jacobians, dtype=float), axes=(0, 0))
        columns = [
            self.alpha * (basis_jacobians[index].T @ field + field_jacobian.T @ basis[:, index])
            for index in range(basis.shape[1])
        ]
        return np.column_stack(columns)

    def state_jacobian(self, x: Vector, u: Vector) -> Matrix:
        point = np.asarray(x, dtype=float)
        columns = []
        for axis in range(3):
            delta = np.zeros(3, dtype=float)
            delta[axis] = self.diff.position_eps
            columns.append(
                (self.force(point + delta, u) - self.force(point - delta, u)) / (2.0 * self.diff.position_eps)
            )
        return np.column_stack(columns)

    def diagnostics(self, x: Vector, u: Vector, ball_radius: float | None = None) -> InducedDipoleDiagnostics:
        """Return local diagnostic quantities used by scripts, tests, and UI."""

        field = self.field_model.field(x, u)
        grad_norm_sq = self.grad_field_energy(x, u)
        inside = None
        min_distance = None
        if ball_radius is not None:
            inside = self.field_model.cube.contains(x, ball_radius)
            min_distance = self.field_model.cube.min_distance_to_boundary(x, ball_radius)
        return InducedDipoleDiagnostics(
            field_norm=float(np.linalg.norm(field)),
            grad_norm_sq_norm=float(np.linalg.norm(grad_norm_sq)),
            force_norm=0.5 * self.alpha * float(np.linalg.norm(grad_norm_sq)),
            finite_difference_eps=float(self.diff.position_eps),
            min_distance_to_boundary=min_distance,
            inside_admissible_region=inside,
        )
