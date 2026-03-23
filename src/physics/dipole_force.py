"""Fixed-dipole magnetic force model using a dipole field superposition."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from physics.magnetic_field_model import FaceDipoleFieldModel


Vector = NDArray[np.float64]
Matrix = NDArray[np.float64]


@dataclass(frozen=True)
class DifferenceSettings:
    """Finite-difference settings for physical-force evaluation."""

    position_eps: float = 1.0e-5
    input_eps: float = 1.0e-5


class FixedDipoleForceModel:
    """Force model F = grad(mu dot B)."""

    def __init__(
        self,
        field_model: FaceDipoleFieldModel,
        magnetic_moment: Vector,
        diff: DifferenceSettings | None = None,
    ) -> None:
        self.field_model = field_model
        self.magnetic_moment = np.asarray(magnetic_moment, dtype=float)
        self.diff = DifferenceSettings() if diff is None else diff

    def actuator_bounds(self) -> tuple[Vector, Vector]:
        return self.field_model.actuator_bounds()

    def scalar_potential(self, x: Vector, u: Vector) -> float:
        return -float(np.dot(self.magnetic_moment, self.field_model.field(x, u)))

    def force(self, x: Vector, u: Vector) -> Vector:
        field_jacobian = self.field_model.field_jacobian(x, u)
        return field_jacobian.T @ self.magnetic_moment

    def force_from_gradient_fd(self, x: Vector, u: Vector, position_eps: float | None = None) -> Vector:
        """Legacy central-difference estimate used for diagnostics."""

        eps = self.diff.position_eps if position_eps is None else float(position_eps)
        point = np.asarray(x, dtype=float)
        gradient = np.zeros(3, dtype=float)
        for axis in range(3):
            delta = np.zeros(3, dtype=float)
            delta[axis] = eps
            phi_plus = -self.scalar_potential(point + delta, u)
            phi_minus = -self.scalar_potential(point - delta, u)
            gradient[axis] = (phi_plus - phi_minus) / (2.0 * eps)
        return gradient

    def force_matrix(self, x: Vector) -> Matrix:
        """Return the affine force map ``F(x, u) = G(x) u`` for fixed-dipole actuation."""

        return self.input_jacobian(x)

    def input_jacobian(self, x: Vector, u: Vector | None = None) -> Matrix:
        del u
        basis_jacobians = self.field_model.field_basis_jacobians(x)
        columns = [basis_jacobian.T @ self.magnetic_moment for basis_jacobian in basis_jacobians]
        return np.column_stack(columns)

    def state_jacobian(self, x: Vector, u: Vector) -> Matrix:
        x = np.asarray(x, dtype=float)
        columns = []
        for axis in range(3):
            delta = np.zeros(3, dtype=float)
            delta[axis] = self.diff.position_eps
            columns.append(
                (self.force(x + delta, u) - self.force(x - delta, u)) / (2.0 * self.diff.position_eps)
            )
        return np.column_stack(columns)
