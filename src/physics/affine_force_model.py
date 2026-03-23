"""Control-affine surrogate magnetic force model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from physics.magnetic_field_model import CubeGeometry, FaceActuatorSpec


Vector = NDArray[np.float64]
Matrix = NDArray[np.float64]


@dataclass(frozen=True)
class AuthoritySnapshot:
    """Local force-authority summary at a point."""

    rank: int
    singular_values: Vector
    condition_number: float


class AffineFaceMagneticForceModel:
    """Inverse-distance-decay affine face-actuator surrogate."""

    def __init__(self, cube: CubeGeometry, actuators: tuple[FaceActuatorSpec, ...]) -> None:
        self.cube = cube
        self.actuators = actuators

    def actuator_bounds(self) -> tuple[Vector, Vector]:
        lower = np.array([spec.u_min for spec in self.actuators], dtype=float)
        upper = np.array([spec.u_max for spec in self.actuators], dtype=float)
        return lower, upper

    def inward_distances(self, x: Vector) -> Vector:
        x = np.asarray(x, dtype=float)
        return np.array(
            [float(np.dot(spec.normal, x - spec.location)) for spec in self.actuators],
            dtype=float,
        )

    def force_matrix(self, x: Vector) -> Matrix:
        x = np.asarray(x, dtype=float)
        return np.column_stack([self._basis_vector(spec, x) for spec in self.actuators])

    def force(self, x: Vector, u: Vector) -> Vector:
        return self.force_matrix(x) @ np.asarray(u, dtype=float)

    def input_jacobian(self, x: Vector, u: Vector | None = None) -> Matrix:
        del u
        return self.force_matrix(x)

    def basis_jacobians(self, x: Vector) -> NDArray[np.float64]:
        x = np.asarray(x, dtype=float)
        jacobians = []
        for spec in self.actuators:
            distance = float(np.dot(spec.normal, x - spec.location))
            scale = -spec.affine_gain * spec.decay_power / (distance + spec.decay_offset) ** (
                spec.decay_power + 1.0
            )
            jacobians.append(scale * np.outer(spec.normal, spec.normal))
        return np.asarray(jacobians, dtype=float)

    def state_jacobian(self, x: Vector, u: Vector) -> Matrix:
        u = np.asarray(u, dtype=float)
        return np.tensordot(u, self.basis_jacobians(x), axes=(0, 0))

    def authority(self, x: Vector) -> AuthoritySnapshot:
        singular_values = np.linalg.svd(self.force_matrix(x), compute_uv=False)
        rank = int(np.linalg.matrix_rank(self.force_matrix(x)))
        if singular_values[-1] <= 1.0e-12:
            condition_number = float(np.inf)
        else:
            condition_number = float(singular_values[0] / singular_values[-1])
        return AuthoritySnapshot(rank=rank, singular_values=singular_values, condition_number=condition_number)

    @staticmethod
    def _basis_vector(spec: FaceActuatorSpec, x: Vector) -> Vector:
        distance = float(np.dot(spec.normal, x - spec.location))
        scale = spec.affine_gain / (distance + spec.decay_offset) ** spec.decay_power
        return scale * spec.normal

