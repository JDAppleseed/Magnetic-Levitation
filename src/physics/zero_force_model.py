"""Explicit gravity-only mode with no magnetic actuation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


Vector = NDArray[np.float64]
Matrix = NDArray[np.float64]


class ZeroMagneticForceModel:
    """Force model used for free dynamics validation."""

    def __init__(self, actuator_count: int = 6) -> None:
        self.actuator_count = int(actuator_count)

    def actuator_bounds(self) -> tuple[Vector, Vector]:
        zeros = np.zeros(self.actuator_count, dtype=float)
        return zeros.copy(), zeros.copy()

    def force(self, x: Vector, u: Vector) -> Vector:
        del x, u
        return np.zeros(3, dtype=float)

    def input_jacobian(self, x: Vector, u: Vector | None = None) -> Matrix:
        del x, u
        return np.zeros((3, self.actuator_count), dtype=float)

    def state_jacobian(self, x: Vector, u: Vector) -> Matrix:
        del x, u
        return np.zeros((3, 3), dtype=float)

