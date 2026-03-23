"""State dataclasses for translational simulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


Vector = NDArray[np.float64]


@dataclass
class RigidBodyState:
    """Translational state of the spherical robot."""

    position: Vector
    velocity: Vector
    time: float = 0.0

    def copy(self) -> "RigidBodyState":
        return RigidBodyState(self.position.copy(), self.velocity.copy(), self.time)

    @classmethod
    def from_iterables(cls, position: list[float], velocity: list[float], time: float = 0.0) -> "RigidBodyState":
        return cls(np.asarray(position, dtype=float), np.asarray(velocity, dtype=float), float(time))

