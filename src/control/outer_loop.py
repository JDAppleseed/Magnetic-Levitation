"""Outer-loop force command laws from Problems 3 and 4."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from physics.magnetic_field_model import E_Z


Vector = NDArray[np.float64]


@dataclass(frozen=True)
class DiagonalGains:
    """Diagonal gain matrices stored as 3-vectors."""

    kp: Vector
    kd: Vector
    ki: Vector

    def __post_init__(self) -> None:
        for gain in (self.kp, self.kd, self.ki):
            if np.asarray(gain, dtype=float).shape != (3,):
                raise ValueError("Gain vectors must have shape (3,).")

    @property
    def Kp(self) -> NDArray[np.float64]:
        return np.diag(self.kp)

    @property
    def Kd(self) -> NDArray[np.float64]:
        return np.diag(self.kd)

    @property
    def Ki(self) -> NDArray[np.float64]:
        return np.diag(self.ki)


def regulation_force_command(
    mass: float,
    gravity: float,
    error: Vector,
    error_rate: Vector,
    integral_state: Vector,
    gains: DiagonalGains,
) -> Vector:
    """Problem 3 force law for a static target."""

    return mass * gravity * E_Z - gains.Kp @ error - gains.Kd @ error_rate - gains.Ki @ integral_state


def tracking_force_command(
    mass: float,
    gravity: float,
    reference_acceleration: Vector,
    error: Vector,
    error_rate: Vector,
    integral_state: Vector,
    gains: DiagonalGains,
) -> Vector:
    """Problem 4 force law for a moving reference."""

    return (
        mass * gravity * E_Z
        + mass * np.asarray(reference_acceleration, dtype=float)
        - gains.Kp @ error
        - gains.Kd @ error_rate
        - gains.Ki @ integral_state
    )

