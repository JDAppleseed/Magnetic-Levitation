"""C^2 quintic trajectory generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


Vector = NDArray[np.float64]


@dataclass(frozen=True)
class TrajectorySample:
    """Position, velocity, and acceleration sample."""

    time: float
    position: Vector
    velocity: Vector
    acceleration: Vector


class QuinticTrajectory:
    """Minimum-jerk quintic point-to-point trajectory with zero endpoint velocity and acceleration."""

    def __init__(self, x0: Vector, x1: Vector, duration: float, start_time: float = 0.0) -> None:
        if duration <= 0.0:
            raise ValueError("Trajectory duration must be positive.")
        self.x0 = np.asarray(x0, dtype=float)
        self.x1 = np.asarray(x1, dtype=float)
        self.duration = float(duration)
        self.start_time = float(start_time)
        self.end_time = self.start_time + self.duration

    def evaluate(self, time: float) -> TrajectorySample:
        tau = np.clip((time - self.start_time) / self.duration, 0.0, 1.0)
        tau2 = tau * tau
        tau3 = tau2 * tau
        tau4 = tau3 * tau
        tau5 = tau4 * tau
        blend = 10.0 * tau3 - 15.0 * tau4 + 6.0 * tau5
        blend_dot = (30.0 * tau2 - 60.0 * tau3 + 30.0 * tau4) / self.duration
        blend_ddot = (60.0 * tau - 180.0 * tau2 + 120.0 * tau3) / (self.duration**2)
        delta = self.x1 - self.x0
        return TrajectorySample(
            time=float(time),
            position=self.x0 + blend * delta,
            velocity=blend_dot * delta,
            acceleration=blend_ddot * delta,
        )

