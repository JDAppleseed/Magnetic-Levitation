"""Two-layer closed-loop controllers for regulation and tracking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from control.allocator import AllocationResult, allocate_force_request
from control.outer_loop import DiagonalGains, regulation_force_command, tracking_force_command


Vector = NDArray[np.float64]


@dataclass(frozen=True)
class ControllerSnapshot:
    """Force-command and allocation diagnostics for one control update."""

    target_position: Vector
    reference_velocity: Vector
    reference_acceleration: Vector
    error: Vector
    error_rate: Vector
    integral_state: Vector
    commanded_force: Vector
    achieved_force: Vector
    allocation: AllocationResult
    proof_regime: str


class TwoLayerController:
    """Outer-loop force controller plus inner-loop actuator allocator."""

    def __init__(
        self,
        force_model: object,
        mass: float,
        gravity: float,
        gains: DiagonalGains,
        allocator_mode: str = "bounded_ls",
        integral_limit: float = 0.25,
        local_regime_radius: float = 0.15,
    ) -> None:
        self.force_model = force_model
        self.mass = float(mass)
        self.gravity = float(gravity)
        self.gains = gains
        self.allocator_mode = allocator_mode
        self.integral_limit = float(integral_limit)
        self.local_regime_radius = float(local_regime_radius)
        self._eta = np.zeros(3, dtype=float)
        lower, upper = self.force_model.actuator_bounds()
        self._u_last = 0.5 * (lower + upper)

    def reset(self) -> None:
        self._eta = np.zeros(3, dtype=float)
        lower, upper = self.force_model.actuator_bounds()
        self._u_last = 0.5 * (lower + upper)

    def set_last_input(self, u: Vector) -> None:
        """Record the last actuator vector actually applied to the plant."""

        lower, upper = self.force_model.actuator_bounds()
        self._u_last = np.clip(np.asarray(u, dtype=float), lower, upper)

    def compute_regulation(self, position: Vector, velocity: Vector, target: Vector, dt: float) -> tuple[Vector, ControllerSnapshot]:
        reference_velocity = np.zeros(3, dtype=float)
        reference_acceleration = np.zeros(3, dtype=float)
        error = np.asarray(position, dtype=float) - np.asarray(target, dtype=float)
        error_rate = np.asarray(velocity, dtype=float) - reference_velocity
        return self._compute_common(target, reference_velocity, reference_acceleration, error, error_rate, dt, tracking=False)

    def compute_tracking(
        self,
        position: Vector,
        velocity: Vector,
        reference_position: Vector,
        reference_velocity: Vector,
        reference_acceleration: Vector,
        dt: float,
    ) -> tuple[Vector, ControllerSnapshot]:
        error = np.asarray(position, dtype=float) - np.asarray(reference_position, dtype=float)
        error_rate = np.asarray(velocity, dtype=float) - np.asarray(reference_velocity, dtype=float)
        return self._compute_common(
            np.asarray(reference_position, dtype=float),
            np.asarray(reference_velocity, dtype=float),
            np.asarray(reference_acceleration, dtype=float),
            error,
            error_rate,
            dt,
            tracking=True,
        )

    def _compute_common(
        self,
        reference_position: Vector,
        reference_velocity: Vector,
        reference_acceleration: Vector,
        error: Vector,
        error_rate: Vector,
        dt: float,
        tracking: bool,
    ) -> tuple[Vector, ControllerSnapshot]:
        eta_candidate = np.clip(self._eta + error * dt, -self.integral_limit, self.integral_limit)
        if tracking:
            commanded_force = tracking_force_command(
                self.mass,
                self.gravity,
                reference_acceleration,
                error,
                error_rate,
                eta_candidate,
                self.gains,
            )
        else:
            commanded_force = regulation_force_command(
                self.mass,
                self.gravity,
                error,
                error_rate,
                eta_candidate,
                self.gains,
            )
        allocation = allocate_force_request(
            self.force_model,
            reference_position + error,
            commanded_force,
            mode=self.allocator_mode,
            u_seed=self._u_last,
        )
        if allocation.feasible and not allocation.saturated:
            self._eta = eta_candidate
        self._u_last = allocation.u
        proof_regime = self._proof_regime(error, allocation)
        snapshot = ControllerSnapshot(
            target_position=reference_position,
            reference_velocity=reference_velocity,
            reference_acceleration=reference_acceleration,
            error=error,
            error_rate=error_rate,
            integral_state=self._eta.copy(),
            commanded_force=commanded_force,
            achieved_force=allocation.achieved_force,
            allocation=allocation,
            proof_regime=proof_regime,
        )
        return allocation.u, snapshot

    def _proof_regime(self, error: Vector, allocation: AllocationResult) -> str:
        if not allocation.feasible or allocation.residual_norm > 1.0e-3:
            return "red"
        if np.linalg.norm(error) > self.local_regime_radius or allocation.saturation_fraction > 0.9:
            return "yellow"
        return "green"
