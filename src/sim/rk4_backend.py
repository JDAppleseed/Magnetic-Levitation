"""RK4 backend for translation-only dynamics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from analysis.logs import SimulationLog
from sim.backend_interface import StepResult
from sim.contact_utils import apply_hard_wall_contact
from sim.state import RigidBodyState


Vector = NDArray[np.float64]


@dataclass(frozen=True)
class NullSnapshot:
    """Default diagnostics when no feedback controller is present."""

    target_position: Vector
    reference_velocity: Vector
    reference_acceleration: Vector
    error: Vector
    error_rate: Vector
    integral_state: Vector
    commanded_force: Vector
    achieved_force: Vector
    proof_regime: str

    @staticmethod
    def from_state(state: RigidBodyState) -> "NullSnapshot":
        zeros = np.zeros(3, dtype=float)
        return NullSnapshot(
            target_position=state.position.copy(),
            reference_velocity=zeros.copy(),
            reference_acceleration=zeros.copy(),
            error=zeros.copy(),
            error_rate=zeros.copy(),
            integral_state=zeros.copy(),
            commanded_force=zeros.copy(),
            achieved_force=zeros.copy(),
            proof_regime="green",
        )


class RK4Backend:
    """Direct RK4 integrator with optional explicit contact projection."""

    def __init__(
        self,
        side_length: float,
        ball_radius: float,
        mass: float,
        gravity: float,
        damping: float = 0.0,
        contact_mode: str = "hard_wall",
        restitution: float = 0.0,
        enable_contact: bool = True,
    ) -> None:
        self.side_length = float(side_length)
        self.ball_radius = float(ball_radius)
        self.mass = float(mass)
        self.gravity = float(gravity)
        self.damping = float(damping)
        self.contact_mode = contact_mode
        self.restitution = float(restitution)
        self.enable_contact = enable_contact
        self.lower = np.full(3, self.ball_radius, dtype=float)
        self.upper = np.full(3, self.side_length - self.ball_radius, dtype=float)

    def step(self, state: RigidBodyState, force_model: object, u: Vector, dt: float) -> StepResult:
        """Advance one timestep with zero-order-held input."""

        u = np.asarray(u, dtype=float)
        stage_projection_count = 0

        def admissible_position(position: Vector) -> Vector:
            nonlocal stage_projection_count
            point = np.asarray(position, dtype=float)
            if self.enable_contact and np.any((point < self.lower) | (point > self.upper)):
                stage_projection_count += 1
                return np.clip(point, self.lower, self.upper)
            return point

        def derivative(position: Vector, velocity: Vector) -> tuple[Vector, Vector]:
            eval_position = admissible_position(position)
            magnetic_force = force_model.force(eval_position, u)
            damping_force = -self.damping * velocity
            acceleration = (magnetic_force + damping_force + np.array([0.0, 0.0, -self.mass * self.gravity])) / self.mass
            return velocity, acceleration

        x0 = state.position
        v0 = state.velocity
        k1_x, k1_v = derivative(x0, v0)
        k2_x, k2_v = derivative(x0 + 0.5 * dt * k1_x, v0 + 0.5 * dt * k1_v)
        k3_x, k3_v = derivative(x0 + 0.5 * dt * k2_x, v0 + 0.5 * dt * k2_v)
        k4_x, k4_v = derivative(x0 + dt * k3_x, v0 + dt * k3_v)
        next_position = x0 + (dt / 6.0) * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x)
        next_velocity = v0 + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)

        contact_force = np.zeros(3, dtype=float)
        if self.enable_contact and self.contact_mode == "hard_wall":
            contact = apply_hard_wall_contact(
                next_position,
                next_velocity,
                self.lower,
                self.upper,
                self.mass,
                dt,
                restitution=self.restitution,
            )
            next_position = contact.position
            next_velocity = contact.velocity
            contact_force = contact.contact_force

        magnetic_force = force_model.force(next_position, u)
        damping_force = -self.damping * next_velocity
        net_acceleration = (
            magnetic_force + damping_force + contact_force + np.array([0.0, 0.0, -self.mass * self.gravity])
        ) / self.mass
        inside_admissible_region = bool(np.all(next_position >= self.lower - 1.0e-12) and np.all(next_position <= self.upper + 1.0e-12))
        boundary_distance = float(np.min(np.minimum(next_position - self.lower, self.upper - next_position)))
        return StepResult(
            state=RigidBodyState(next_position, next_velocity, state.time + dt),
            magnetic_force=magnetic_force,
            damping_force=damping_force,
            contact_force=contact_force,
            net_acceleration=net_acceleration,
            stage_projection_count=stage_projection_count,
            inside_admissible_region=inside_admissible_region,
            boundary_distance=boundary_distance,
        )

    def simulate(
        self,
        initial_state: RigidBodyState,
        force_model: object,
        duration: float,
        dt: float,
        controller: object | None = None,
        control_input_fn: Callable[[float, RigidBodyState], Vector] | None = None,
        target_provider: Callable[[float], Vector] | None = None,
    ) -> SimulationLog:
        """Run an open-loop or closed-loop simulation."""

        if controller is None and control_input_fn is None:
            raise ValueError("Provide a controller or a control_input_fn.")

        state = initial_state.copy()
        log = SimulationLog()
        steps = int(np.ceil(duration / dt))
        for _ in range(steps):
            if controller is not None:
                if target_provider is None:
                    u, snapshot = controller.compute_regulation(state.position, state.velocity, state.position, dt)
                else:
                    reference = target_provider(state.time)
                    if hasattr(reference, "position"):
                        u, snapshot = controller.compute_tracking(
                            state.position,
                            state.velocity,
                            reference.position,
                            reference.velocity,
                            reference.acceleration,
                            dt,
                        )
                    else:
                        u, snapshot = controller.compute_regulation(state.position, state.velocity, reference, dt)
            else:
                u = control_input_fn(state.time, state)
                snapshot = NullSnapshot.from_state(state)
            step = self.step(state, force_model, u, dt)
            log.append(
                time=step.state.time,
                position=step.state.position,
                velocity=step.state.velocity,
                acceleration=step.net_acceleration,
                control_input=u,
                commanded_force=snapshot.commanded_force,
                achieved_force=step.magnetic_force,
                tracking_error=snapshot.error,
                residual_norm=float(np.linalg.norm(step.magnetic_force - snapshot.commanded_force)),
                proof_regime=snapshot.proof_regime,
                contact_force=step.contact_force,
                damping_force=step.damping_force,
                saturated=getattr(snapshot, "allocation", None).saturated if hasattr(snapshot, "allocation") else False,
                stage_projection_count=step.stage_projection_count,
                inside_admissible=step.inside_admissible_region,
                boundary_distance=0.0 if step.boundary_distance is None else step.boundary_distance,
            )
            state = step.state
        return log
