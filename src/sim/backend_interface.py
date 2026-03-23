"""Backend dataclasses shared by all simulators."""

from __future__ import annotations

from dataclasses import dataclass

from numpy.typing import NDArray

from sim.state import RigidBodyState


Vector = NDArray[float]


@dataclass(frozen=True)
class StepResult:
    """State and explicit forces returned by one backend step."""

    state: RigidBodyState
    magnetic_force: Vector
    damping_force: Vector
    contact_force: Vector
    net_acceleration: Vector
    stage_projection_count: int = 0
    inside_admissible_region: bool = True
    boundary_distance: float | None = None
