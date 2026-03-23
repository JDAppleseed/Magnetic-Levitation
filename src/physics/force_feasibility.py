"""Force-feasibility and hover diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from control.allocator import AllocationResult, allocate_force_request
from physics.magnetic_field_model import E_Z


Vector = NDArray[np.float64]


@dataclass(frozen=True)
class HoverFeasibility:
    """Feasibility summary for hover at a point."""

    position: Vector
    required_force: Vector
    allocation: AllocationResult


@dataclass(frozen=True)
class TrajectoryFeasibilitySummary:
    """Pointwise force-feasibility summary along a reference."""

    max_residual_norm: float
    min_saturation_margin: float
    all_feasible: bool
    samples: list[AllocationResult]


def hover_force(mass: float, gravity: float) -> Vector:
    return float(mass * gravity) * E_Z


def evaluate_force_request(
    force_model: object,
    x: Vector,
    requested_force: Vector,
    mode: str = "bounded_ls",
    u_seed: Vector | None = None,
) -> AllocationResult:
    return allocate_force_request(force_model, np.asarray(x, dtype=float), requested_force, mode=mode, u_seed=u_seed)


def evaluate_hover_feasibility(
    force_model: object,
    x: Vector,
    mass: float,
    gravity: float,
    mode: str = "bounded_ls",
) -> HoverFeasibility:
    required = hover_force(mass, gravity)
    allocation = evaluate_force_request(force_model, x, required, mode=mode)
    return HoverFeasibility(position=np.asarray(x, dtype=float), required_force=required, allocation=allocation)


def evaluate_trajectory_force_feasibility(
    force_model: object,
    trajectory: object,
    mass: float,
    gravity: float,
    sample_count: int = 101,
    mode: str = "bounded_ls",
) -> TrajectoryFeasibilitySummary:
    samples: list[AllocationResult] = []
    for time in np.linspace(trajectory.start_time, trajectory.end_time, sample_count):
        sample = trajectory.evaluate(float(time))
        requested_force = hover_force(mass, gravity) + mass * sample.acceleration
        samples.append(evaluate_force_request(force_model, sample.position, requested_force, mode=mode))
    max_residual_norm = max(item.residual_norm for item in samples)
    min_saturation_margin = min(1.0 - item.saturation_fraction for item in samples)
    return TrajectoryFeasibilitySummary(
        max_residual_norm=max_residual_norm,
        min_saturation_margin=min_saturation_margin,
        all_feasible=all(item.feasible for item in samples),
        samples=samples,
    )

