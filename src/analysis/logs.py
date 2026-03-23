"""Simulation logging structures."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


Vector = NDArray[np.float64]


@dataclass
class SimulationLog:
    """Time history collected during a simulation run."""

    times: list[float] = field(default_factory=list)
    positions: list[Vector] = field(default_factory=list)
    velocities: list[Vector] = field(default_factory=list)
    accelerations: list[Vector] = field(default_factory=list)
    control_inputs: list[Vector] = field(default_factory=list)
    commanded_forces: list[Vector] = field(default_factory=list)
    achieved_forces: list[Vector] = field(default_factory=list)
    tracking_errors: list[Vector] = field(default_factory=list)
    residual_norms: list[float] = field(default_factory=list)
    proof_regimes: list[str] = field(default_factory=list)
    contact_forces: list[Vector] = field(default_factory=list)
    damping_forces: list[Vector] = field(default_factory=list)
    saturation_flags: list[bool] = field(default_factory=list)
    stage_projection_counts: list[int] = field(default_factory=list)
    inside_admissible_flags: list[bool] = field(default_factory=list)
    boundary_distances: list[float] = field(default_factory=list)

    def append(
        self,
        time: float,
        position: Vector,
        velocity: Vector,
        acceleration: Vector,
        control_input: Vector,
        commanded_force: Vector,
        achieved_force: Vector,
        tracking_error: Vector,
        residual_norm: float,
        proof_regime: str,
        contact_force: Vector,
        damping_force: Vector,
        saturated: bool,
        stage_projection_count: int = 0,
        inside_admissible: bool = True,
        boundary_distance: float = 0.0,
    ) -> None:
        self.times.append(float(time))
        self.positions.append(np.asarray(position, dtype=float))
        self.velocities.append(np.asarray(velocity, dtype=float))
        self.accelerations.append(np.asarray(acceleration, dtype=float))
        self.control_inputs.append(np.asarray(control_input, dtype=float))
        self.commanded_forces.append(np.asarray(commanded_force, dtype=float))
        self.achieved_forces.append(np.asarray(achieved_force, dtype=float))
        self.tracking_errors.append(np.asarray(tracking_error, dtype=float))
        self.residual_norms.append(float(residual_norm))
        self.proof_regimes.append(str(proof_regime))
        self.contact_forces.append(np.asarray(contact_force, dtype=float))
        self.damping_forces.append(np.asarray(damping_force, dtype=float))
        self.saturation_flags.append(bool(saturated))
        self.stage_projection_counts.append(int(stage_projection_count))
        self.inside_admissible_flags.append(bool(inside_admissible))
        self.boundary_distances.append(float(boundary_distance))

    def as_array(self, field_name: str) -> NDArray[np.float64]:
        """Convert a vector-valued field into an array."""

        values = getattr(self, field_name)
        return np.asarray(values, dtype=float)

    def rms_tracking_error(self) -> float:
        """Return RMS position tracking error."""

        errors = self.as_array("tracking_errors")
        return float(np.sqrt(np.mean(np.sum(errors**2, axis=1))))
