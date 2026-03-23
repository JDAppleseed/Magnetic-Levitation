"""Explicit hard-wall contact handling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


Vector = NDArray[np.float64]


@dataclass(frozen=True)
class ContactResult:
    """Projection-based contact response."""

    position: Vector
    velocity: Vector
    contact_force: Vector
    active_faces: tuple[str, ...]


def apply_hard_wall_contact(
    position: Vector,
    velocity: Vector,
    lower: Vector,
    upper: Vector,
    mass: float,
    dt: float,
    restitution: float = 0.0,
) -> ContactResult:
    """Project the state into bounds and remove outward velocity."""

    corrected_position = np.asarray(position, dtype=float).copy()
    corrected_velocity = np.asarray(velocity, dtype=float).copy()
    contact_force = np.zeros(3, dtype=float)
    active_faces: list[str] = []
    face_labels = (("x-", "x+"), ("y-", "y+"), ("z-", "z+"))

    for axis in range(3):
        if corrected_position[axis] < lower[axis]:
            corrected_position[axis] = lower[axis]
            if corrected_velocity[axis] < 0.0:
                new_velocity = -restitution * corrected_velocity[axis]
                contact_force[axis] += mass * (new_velocity - corrected_velocity[axis]) / dt
                corrected_velocity[axis] = new_velocity
            active_faces.append(face_labels[axis][0])
        elif corrected_position[axis] > upper[axis]:
            corrected_position[axis] = upper[axis]
            if corrected_velocity[axis] > 0.0:
                new_velocity = -restitution * corrected_velocity[axis]
                contact_force[axis] += mass * (new_velocity - corrected_velocity[axis]) / dt
                corrected_velocity[axis] = new_velocity
            active_faces.append(face_labels[axis][1])

    return ContactResult(
        position=corrected_position,
        velocity=corrected_velocity,
        contact_force=contact_force,
        active_faces=tuple(active_faces),
    )

