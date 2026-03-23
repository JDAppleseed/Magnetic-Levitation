"""Grid sampling helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


Vector = NDArray[np.float64]


def sample_cube_interior(lower: Vector, upper: Vector, samples_per_axis: int) -> list[Vector]:
    """Return a uniform interior grid."""

    axes = [
        np.linspace(lower[index], upper[index], samples_per_axis, dtype=float)
        for index in range(3)
    ]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    return [point.astype(float) for point in points]


def sample_admissible_region(cube: object, ball_radius: float, samples_per_axis: int) -> list[Vector]:
    """Sample the admissible center-of-mass region."""

    lower, upper = cube.admissible_bounds(ball_radius)
    return sample_cube_interior(lower, upper, samples_per_axis)

