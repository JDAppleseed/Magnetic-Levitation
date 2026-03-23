"""Reachable-force summaries for the affine surrogate."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from numpy.typing import NDArray


Vector = NDArray[np.float64]
Matrix = NDArray[np.float64]


@dataclass(frozen=True)
class ReachableSetSummary:
    """Vertex and authority summary of the local reachable-force set."""

    vertices: Matrix
    centroid: Vector
    rank: int
    singular_values: Vector


def reachable_force_vertices(force_model: object, x: Vector) -> Matrix:
    """Enumerate box-image vertices for the affine model."""

    lower, upper = force_model.actuator_bounds()
    G = force_model.force_matrix(x)
    vertices = []
    for selector in product([0, 1], repeat=lower.size):
        u = np.array([upper[i] if bit else lower[i] for i, bit in enumerate(selector)], dtype=float)
        vertices.append(G @ u)
    return np.asarray(vertices, dtype=float)


def summarize_reachable_set(force_model: object, x: Vector) -> ReachableSetSummary:
    """Summarize the local reachable-force polytope."""

    vertices = reachable_force_vertices(force_model, x)
    G = force_model.force_matrix(x)
    singular_values = np.linalg.svd(G, compute_uv=False)
    return ReachableSetSummary(
        vertices=vertices,
        centroid=np.mean(vertices, axis=0),
        rank=int(np.linalg.matrix_rank(G)),
        singular_values=singular_values,
    )

