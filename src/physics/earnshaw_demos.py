"""Earnshaw-style static-potential diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from analysis.sampling import sample_cube_interior


Vector = NDArray[np.float64]
Matrix = NDArray[np.float64]


@dataclass(frozen=True)
class EarnshawSample:
    """Potential and Hessian diagnostic at one point."""

    position: Vector
    eigenvalues: Vector
    trace: float


@dataclass(frozen=True)
class EarnshawSummary:
    """Aggregate obstruction summary."""

    strict_minimum_count: int
    sample_count: int
    max_abs_trace: float
    samples: list[EarnshawSample]


def potential_hessian(potential_fn: object, x: Vector, eps: float = 5.0e-5) -> Matrix:
    """Centered finite-difference Hessian."""

    x = np.asarray(x, dtype=float)
    hessian = np.zeros((3, 3), dtype=float)
    f0 = potential_fn(x)
    for i in range(3):
        ei = np.zeros(3, dtype=float)
        ei[i] = eps
        for j in range(3):
            if i == j:
                hessian[i, j] = (potential_fn(x + ei) - 2.0 * f0 + potential_fn(x - ei)) / (eps * eps)
            else:
                ej = np.zeros(3, dtype=float)
                ej[j] = eps
                hessian[i, j] = (
                    potential_fn(x + ei + ej)
                    - potential_fn(x + ei - ej)
                    - potential_fn(x - ei + ej)
                    + potential_fn(x - ei - ej)
                ) / (4.0 * eps * eps)
    return hessian


def analyze_fixed_input_potential(
    potential_fn: object,
    lower: Vector,
    upper: Vector,
    samples_per_axis: int = 5,
) -> EarnshawSummary:
    """Sample an interior grid and count strict positive-definite minima."""

    points = sample_cube_interior(lower, upper, samples_per_axis)
    samples: list[EarnshawSample] = []
    strict_minimum_count = 0
    max_abs_trace = 0.0
    for point in points:
        hessian = potential_hessian(potential_fn, point)
        eigenvalues = np.linalg.eigvalsh(hessian)
        trace = float(np.trace(hessian))
        max_abs_trace = max(max_abs_trace, abs(trace))
        if np.all(eigenvalues > 1.0e-6):
            strict_minimum_count += 1
        samples.append(EarnshawSample(position=point, eigenvalues=eigenvalues, trace=trace))
    return EarnshawSummary(
        strict_minimum_count=strict_minimum_count,
        sample_count=len(samples),
        max_abs_trace=max_abs_trace,
        samples=samples,
    )
