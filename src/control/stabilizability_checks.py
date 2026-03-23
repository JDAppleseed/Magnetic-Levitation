"""Problem 2 stabilizability diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


Matrix = NDArray[np.float64]


@dataclass(frozen=True)
class StabilizabilityReport:
    """Rank and PBH-style stabilizability result."""

    input_rank: int
    full_translational_authority: bool
    stabilizable: bool
    violating_eigenvalues: tuple[complex, ...]


def pbh_stabilizable(A: Matrix, B: Matrix, tol: float = 1.0e-8) -> tuple[bool, tuple[complex, ...]]:
    """Check PBH stabilizability for eigenvalues with nonnegative real part."""

    eigenvalues = np.linalg.eigvals(A)
    n = A.shape[0]
    violating: list[complex] = []
    for eigenvalue in eigenvalues:
        if np.real(eigenvalue) < -tol:
            continue
        test_matrix = np.hstack((eigenvalue * np.eye(n, dtype=complex) - A.astype(complex), B.astype(complex)))
        if np.linalg.matrix_rank(test_matrix, tol) < n:
            violating.append(complex(eigenvalue))
    return len(violating) == 0, tuple(violating)


def evaluate_stabilizability(A_x: Matrix, B_x: Matrix, A: Matrix, B: Matrix, tol: float = 1.0e-8) -> StabilizabilityReport:
    """Evaluate the natural Problem 2 rank and PBH conditions."""

    rank = int(np.linalg.matrix_rank(B_x, tol))
    stabilizable, violating = pbh_stabilizable(A, B, tol=tol)
    return StabilizabilityReport(
        input_rank=rank,
        full_translational_authority=rank == 3,
        stabilizable=stabilizable,
        violating_eigenvalues=violating,
    )

