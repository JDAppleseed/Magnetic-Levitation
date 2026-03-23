"""Analytic and finite-difference linearization tools."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


Vector = NDArray[np.float64]
Matrix = NDArray[np.float64]


@dataclass(frozen=True)
class LinearizationResult:
    """Continuous-time linearization about an equilibrium."""

    A_x: Matrix
    B_x: Matrix
    A: Matrix
    B: Matrix


def state_space_pair(A_x: Matrix, B_x: Matrix, mass: float) -> tuple[Matrix, Matrix]:
    """Build the first-order state-space pair from Problem 2."""

    zero = np.zeros((3, 3), dtype=float)
    identity = np.eye(3, dtype=float)
    A = np.block([[zero, identity], [A_x / mass, zero]])
    B = np.vstack([np.zeros((3, B_x.shape[1]), dtype=float), B_x / mass])
    return A, B


def analytic_linearization(force_model: object, x_eq: Vector, u_eq: Vector, mass: float) -> LinearizationResult:
    """Use model-provided Jacobians."""

    A_x = np.asarray(force_model.state_jacobian(x_eq, u_eq), dtype=float)
    B_x = np.asarray(force_model.input_jacobian(x_eq, u_eq), dtype=float)
    A, B = state_space_pair(A_x, B_x, mass)
    return LinearizationResult(A_x=A_x, B_x=B_x, A=A, B=B)


def finite_difference_linearization(
    force_model: object,
    x_eq: Vector,
    u_eq: Vector,
    mass: float,
    eps_x: float = 1.0e-6,
    eps_u: float = 1.0e-6,
) -> LinearizationResult:
    """Finite-difference linearization of the force law."""

    x_eq = np.asarray(x_eq, dtype=float)
    u_eq = np.asarray(u_eq, dtype=float)
    A_x = np.zeros((3, 3), dtype=float)
    B_x = np.zeros((3, u_eq.size), dtype=float)
    for axis in range(3):
        delta = np.zeros(3, dtype=float)
        delta[axis] = eps_x
        A_x[:, axis] = (force_model.force(x_eq + delta, u_eq) - force_model.force(x_eq - delta, u_eq)) / (
            2.0 * eps_x
        )
    for index in range(u_eq.size):
        delta = np.zeros_like(u_eq)
        delta[index] = eps_u
        B_x[:, index] = (force_model.force(x_eq, u_eq + delta) - force_model.force(x_eq, u_eq - delta)) / (
            2.0 * eps_u
        )
    A, B = state_space_pair(A_x, B_x, mass)
    return LinearizationResult(A_x=A_x, B_x=B_x, A=A, B=B)


def operator_norm_difference(left: Matrix, right: Matrix) -> float:
    """Return the spectral norm of a matrix difference."""

    return float(np.linalg.norm(left - right, ord=2))

