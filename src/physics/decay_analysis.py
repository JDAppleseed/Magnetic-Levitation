"""Problem 5 decay-limit analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DecayAnalysisResult:
    """Critical-size and center-authority summary."""

    side_length: float
    center_force_bound: float
    required_force: float
    critical_side_length: float
    center_hover_feasible: bool


def critical_domain_size(mass: float, gravity: float, u_max: float, decay_constant: float, alpha: float) -> float:
    """Return L* from Problem 5."""

    return 2.0 * ((6.0 * u_max * decay_constant) / (mass * gravity)) ** (1.0 / alpha)


def center_force_bound(side_length: float, u_max: float, decay_constant: float, alpha: float) -> float:
    """Upper bound on the achievable force magnitude at the cube center."""

    return 6.0 * u_max * decay_constant / (0.5 * side_length) ** alpha


def analyze_decay_limit(
    side_length: float,
    mass: float,
    gravity: float,
    u_max: float,
    decay_constant: float,
    alpha: float,
) -> DecayAnalysisResult:
    required = mass * gravity
    bound = center_force_bound(side_length, u_max, decay_constant, alpha)
    critical = critical_domain_size(mass, gravity, u_max, decay_constant, alpha)
    return DecayAnalysisResult(
        side_length=side_length,
        center_force_bound=bound,
        required_force=required,
        critical_side_length=critical,
        center_hover_feasible=bound >= required,
    )

