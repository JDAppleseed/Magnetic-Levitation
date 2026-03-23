"""Factory helpers for building the configured simulator stack."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from analysis.config import load_yaml, repo_path
from control.outer_loop import DiagonalGains
from physics.affine_force_model import AffineFaceMagneticForceModel
from physics.dipole_force import FixedDipoleForceModel
from physics.induced_dipole_force import InducedDipoleForceModel
from physics.magnetic_field_model import FaceDipoleFieldModel, SystemParameters, load_system_parameters
from sim.rk4_backend import RK4Backend
from sim.state import RigidBodyState


@dataclass(frozen=True)
class ControllerConfig:
    """Controller and allocator configuration."""

    gains: DiagonalGains
    integral_limit: float
    allocator_mode: str
    saturate_warning_fraction: float


@dataclass(frozen=True)
class SimDefaults:
    """Default script-level simulation settings."""

    duration: float
    dt: float
    start: np.ndarray
    target: np.ndarray
    trajectory_duration: float
    hover_point: np.ndarray
    free_fall_duration: float
    regulation_duration: float
    tracking_duration: float


def load_controller_config(config_path: str | None = None) -> ControllerConfig:
    """Load controller gains and allocator options from YAML."""

    path = repo_path("configs", "controller.yaml") if config_path is None else repo_path(config_path)
    raw = load_yaml(path)
    gains = DiagonalGains(
        kp=np.asarray(raw["gains"]["kp"], dtype=float),
        kd=np.asarray(raw["gains"]["kd"], dtype=float),
        ki=np.asarray(raw["gains"]["ki"], dtype=float),
    )
    return ControllerConfig(
        gains=gains,
        integral_limit=float(raw["integral_limit"]),
        allocator_mode=str(raw["allocator"]["mode"]),
        saturate_warning_fraction=float(raw["allocator"]["saturate_warning_fraction"]),
    )


def load_sim_defaults(config_path: str | None = None) -> SimDefaults:
    """Load script-level simulation defaults."""

    path = repo_path("configs", "sim.yaml") if config_path is None else repo_path(config_path)
    raw = load_yaml(path)
    return SimDefaults(
        duration=float(raw["defaults"]["duration"]),
        dt=float(raw["defaults"]["dt"]),
        start=np.asarray(raw["defaults"]["start"], dtype=float),
        target=np.asarray(raw["defaults"]["target"], dtype=float),
        trajectory_duration=float(raw["defaults"]["trajectory_duration"]),
        hover_point=np.asarray(raw["defaults"]["hover_point"], dtype=float),
        free_fall_duration=float(raw["validation"]["free_fall_duration"]),
        regulation_duration=float(raw["validation"]["regulation_duration"]),
        tracking_duration=float(raw["validation"]["tracking_duration"]),
    )


def build_affine_model(system: SystemParameters | None = None) -> AffineFaceMagneticForceModel:
    """Build the configured affine surrogate model."""

    params = load_system_parameters() if system is None else system
    return AffineFaceMagneticForceModel(params.cube, params.actuators)


def build_physical_models(system: SystemParameters | None = None) -> tuple[FixedDipoleForceModel, InducedDipoleForceModel]:
    """Build both physical magnetic force models."""

    params = load_system_parameters() if system is None else system
    field_model = FaceDipoleFieldModel(params.cube, params.actuators)
    return (
        FixedDipoleForceModel(field_model, params.coupling.fixed_dipole_moment),
        InducedDipoleForceModel(field_model, params.coupling.induced_alpha),
    )


def build_backend(system: SystemParameters | None = None, enable_contact: bool = True) -> RK4Backend:
    """Build the configured RK4 backend."""

    params = load_system_parameters() if system is None else system
    return RK4Backend(
        side_length=params.cube.side_length,
        ball_radius=params.ball.radius,
        mass=params.ball.mass,
        gravity=params.cube.gravity,
        damping=params.backend.damping,
        contact_mode=params.backend.contact_mode,
        restitution=params.backend.restitution,
        enable_contact=enable_contact,
    )


def default_initial_state(system: SystemParameters | None = None, start: np.ndarray | None = None) -> RigidBodyState:
    """Build a default stationary initial state."""

    params = load_system_parameters() if system is None else system
    if start is None:
        defaults = load_sim_defaults()
        start = defaults.start
    return RigidBodyState(np.asarray(start, dtype=float), np.zeros(3, dtype=float), 0.0)

