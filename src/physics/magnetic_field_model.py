"""Shared geometry, actuator, and magnetic-field helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from analysis.config import load_yaml, repo_path


Vector = NDArray[np.float64]
Matrix = NDArray[np.float64]

E_X = np.array([1.0, 0.0, 0.0], dtype=float)
E_Y = np.array([0.0, 1.0, 0.0], dtype=float)
E_Z = np.array([0.0, 0.0, 1.0], dtype=float)
MU0_OVER_4PI = 1.0e-7


@dataclass(frozen=True)
class CubeGeometry:
    """Cube geometry and gravity parameters."""

    side_length: float
    gravity: float

    def admissible_bounds(self, ball_radius: float) -> tuple[Vector, Vector]:
        lower = np.full(3, ball_radius, dtype=float)
        upper = np.full(3, self.side_length - ball_radius, dtype=float)
        return lower, upper

    def contains(self, x: Vector, ball_radius: float, atol: float = 1.0e-12) -> bool:
        """Return whether ``x`` lies inside the center-of-mass admissible set."""

        lower, upper = self.admissible_bounds(ball_radius)
        point = np.asarray(x, dtype=float)
        return bool(np.all(point >= lower - atol) and np.all(point <= upper + atol))

    def min_distance_to_boundary(self, x: Vector, ball_radius: float) -> float:
        """Return the minimum distance from ``x`` to the admissible boundary."""

        lower, upper = self.admissible_bounds(ball_radius)
        point = np.asarray(x, dtype=float)
        return float(np.min(np.minimum(point - lower, upper - point)))


@dataclass(frozen=True)
class BallProperties:
    """Ball inertial parameters."""

    radius: float
    mass: float
    inertia: float


@dataclass(frozen=True)
class FaceActuatorSpec:
    """Face-centered actuator with inward-pointing normal."""

    name: str
    location: Vector
    normal: Vector
    u_min: float
    u_max: float
    affine_gain: float
    decay_power: float
    decay_offset: float
    dipole_strength: float


@dataclass(frozen=True)
class BackendParameters:
    """Integrator and explicit contact parameters."""

    name: str
    dt: float
    dt_by_mode: dict[str, float]
    damping: float
    contact_mode: str
    restitution: float


@dataclass(frozen=True)
class MagneticCoupling:
    """Robot magnetic coupling parameters."""

    fixed_dipole_moment: Vector
    induced_alpha: float
    physical_preset: str


@dataclass(frozen=True)
class PhysicalPreset:
    """Named magnetic parameter preset for physical field models."""

    name: str
    dipole_strength: float
    fixed_dipole_moment: Vector
    induced_alpha: float


@dataclass(frozen=True)
class InducedModeParameters:
    """Safety and numerical settings for induced-dipole mode."""

    finite_difference_eps: float
    max_force_norm: float
    max_gradient_norm: float
    actuator_slew_rate_limit: float
    min_boundary_margin: float
    max_velocity_norm: float
    allow_exploratory_default: bool


@dataclass(frozen=True)
class PhysicalStatusParameters:
    """Thresholds for classifying physical-mode operating states."""

    exact_residual_norm: float
    marginal_residual_norm: float
    exact_margin: float
    marginal_margin: float
    exact_saturation_margin: float
    marginal_saturation_margin: float
    exact_force_to_weight_ratio: float
    marginal_force_to_weight_ratio: float
    exact_condition_number: float
    marginal_condition_number: float
    path_sample_count: int


@dataclass(frozen=True)
class SystemParameters:
    """Top-level system parameters loaded from configuration."""

    cube: CubeGeometry
    ball: BallProperties
    backend: BackendParameters
    actuators: tuple[FaceActuatorSpec, ...]
    coupling: MagneticCoupling
    physical_presets: dict[str, PhysicalPreset]
    induced_mode: InducedModeParameters
    physical_status: PhysicalStatusParameters


def _vector(values: Iterable[float]) -> Vector:
    return np.asarray(list(values), dtype=float)


def build_face_actuators(
    side_length: float,
    u_min: float,
    u_max: float,
    affine_gain: float,
    decay_power: float,
    decay_offset: float,
    dipole_strength: float,
) -> tuple[FaceActuatorSpec, ...]:
    """Create six inward-pointing face actuators."""

    half = 0.5 * side_length
    specs = (
        ("x-", np.array([0.0, half, half]), E_X),
        ("x+", np.array([side_length, half, half]), -E_X),
        ("y-", np.array([half, 0.0, half]), E_Y),
        ("y+", np.array([half, side_length, half]), -E_Y),
        ("z-", np.array([half, half, 0.0]), E_Z),
        ("z+", np.array([half, half, side_length]), -E_Z),
    )
    return tuple(
        FaceActuatorSpec(
            name=name,
            location=location.astype(float),
            normal=normal.astype(float),
            u_min=float(u_min),
            u_max=float(u_max),
            affine_gain=float(affine_gain),
            decay_power=float(decay_power),
            decay_offset=float(decay_offset),
            dipole_strength=float(dipole_strength),
        )
        for name, location, normal in specs
    )


def load_system_parameters(config_path: str | None = None, physical_preset: str | None = None) -> SystemParameters:
    """Load default repository configuration."""

    path = repo_path("configs", "models.yaml") if config_path is None else repo_path(config_path)
    raw = load_yaml(path)
    cube = CubeGeometry(
        side_length=float(raw["cube"]["side_length"]),
        gravity=float(raw["cube"]["gravity"]),
    )
    ball = BallProperties(
        radius=float(raw["ball"]["radius"]),
        mass=float(raw["ball"]["mass"]),
        inertia=float(raw["ball"]["inertia"]),
    )
    backend = BackendParameters(
        name=str(raw["backend"]["name"]),
        dt=float(raw["backend"]["dt"]),
        dt_by_mode={str(key): float(value) for key, value in raw["backend"].get("dt_by_mode", {}).items()},
        damping=float(raw["backend"]["damping"]),
        contact_mode=str(raw["backend"]["contact_mode"]),
        restitution=float(raw["backend"]["restitution"]),
    )
    preset_block = raw.get("physical_presets", {})
    default_preset_name = str(preset_block.get("default", raw.get("robot_coupling", {}).get("physical_preset", "default")))
    preset_name = default_preset_name if physical_preset is None else str(physical_preset)
    preset_entries = {
        str(name): PhysicalPreset(
            name=str(name),
            dipole_strength=float(values["dipole_strength"]),
            fixed_dipole_moment=_vector(values["fixed_dipole_moment"]),
            induced_alpha=float(values["induced_alpha"]),
        )
        for name, values in preset_block.items()
        if name != "default"
    }
    if not preset_entries:
        preset_entries["default"] = PhysicalPreset(
            name="default",
            dipole_strength=float(raw["actuators"]["dipole_strength"]),
            fixed_dipole_moment=_vector(raw["robot_coupling"]["fixed_dipole_moment"]),
            induced_alpha=float(raw["robot_coupling"]["induced_alpha"]),
        )
        default_preset_name = "default"
        if physical_preset is None:
            preset_name = "default"
    if preset_name not in preset_entries:
        raise KeyError(f"Unknown physical preset '{preset_name}'. Available presets: {sorted(preset_entries)}")
    active_preset = preset_entries[preset_name]
    actuators = build_face_actuators(
        side_length=cube.side_length,
        u_min=float(raw["actuators"]["u_min"]),
        u_max=float(raw["actuators"]["u_max"]),
        affine_gain=float(raw["actuators"]["affine_gain"]),
        decay_power=float(raw["actuators"]["decay_power"]),
        decay_offset=float(raw["actuators"]["decay_offset"]),
        dipole_strength=active_preset.dipole_strength,
    )
    coupling = MagneticCoupling(
        fixed_dipole_moment=active_preset.fixed_dipole_moment,
        induced_alpha=active_preset.induced_alpha,
        physical_preset=active_preset.name,
    )
    induced_mode = raw.get("induced_mode", {})
    physical_status = raw.get("physical_status", {})
    return SystemParameters(
        cube=cube,
        ball=ball,
        backend=backend,
        actuators=actuators,
        coupling=coupling,
        physical_presets=preset_entries,
        induced_mode=InducedModeParameters(
            finite_difference_eps=float(induced_mode.get("finite_difference_eps", 1.0e-5)),
            max_force_norm=float(induced_mode.get("max_force_norm", 8.0)),
            max_gradient_norm=float(induced_mode.get("max_gradient_norm", 12.0)),
            actuator_slew_rate_limit=float(induced_mode.get("actuator_slew_rate_limit", 30.0)),
            min_boundary_margin=float(induced_mode.get("min_boundary_margin", 0.002)),
            max_velocity_norm=float(induced_mode.get("max_velocity_norm", 3.0)),
            allow_exploratory_default=bool(induced_mode.get("allow_exploratory_default", False)),
        ),
        physical_status=PhysicalStatusParameters(
            exact_residual_norm=float(physical_status.get("exact_residual_norm", 1.0e-6)),
            marginal_residual_norm=float(physical_status.get("marginal_residual_norm", 1.0e-3)),
            exact_margin=float(physical_status.get("exact_margin", 0.05)),
            marginal_margin=float(physical_status.get("marginal_margin", -0.05)),
            exact_saturation_margin=float(physical_status.get("exact_saturation_margin", 0.10)),
            marginal_saturation_margin=float(physical_status.get("marginal_saturation_margin", 0.01)),
            exact_force_to_weight_ratio=float(physical_status.get("exact_force_to_weight_ratio", 1.05)),
            marginal_force_to_weight_ratio=float(physical_status.get("marginal_force_to_weight_ratio", 0.95)),
            exact_condition_number=float(physical_status.get("exact_condition_number", 1.0e3)),
            marginal_condition_number=float(physical_status.get("marginal_condition_number", 1.0e5)),
            path_sample_count=int(physical_status.get("path_sample_count", 101)),
        ),
    )


class FaceDipoleFieldModel:
    """Static magnetic field from six face-centered point dipoles."""

    def __init__(self, cube: CubeGeometry, actuators: tuple[FaceActuatorSpec, ...]) -> None:
        self.cube = cube
        self.actuators = actuators

    def actuator_bounds(self) -> tuple[Vector, Vector]:
        lower = np.array([spec.u_min for spec in self.actuators], dtype=float)
        upper = np.array([spec.u_max for spec in self.actuators], dtype=float)
        return lower, upper

    def field_basis(self, x: Vector) -> Matrix:
        """Return unit-input field columns at position x."""

        return np.column_stack([self._single_face_field(spec, x) for spec in self.actuators])

    def field_basis_jacobians(self, x: Vector) -> tuple[Matrix, ...]:
        """Return ``dB_i/dx`` for each unit-input face actuator."""

        return tuple(self._single_face_field_jacobian(spec, x) for spec in self.actuators)

    def field_basis_and_jacobians(self, x: Vector) -> tuple[Matrix, tuple[Matrix, ...]]:
        """Return unit-input field basis and spatial Jacobians."""

        basis_columns: list[Vector] = []
        jacobians: list[Matrix] = []
        for spec in self.actuators:
            field, jacobian = self._single_face_field_and_jacobian(spec, x)
            basis_columns.append(field)
            jacobians.append(jacobian)
        return np.column_stack(basis_columns), tuple(jacobians)

    def field(self, x: Vector, u: Vector) -> Vector:
        return self.field_basis(x) @ np.asarray(u, dtype=float)

    def field_jacobian(self, x: Vector, u: Vector) -> Matrix:
        """Return ``dB/dx`` for the superposed field at ``(x, u)``."""

        _, basis_jacobians = self.field_basis_and_jacobians(x)
        return np.tensordot(np.asarray(u, dtype=float), np.asarray(basis_jacobians, dtype=float), axes=(0, 0))

    @staticmethod
    def _single_face_field(spec: FaceActuatorSpec, x: Vector) -> Vector:
        field, _ = FaceDipoleFieldModel._single_face_field_and_jacobian(spec, x)
        return field

    @staticmethod
    def _single_face_field_jacobian(spec: FaceActuatorSpec, x: Vector) -> Matrix:
        _, jacobian = FaceDipoleFieldModel._single_face_field_and_jacobian(spec, x)
        return jacobian

    @staticmethod
    def _single_face_field_and_jacobian(spec: FaceActuatorSpec, x: Vector) -> tuple[Vector, Matrix]:
        displacement = np.asarray(x, dtype=float) - spec.location
        radius = float(np.linalg.norm(displacement))
        if radius == 0.0:
            raise ValueError("Point dipole field undefined at source location.")
        moment = spec.dipole_strength * spec.normal
        s = float(np.dot(moment, displacement))
        radius2 = radius * radius
        radius5 = radius2 * radius2 * radius
        radius7 = radius5 * radius2
        outer_rr = np.outer(displacement, displacement)
        basis_matrix = 3.0 * s * np.eye(3, dtype=float) + 3.0 * (
            np.outer(displacement, moment) + np.outer(moment, displacement)
        )
        jacobian = MU0_OVER_4PI * (basis_matrix / radius5 - 15.0 * s * outer_rr / radius7)
        field = MU0_OVER_4PI * (3.0 * displacement * s / radius5 - moment / (radius**3))
        return field, jacobian
