"""Target acceptance, path feasibility, and operating-state helpers for physical modes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from control.allocator import AllocationResult
from control.trajectory_generator import QuinticTrajectory, TrajectorySample
from physics.force_feasibility import evaluate_force_request, evaluate_hover_feasibility, hover_force
from physics.induced_dipole_force import InducedDipoleForceModel
from physics.magnetic_field_model import InducedModeParameters, PhysicalStatusParameters


Vector = NDArray[np.float64]
Matrix = NDArray[np.float64]

ENDPOINT_EXACT_FEASIBLE = "endpoint_exact_feasible"
ENDPOINT_MARGINAL = "endpoint_marginal"
ENDPOINT_BLOCKED = "endpoint_blocked"
PATH_FEASIBLE = "path_feasible"
PATH_MARGINAL = "path_marginal"
PATH_BLOCKED = "path_blocked"
PATH_NOT_REQUIRED = "path_not_required"
START_ALLOWED = "start_allowed"
START_ALLOWED_CAUTION = "start_allowed_caution"
START_REQUIRES_EXPLORATORY_OVERRIDE = "start_requires_exploratory_override"
START_BLOCKED = "start_blocked"
START_UNSAFE_ABORT = "start_unsafe_abort"
EXECUTION_HOLD = "hold"
EXECUTION_DIRECT_TRAJECTORY = "direct_trajectory"
EXECUTION_FALLBACK_REGULATION = "fallback_regulation"
EXECUTION_GUARDED_FALLBACK = "guarded_fallback"
EXECUTION_BLOCKED = "blocked"
EXECUTION_UNSAFE_ABORT = "unsafe_abort"

# Compatibility aliases for earlier UI/test code.
EXACT_FEASIBLE = START_ALLOWED
MARGINAL_FEASIBLE = START_ALLOWED_CAUTION
GUARDED_EXPLORATORY = START_REQUIRES_EXPLORATORY_OVERRIDE
INFEASIBLE_BLOCKED = START_BLOCKED
UNSAFE_ABORT = START_UNSAFE_ABORT


@dataclass(frozen=True)
class HoverCheckResult:
    """Endpoint hover-feasibility summary used by the UI and gating logic."""

    position: Vector
    feasible: bool
    accepted_by_default: bool
    required_force: Vector
    achieved_force: Vector
    residual_norm: float
    margin_estimate: float
    saturation_margin: float
    force_to_weight_ratio: float
    condition_number: float | None
    safety_limited: bool
    reason: str
    allocation: AllocationResult


@dataclass(frozen=True)
class PathSampleCheck:
    """One sampled path-feasibility diagnostic."""

    index: int
    time: float
    position: Vector
    required_force: Vector
    residual_norm: float
    margin_estimate: float
    feasible: bool
    safety_limited: bool
    reason: str


@dataclass(frozen=True)
class PathCheckResult:
    """Pointwise force/path feasibility over a planned transfer."""

    feasible: bool
    accepted_by_default: bool
    first_failure_index: int | None
    first_failure_time: float | None
    worst_residual_norm: float
    minimum_margin: float
    minimum_saturation_margin: float
    sample_count: int
    safety_limited: bool
    reason: str
    samples: tuple[PathSampleCheck, ...]


@dataclass(frozen=True)
class PhysicalOperatingAssessment:
    """Classified operating state for a physical target request."""

    state: str
    target_status: str
    path_status: str
    color: str
    target_color: str
    start_allowed: bool
    start_requires_override: bool
    physical_feasibility: str
    execution_mode: str
    execution_mode_text: str
    endpoint_grade: str
    path_grade: str | None
    exact_hover_certified: bool
    exact_path_certified: bool | None
    endpoint_status_text: str
    path_status_text: str
    start_status_text: str
    margin_text: str
    message: str
    force_to_weight_ratio: float
    exploratory_enabled: bool
    guard_active: bool
    unsafe_reason: str | None


@dataclass(frozen=True)
class PhysicalTargetPlan:
    """Validated target request for a physical mode."""

    requested_target: Vector
    source_position: Vector
    accepted: bool
    status: str
    reason: str
    endpoint: HoverCheckResult
    path: PathCheckResult | None
    trajectory: QuinticTrajectory | None
    assessment: PhysicalOperatingAssessment


@dataclass
class PhysicalTargetState:
    """Last-feasible-target memory and pending/active transfer state."""

    hold_target: Vector | None = None
    last_feasible_target: Vector | None = None
    requested_target: Vector | None = None
    pending_plan: PhysicalTargetPlan | None = None
    active_plan: PhysicalTargetPlan | None = None
    endpoint: HoverCheckResult | None = None
    path: PathCheckResult | None = None
    preview_status: str = "idle"
    warning: str = ""
    unsafe_abort_reason: str = ""

    def reset(self) -> None:
        self.hold_target = None
        self.last_feasible_target = None
        self.requested_target = None
        self.pending_plan = None
        self.active_plan = None
        self.endpoint = None
        self.path = None
        self.preview_status = "idle"
        self.warning = ""
        self.unsafe_abort_reason = ""

    def set_hold_target(self, target: Vector, record_last_feasible: bool = True) -> None:
        hold = np.asarray(target, dtype=float).copy()
        self.hold_target = hold
        if record_last_feasible:
            self.last_feasible_target = hold.copy()

    def apply_request(self, plan: PhysicalTargetPlan, simulation_active: bool) -> bool:
        self.requested_target = np.asarray(plan.requested_target, dtype=float).copy()
        self.endpoint = plan.endpoint
        self.path = plan.path
        self.preview_status = plan.assessment.target_status
        self.warning = plan.assessment.message
        if plan.accepted:
            if simulation_active:
                self.active_plan = plan
                self.pending_plan = None
            else:
                self.pending_plan = plan
            return True
        if not simulation_active:
            self.pending_plan = None
        return False

    def activate_pending(self) -> PhysicalTargetPlan | None:
        if self.pending_plan is None:
            return None
        if self.pending_plan.assessment.execution_mode == EXECUTION_HOLD:
            self.set_hold_target(self.pending_plan.requested_target, record_last_feasible=False)
            self.pending_plan = None
            self.active_plan = None
            return None
        self.active_plan = self.pending_plan
        self.pending_plan = None
        return self.active_plan

    def active_goal(self) -> Vector | None:
        if self.active_plan is not None:
            return np.asarray(self.active_plan.requested_target, dtype=float)
        return None if self.hold_target is None else np.asarray(self.hold_target, dtype=float)

    def active_reference(self, time: float) -> Vector | TrajectorySample | None:
        if self.active_plan is not None and self.active_plan.assessment.execution_mode == EXECUTION_DIRECT_TRAJECTORY:
            if self.active_plan.trajectory is not None and time >= self.active_plan.trajectory.end_time:
                self.transfer_completed()
                return None if self.hold_target is None else self.hold_target.copy()
            if self.active_plan.trajectory is not None:
                return self.active_plan.trajectory.evaluate(time)
        if self.active_plan is not None:
            return np.asarray(self.active_plan.requested_target, dtype=float).copy()
        return None if self.hold_target is None else self.hold_target.copy()

    def clear_active_plan(self) -> None:
        self.active_plan = None

    def set_unsafe_abort(self, reason: str) -> None:
        self.preview_status = ENDPOINT_BLOCKED
        self.warning = reason
        self.unsafe_abort_reason = reason

    def clear_unsafe_abort(self) -> None:
        self.unsafe_abort_reason = ""
        if self.preview_status == UNSAFE_ABORT:
            self.preview_status = "idle"

    def transfer_completed(self) -> None:
        if self.active_plan is None:
            return
        record_last = self.active_plan.assessment.target_status in {ENDPOINT_EXACT_FEASIBLE, ENDPOINT_MARGINAL}
        self.set_hold_target(self.active_plan.requested_target, record_last_feasible=record_last)
        self.preview_status = self.active_plan.assessment.target_status
        self.warning = "Transfer completed. Holding validated hover target."
        self.active_plan = None


def allocator_mode_for_force_model(force_model: object) -> str:
    """Return the default allocator mode for a force model."""

    if hasattr(force_model, "force_matrix"):
        return "bounded_ls"
    if isinstance(force_model, InducedDipoleForceModel):
        return "nonlinear"
    return "nonlinear_local"


def corner_sampled_max_force_norm(force_model: object, x: Vector) -> float:
    """Estimate local force authority by evaluating actuator-box corners."""

    lower, upper = force_model.actuator_bounds()
    point = np.asarray(x, dtype=float)
    max_norm = 0.0
    for selector in np.ndindex(*(2 for _ in range(lower.size))):
        u = np.array([upper[idx] if bit else lower[idx] for idx, bit in enumerate(selector)], dtype=float)
        max_norm = max(max_norm, float(np.linalg.norm(force_model.force(point, u))))
    return max_norm


def allocation_margin(requested_force: Vector, allocation: AllocationResult) -> float:
    """Return a signed feasibility margin from an allocation result."""

    saturation_margin = 1.0 - allocation.saturation_fraction
    if allocation.feasible:
        return float(saturation_margin)
    scale = max(float(np.linalg.norm(requested_force)), 1.0e-12)
    return -float(allocation.residual_norm / scale)


def local_condition_number(force_model: object, x: Vector, u: Vector) -> float | None:
    """Return the local input-force condition number when defined."""

    if hasattr(force_model, "force_matrix"):
        matrix = np.asarray(force_model.force_matrix(x), dtype=float)
    elif hasattr(force_model, "input_jacobian"):
        matrix = np.asarray(force_model.input_jacobian(x, u), dtype=float)
    else:
        return None
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if singular_values.size == 0:
        return None
    if singular_values[-1] <= 1.0e-12:
        return float(np.inf)
    return float(singular_values[0] / singular_values[-1])


def _induced_safety_margin(
    force_model: InducedDipoleForceModel,
    x: Vector,
    u: Vector,
    ball_radius: float,
    safety: InducedModeParameters,
) -> tuple[float, list[str]]:
    diagnostics = force_model.diagnostics(x, u, ball_radius)
    margins = [np.inf]
    reasons: list[str] = []
    if diagnostics.min_distance_to_boundary is not None:
        boundary_margin = (
            diagnostics.min_distance_to_boundary - safety.min_boundary_margin
        ) / max(safety.min_boundary_margin, 1.0e-12)
        margins.append(float(boundary_margin))
        if diagnostics.min_distance_to_boundary < safety.min_boundary_margin:
            reasons.append("boundary margin below safety threshold")
    gradient_margin = 1.0 - diagnostics.grad_norm_sq_norm / max(safety.max_gradient_norm, 1.0e-12)
    margins.append(float(gradient_margin))
    if diagnostics.grad_norm_sq_norm > safety.max_gradient_norm:
        reasons.append("gradient norm exceeds safety threshold")
    return float(min(margins)), reasons


def _grade_metrics(
    *,
    residual_norm: float,
    margin_estimate: float,
    saturation_margin: float,
    force_to_weight_ratio: float,
    condition_number: float | None,
    thresholds: PhysicalStatusParameters,
) -> str:
    exact_ok = (
        residual_norm <= thresholds.exact_residual_norm
        and margin_estimate >= thresholds.exact_margin
        and saturation_margin >= thresholds.exact_saturation_margin
        and force_to_weight_ratio >= thresholds.exact_force_to_weight_ratio
    )
    if exact_ok and condition_number is not None:
        exact_ok = bool(np.isfinite(condition_number) and condition_number <= thresholds.exact_condition_number)
    if exact_ok:
        return "exact"
    if (
        residual_norm <= thresholds.marginal_residual_norm
        and margin_estimate >= thresholds.marginal_margin
        and saturation_margin >= thresholds.marginal_saturation_margin
        and force_to_weight_ratio >= thresholds.marginal_force_to_weight_ratio
    ):
        if condition_number is not None and np.isfinite(condition_number) and condition_number > thresholds.marginal_condition_number:
            return "blocked"
        return "marginal"
    return "blocked"


def _path_grade(
    path: PathCheckResult | None,
    thresholds: PhysicalStatusParameters,
    force_to_weight_ratio: float,
) -> str | None:
    if path is None:
        return None
    return _grade_metrics(
        residual_norm=path.worst_residual_norm,
        margin_estimate=path.minimum_margin,
        saturation_margin=path.minimum_saturation_margin,
        force_to_weight_ratio=force_to_weight_ratio,
        condition_number=None,
        thresholds=thresholds,
    )


def _path_failure_time_text(path: PathCheckResult | None, start_time: float) -> str:
    if path is None or path.first_failure_time is None:
        return "-"
    return f"{max(path.first_failure_time - start_time, 0.0):.2f} s"


def _endpoint_status_from_grade(grade: str) -> str:
    return {
        "exact": ENDPOINT_EXACT_FEASIBLE,
        "marginal": ENDPOINT_MARGINAL,
        "blocked": ENDPOINT_BLOCKED,
    }[grade]


def _path_status_from_grade(grade: str | None) -> str:
    if grade is None:
        return PATH_NOT_REQUIRED
    return {
        "exact": PATH_FEASIBLE,
        "marginal": PATH_MARGINAL,
        "blocked": PATH_BLOCKED,
    }[grade]


def _target_color(status: str) -> str:
    return {
        ENDPOINT_EXACT_FEASIBLE: "green",
        ENDPOINT_MARGINAL: "yellow",
        ENDPOINT_BLOCKED: "red",
    }[status]


def _start_color(state: str) -> str:
    return {
        START_ALLOWED: "green",
        START_ALLOWED_CAUTION: "yellow",
        START_REQUIRES_EXPLORATORY_OVERRIDE: "yellow",
        START_BLOCKED: "red",
        START_UNSAFE_ABORT: "red",
    }[state]


def _start_label(state: str) -> str:
    return {
        START_ALLOWED: "start allowed",
        START_ALLOWED_CAUTION: "start allowed with caution",
        START_REQUIRES_EXPLORATORY_OVERRIDE: "exploratory override required",
        START_BLOCKED: "start blocked",
        START_UNSAFE_ABORT: "unsafe abort",
    }[state]


def _execution_mode_text(mode: str) -> str:
    return {
        EXECUTION_HOLD: "Hold target",
        EXECUTION_DIRECT_TRAJECTORY: "Nominal trajectory",
        EXECUTION_FALLBACK_REGULATION: "Fallback direct regulation",
        EXECUTION_GUARDED_FALLBACK: "Guarded fallback regulation",
        EXECUTION_BLOCKED: "Blocked",
        EXECUTION_UNSAFE_ABORT: "Unsafe abort",
    }[mode]


def _component_text(name: str, grade: str, residual_norm: float, margin: float, extra: str = "") -> str:
    prefix = {"exact": "exact certified", "marginal": "near-feasible", "blocked": "blocked"}[grade]
    suffix = f", {extra}" if extra else ""
    return f"{name}: {prefix}; residual={residual_norm:.2e}, margin={margin:.2f}{suffix}"


def classify_physical_plan(
    endpoint: HoverCheckResult,
    path: PathCheckResult | None,
    thresholds: PhysicalStatusParameters,
    *,
    mode_name: str,
    exploratory_enabled: bool,
    exploratory_supported: bool = False,
    runtime_guard_active: bool = False,
    runtime_unsafe_reason: str | None = None,
) -> PhysicalOperatingAssessment:
    """Classify endpoint hover, path feasibility, and start policy separately."""

    endpoint_grade = _grade_metrics(
        residual_norm=endpoint.residual_norm,
        margin_estimate=endpoint.margin_estimate,
        saturation_margin=endpoint.saturation_margin,
        force_to_weight_ratio=endpoint.force_to_weight_ratio,
        condition_number=endpoint.condition_number,
        thresholds=thresholds,
    )
    path_grade = _path_grade(path, thresholds, endpoint.force_to_weight_ratio)
    endpoint_status = _endpoint_status_from_grade(endpoint_grade)
    path_status = _path_status_from_grade(path_grade)
    exact_hover_certified = endpoint_grade == "exact"
    exact_path_certified = None if path is None else path_grade == "exact"
    safety_limited = endpoint.safety_limited or (path.safety_limited if path is not None else False)
    path_blocks_transfer = path_status == PATH_BLOCKED
    moving_target = path is not None

    if runtime_unsafe_reason:
        start_state = START_UNSAFE_ABORT
        execution_mode = EXECUTION_UNSAFE_ABORT
    elif endpoint_status == ENDPOINT_BLOCKED:
        start_state = START_BLOCKED
        execution_mode = EXECUTION_BLOCKED
    elif not moving_target:
        start_state = START_ALLOWED if endpoint_status == ENDPOINT_EXACT_FEASIBLE and not safety_limited else START_ALLOWED_CAUTION
        execution_mode = EXECUTION_HOLD
    elif runtime_guard_active:
        start_state = START_ALLOWED_CAUTION
        execution_mode = EXECUTION_GUARDED_FALLBACK if exploratory_supported else EXECUTION_FALLBACK_REGULATION
    elif path_blocks_transfer:
        start_state = START_ALLOWED_CAUTION
        execution_mode = EXECUTION_GUARDED_FALLBACK if exploratory_supported else EXECUTION_FALLBACK_REGULATION
    elif safety_limited:
        start_state = START_ALLOWED_CAUTION
        execution_mode = EXECUTION_GUARDED_FALLBACK if exploratory_supported else EXECUTION_DIRECT_TRAJECTORY
    elif endpoint_status == ENDPOINT_MARGINAL or path_status == PATH_MARGINAL:
        start_state = START_ALLOWED_CAUTION
        execution_mode = EXECUTION_DIRECT_TRAJECTORY
    else:
        start_state = START_ALLOWED
        execution_mode = EXECUTION_DIRECT_TRAJECTORY

    endpoint_text = _component_text("Endpoint hover", endpoint_grade, endpoint.residual_norm, endpoint.margin_estimate)
    if path is None:
        path_text = "Path: no transfer required."
    else:
        failure_text = _path_failure_time_text(path, path.samples[0].time if path.samples else 0.0)
        extra = f"first fail={failure_text}" if path.first_failure_time is not None else ""
        path_text = _component_text("Path", path_grade or "blocked", path.worst_residual_norm, path.minimum_margin, extra)

    if start_state == START_ALLOWED:
        start_text = "Start: allowed."
    elif start_state == START_ALLOWED_CAUTION:
        start_text = "Start: allowed with caution."
    elif start_state == START_REQUIRES_EXPLORATORY_OVERRIDE:
        start_text = "Start: exploratory override required."
    elif start_state == START_BLOCKED:
        start_text = "Start: blocked."
    else:
        start_text = "Start: unsafe abort; reset or change the request."

    execution_text = f"Execution: {_execution_mode_text(execution_mode)}."

    if start_state == START_UNSAFE_ABORT:
        message = f"{mode_name}: unsafe abort. {runtime_unsafe_reason}"
    elif endpoint_status == ENDPOINT_BLOCKED:
        message = (
            f"{mode_name}: requested target blocked at endpoint. "
            f"Residual = {endpoint.residual_norm:.2e}, margin = {endpoint.margin_estimate:.2f}. "
            "Holding previous feasible target."
        )
    elif path_status == PATH_BLOCKED:
        guard_note = " Guarded runtime safety remains active." if exploratory_supported else ""
        message = (
            f"{mode_name}: endpoint hover is acceptable, but the nominal path is blocked at t = {_path_failure_time_text(path, 0.0)}. "
            f"Starting with fallback regulation instead of the nominal path.{guard_note}"
        )
    elif start_state == START_ALLOWED_CAUTION:
        message = (
            f"{mode_name}: endpoint/path are usable but cautionary. "
            f"Endpoint residual = {endpoint.residual_norm:.2e}, endpoint margin = {endpoint.margin_estimate:.2f}."
        )
        if path is not None:
            message += f" Path min margin = {path.minimum_margin:.2f}, worst residual = {path.worst_residual_norm:.2e}."
    else:
        message = f"{mode_name}: endpoint hover and transfer are acceptable. Start allowed."

    path_margin_text = "n/a" if path is None else f"{path.minimum_margin:.2f}"
    margin_text = (
        f"endpoint={endpoint.margin_estimate:.2f}, "
        f"path={path_margin_text}, "
        f"sat={endpoint.saturation_margin:.2f}, "
        f"F/w={endpoint.force_to_weight_ratio:.2f}"
    )
    if endpoint.condition_number is not None and np.isfinite(endpoint.condition_number):
        margin_text += f", cond={endpoint.condition_number:.2e}"
    if exploratory_enabled:
        margin_text += ", exploratory=on"
    else:
        margin_text += ", exploratory=off"
    if safety_limited:
        margin_text += ", safety-limited"

    return PhysicalOperatingAssessment(
        state=start_state,
        target_status=endpoint_status,
        path_status=path_status,
        color=_start_color(start_state),
        target_color=_target_color(endpoint_status),
        start_allowed=start_state in {START_ALLOWED, START_ALLOWED_CAUTION},
        start_requires_override=False,
        physical_feasibility=_start_label(start_state),
        execution_mode=execution_mode,
        execution_mode_text=execution_text,
        endpoint_grade=endpoint_grade,
        path_grade=path_grade,
        exact_hover_certified=exact_hover_certified,
        exact_path_certified=exact_path_certified,
        endpoint_status_text=endpoint_text,
        path_status_text=path_text,
        start_status_text=start_text,
        margin_text=margin_text,
        message=message,
        force_to_weight_ratio=endpoint.force_to_weight_ratio,
        exploratory_enabled=exploratory_enabled,
        guard_active=runtime_guard_active,
        unsafe_reason=runtime_unsafe_reason,
    )


def check_hover_feasibility(
    force_model: object,
    target: Vector,
    mass: float,
    gravity: float,
    *,
    mode: str | None = None,
    ball_radius: float | None = None,
    induced_safety: InducedModeParameters | None = None,
) -> HoverCheckResult:
    """Evaluate pointwise hover feasibility and a signed margin estimate."""

    allocator_mode = allocator_mode_for_force_model(force_model) if mode is None else mode
    hover = evaluate_hover_feasibility(force_model, target, mass, gravity, mode=allocator_mode)
    saturation_margin = 1.0 - hover.allocation.saturation_fraction
    margin = allocation_margin(hover.required_force, hover.allocation)
    reasons: list[str] = []
    accepted_by_default = bool(hover.allocation.feasible)
    safety_limited = False
    if not hover.allocation.feasible:
        reasons.append("exact hover equilibrium not certified")
    if (
        isinstance(force_model, InducedDipoleForceModel)
        and ball_radius is not None
        and induced_safety is not None
        and hover.allocation.feasible
    ):
        safety_margin, safety_reasons = _induced_safety_margin(
            force_model,
            target,
            hover.allocation.u,
            ball_radius,
            induced_safety,
        )
        margin = min(margin, safety_margin)
        if safety_reasons:
            accepted_by_default = False
            safety_limited = True
            reasons.extend(safety_reasons)
    if hover.allocation.feasible and hover.allocation.saturated:
        reasons.append("near actuator saturation")
    condition_number = local_condition_number(force_model, target, hover.allocation.u)
    if condition_number is not None and np.isfinite(condition_number) and condition_number > 1.0e4:
        reasons.append("poor local conditioning")
    force_ratio = corner_sampled_max_force_norm(force_model, target) / max(mass * gravity, 1.0e-12)
    if not reasons:
        reasons.append("hover feasible")
    return HoverCheckResult(
        position=np.asarray(target, dtype=float),
        feasible=bool(hover.allocation.feasible),
        accepted_by_default=bool(accepted_by_default),
        required_force=np.asarray(hover.required_force, dtype=float),
        achieved_force=np.asarray(hover.allocation.achieved_force, dtype=float),
        residual_norm=float(hover.allocation.residual_norm),
        margin_estimate=float(margin),
        saturation_margin=float(saturation_margin),
        force_to_weight_ratio=float(force_ratio),
        condition_number=condition_number,
        safety_limited=safety_limited,
        reason="; ".join(reasons),
        allocation=hover.allocation,
    )


def check_path_feasibility(
    force_model: object,
    trajectory: QuinticTrajectory,
    mass: float,
    gravity: float,
    *,
    mode: str | None = None,
    ball_radius: float | None = None,
    induced_safety: InducedModeParameters | None = None,
    sample_count: int = 101,
) -> PathCheckResult:
    """Sample a planned path and verify nominal force realizability along it."""

    allocator_mode = allocator_mode_for_force_model(force_model) if mode is None else mode
    samples: list[PathSampleCheck] = []
    first_failure_index: int | None = None
    first_failure_time: float | None = None
    reasons: list[str] = []
    worst_residual_norm = 0.0
    minimum_margin = float(np.inf)
    minimum_saturation_margin = float(np.inf)
    accepted_by_default = True
    safety_limited = False
    for index, time in enumerate(np.linspace(trajectory.start_time, trajectory.end_time, sample_count)):
        sample = trajectory.evaluate(float(time))
        required_force = hover_force(mass, gravity) + mass * sample.acceleration
        allocation = evaluate_force_request(
            force_model,
            sample.position,
            required_force,
            mode=allocator_mode,
        )
        margin = allocation_margin(required_force, allocation)
        feasible = bool(allocation.feasible)
        sample_safety_limited = False
        sample_reasons: list[str] = []
        if not feasible:
            sample_reasons.append("exact force allocation not certified")
        if (
            isinstance(force_model, InducedDipoleForceModel)
            and ball_radius is not None
            and induced_safety is not None
            and allocation.feasible
        ):
            safety_margin, safety_reasons = _induced_safety_margin(
                force_model,
                sample.position,
                allocation.u,
                ball_radius,
                induced_safety,
            )
            margin = min(margin, safety_margin)
            if safety_reasons:
                feasible = False
                accepted_by_default = False
                safety_limited = True
                sample_safety_limited = True
                sample_reasons.extend(safety_reasons)
        if not feasible and first_failure_index is None:
            first_failure_index = index
            first_failure_time = float(time)
            accepted_by_default = False
        worst_residual_norm = max(worst_residual_norm, float(allocation.residual_norm))
        minimum_margin = min(minimum_margin, float(margin))
        minimum_saturation_margin = min(minimum_saturation_margin, float(1.0 - allocation.saturation_fraction))
        if not sample_reasons:
            sample_reasons.append("sample feasible")
        samples.append(
            PathSampleCheck(
                index=index,
                time=float(time),
                position=np.asarray(sample.position, dtype=float),
                required_force=np.asarray(required_force, dtype=float),
                residual_norm=float(allocation.residual_norm),
                margin_estimate=float(margin),
                feasible=bool(feasible),
                safety_limited=sample_safety_limited,
                reason="; ".join(sample_reasons),
            )
        )
    feasible = all(item.feasible for item in samples)
    if feasible:
        reasons.append("path feasible")
    elif first_failure_time is not None:
        reasons.append(f"path infeasible at t = {first_failure_time - trajectory.start_time:.2f} s")
    else:
        reasons.append("path infeasible")
    return PathCheckResult(
        feasible=bool(feasible),
        accepted_by_default=bool(accepted_by_default and feasible),
        first_failure_index=first_failure_index,
        first_failure_time=first_failure_time,
        worst_residual_norm=float(worst_residual_norm),
        minimum_margin=float(minimum_margin),
        minimum_saturation_margin=float(minimum_saturation_margin),
        sample_count=int(sample_count),
        safety_limited=safety_limited,
        reason="; ".join(reasons),
        samples=tuple(samples),
    )


def plan_physical_target_request(
    force_model: object,
    source_position: Vector,
    requested_target: Vector,
    mass: float,
    gravity: float,
    *,
    trajectory_duration: float,
    mode: str | None = None,
    ball_radius: float | None = None,
    induced_safety: InducedModeParameters | None = None,
    status_thresholds: PhysicalStatusParameters | None = None,
    sample_count: int = 101,
    exploratory_enabled: bool = False,
    start_time: float = 0.0,
    mode_name: str = "Physical Mode",
) -> PhysicalTargetPlan:
    """Validate a physical-mode target request, including transfer-path feasibility."""

    source = np.asarray(source_position, dtype=float)
    target = np.asarray(requested_target, dtype=float)
    allocator_mode = allocator_mode_for_force_model(force_model) if mode is None else mode
    exploratory_supported = isinstance(force_model, InducedDipoleForceModel)
    endpoint = check_hover_feasibility(
        force_model,
        target,
        mass,
        gravity,
        mode=allocator_mode,
        ball_radius=ball_radius,
        induced_safety=induced_safety,
    )
    trajectory: QuinticTrajectory | None = None
    path: PathCheckResult | None = None
    if np.linalg.norm(target - source) > 1.0e-9:
        trajectory = QuinticTrajectory(source, target, trajectory_duration, start_time=start_time)
        path = check_path_feasibility(
            force_model,
            trajectory,
            mass,
            gravity,
            mode=allocator_mode,
            ball_radius=ball_radius,
            induced_safety=induced_safety,
            sample_count=sample_count,
        )
    if status_thresholds is None:
        accepted = bool(endpoint.accepted_by_default and (path is None or path.accepted_by_default or exploratory_enabled))
        status = START_ALLOWED if accepted else START_BLOCKED
        reason = "Transfer accepted." if accepted else "Target blocked."
        assessment = PhysicalOperatingAssessment(
            state=status,
            target_status=ENDPOINT_EXACT_FEASIBLE if endpoint.feasible else ENDPOINT_BLOCKED,
            path_status=PATH_NOT_REQUIRED if path is None else (PATH_FEASIBLE if path.feasible else PATH_BLOCKED),
            color="green" if accepted else "red",
            target_color="green" if endpoint.feasible else "red",
            start_allowed=accepted,
            start_requires_override=False,
            physical_feasibility="accepted" if accepted else "blocked",
            execution_mode=EXECUTION_DIRECT_TRAJECTORY if path is not None else EXECUTION_HOLD,
            execution_mode_text="Execution: Nominal trajectory." if path is not None else "Execution: Hold target.",
            endpoint_grade="exact" if endpoint.feasible else "blocked",
            path_grade=None if path is None else ("exact" if path.feasible else "blocked"),
            exact_hover_certified=endpoint.feasible,
            exact_path_certified=None if path is None else path.feasible,
            endpoint_status_text=endpoint.reason,
            path_status_text="-" if path is None else path.reason,
            start_status_text="Start: allowed." if accepted else "Start: blocked.",
            margin_text="-",
            message=reason,
            force_to_weight_ratio=endpoint.force_to_weight_ratio,
            exploratory_enabled=exploratory_enabled,
            guard_active=False,
            unsafe_reason=None,
        )
    else:
        assessment = classify_physical_plan(
            endpoint,
            path,
            status_thresholds,
            mode_name=mode_name,
            exploratory_enabled=exploratory_enabled,
            exploratory_supported=exploratory_supported,
        )
        accepted = assessment.start_allowed
        status = assessment.state
        reason = assessment.message
    return PhysicalTargetPlan(
        requested_target=target,
        source_position=source,
        accepted=accepted,
        status=status,
        reason=reason,
        endpoint=endpoint,
        path=path,
        trajectory=trajectory if accepted else trajectory,
        assessment=assessment,
    )
