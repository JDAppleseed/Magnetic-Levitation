"""Main UI window."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QHBoxLayout, QMainWindow, QScrollArea, QSplitter, QWidget

from analysis.config import load_yaml, repo_path
from analysis.runtime import load_sim_defaults
from control.outer_loop import DiagonalGains
from control.physical_mode_guard import guard_induced_command
from control.physical_targeting import (
    EXECUTION_DIRECT_TRAJECTORY,
    PhysicalOperatingAssessment,
    PhysicalTargetPlan,
    PhysicalTargetState,
    START_ALLOWED_CAUTION,
    START_BLOCKED,
    START_REQUIRES_EXPLORATORY_OVERRIDE,
    START_UNSAFE_ABORT,
    allocator_mode_for_force_model,
    classify_physical_plan,
    plan_physical_target_request,
)
from control.pid import TwoLayerController
from physics.affine_force_model import AffineFaceMagneticForceModel
from physics.dipole_force import DifferenceSettings, FixedDipoleForceModel
from physics.induced_dipole_force import InducedDipoleForceModel
from physics.magnetic_field_model import (
    BackendParameters,
    BallProperties,
    CubeGeometry,
    FaceDipoleFieldModel,
    MagneticCoupling,
    SystemParameters,
    build_face_actuators,
    load_system_parameters,
)
from physics.zero_force_model import ZeroMagneticForceModel
from sim.rk4_backend import RK4Backend
from sim.state import RigidBodyState
from ui.controls_panel import ControlsPanel, UIStateSnapshot
from ui.plots_panel import PlotsPanel
from ui.projections_panel import ProjectionsPanel


class MainWindow(QMainWindow):
    """Interactive desktop front end."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Magnetic Levitation Control Panel")
        ui_config = load_yaml(repo_path("configs", "ui.yaml"))
        self.defaults = load_sim_defaults()
        self.resize(int(ui_config["window"]["width"]), int(ui_config["window"]["height"]))
        self.setMinimumSize(1280, 820)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        self.controls = ControlsPanel()
        self.projections = ProjectionsPanel()
        self.plots = PlotsPanel(history_seconds=float(ui_config["plots"]["history_seconds"]))
        controls_scroll = QScrollArea()
        controls_scroll.setWidget(self.controls)
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        controls_scroll.setMinimumWidth(380)
        controls_scroll.setStyleSheet("QScrollArea { border: none; }")
        center_splitter = QSplitter(Qt.Horizontal)
        center_splitter.addWidget(self.projections)
        center_splitter.addWidget(self.plots)
        center_splitter.setStretchFactor(0, 3)
        center_splitter.setStretchFactor(1, 2)
        center_splitter.setSizes([860, 420])
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(controls_scroll)
        splitter.addWidget(center_splitter)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([390, 1110])
        layout.addWidget(splitter, stretch=1)

        self.base_system = load_system_parameters()
        self.system = self.base_system
        self.force_model: object = ZeroMagneticForceModel()
        self.backend = self._build_backend(self.system)
        self.controller: TwoLayerController | None = None
        self.state = RigidBodyState(np.array([0.5, 0.5, 0.5], dtype=float), np.zeros(3, dtype=float), 0.0)
        self.start_position = self.controls.start_position()
        self.target_position = self.controls.target_position()
        self.last_applied_u = np.zeros(6, dtype=float)
        self.physical_targets = PhysicalTargetState()
        self.preview_plan: PhysicalTargetPlan | None = None
        self.current_assessment: PhysicalOperatingAssessment | None = None
        self._runtime_guard_active = False
        self._runtime_guard_events = 0
        self._runtime_unsafe_reason: str | None = None
        self._physical_feasibility_text = "-"
        self._force_to_weight_ratio: float | None = None
        self._induced_stability_text = "-"
        self._warning_text = "-"
        self._exploratory_text = "off"

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer_interval_ms = 20
        self.substeps = 4

        self.controls.start_button.clicked.connect(self.start_simulation)
        self.controls.stop_button.clicked.connect(self.stop_simulation)
        self.controls.reset_button.clicked.connect(self.reset_simulation)
        self.controls.force_model_combo.currentIndexChanged.connect(self.reset_simulation)
        self.controls.physical_preset_combo.currentIndexChanged.connect(self.reset_simulation)
        self.controls.exploratory_check.toggled.connect(self._handle_exploratory_toggle)
        for spin in (self.controls.ball_radius_spin, self.controls.ball_mass_spin, self.controls.u_min_spin, self.controls.u_max_spin):
            spin.valueChanged.connect(self.reset_simulation)
        for spin in (
            self.controls.kp_xy_spin,
            self.controls.kp_z_spin,
            self.controls.kd_xy_spin,
            self.controls.kd_z_spin,
            self.controls.ki_spin,
        ):
            spin.valueChanged.connect(self._handle_gain_change)
        for spin in self.controls.start_spins:
            spin.valueChanged.connect(self._sync_start_from_controls)
        for spin in self.controls.target_spins:
            spin.valueChanged.connect(self._sync_target_from_controls)
        self.projections.pointSelected.connect(self._handle_projection_pick)
        self.statusBar().showMessage("Left-click a projection to set the target. Right-click to set the start.")

        self.reset_simulation()

    def reset_simulation(self) -> None:
        self.stop_simulation()
        self.system = self._system_from_controls()
        self.backend = self._build_backend(self.system)
        self.force_model = self._build_force_model()
        self.controller = self._build_controller()
        self.last_applied_u = np.zeros(6, dtype=float)
        self.physical_targets.reset()
        self.preview_plan = None
        self.current_assessment = None
        self._runtime_guard_active = False
        self._runtime_guard_events = 0
        self._runtime_unsafe_reason = None
        self.controls.set_controller_enabled(self.controller is not None)
        self.controls.set_physical_mode_enabled(self._is_physical_mode())
        self.controls.exploratory_check.setEnabled(self.controls.selected_force_model() == "Induced Dipole")
        lower, upper = self.system.cube.admissible_bounds(self.system.ball.radius)
        self.start_position = np.clip(self.controls.start_position(), lower, upper)
        self.target_position = np.clip(self.controls.target_position(), lower, upper)
        self.controls.set_start_position(self.start_position)
        self.controls.set_target_position(self.target_position)
        self.state = RigidBodyState(self.start_position.copy(), np.zeros(3, dtype=float), 0.0)
        if self.controller is not None:
            self.controller.reset()
        self.substeps = max(1, int(round((self.timer_interval_ms / 1000.0) / self.system.backend.dt)))
        self.plots.reset()
        self._refresh_target_plan(clear_abort=True)
        self._update_views(
            tracking_error=self.state.position - self._display_target(),
            commanded_force=np.zeros(3, dtype=float),
            achieved_force=np.zeros(3, dtype=float),
            actuator_vector=np.zeros(6, dtype=float),
            residual_norm=0.0,
            feasibility_margin=self._display_feasibility_margin(),
            saturated=False,
            proof_regime=self._display_regime("green"),
        )

    def start_simulation(self) -> None:
        self._refresh_target_plan(clear_abort=False)
        if self._is_physical_mode():
            if self.current_assessment is None:
                return
            if self.current_assessment.state == START_UNSAFE_ABORT or not self.current_assessment.start_allowed:
                self.statusBar().showMessage(self._warning_text)
                self._update_start_button_state()
                return
            self.physical_targets.activate_pending()
        self._runtime_guard_events = 0
        self.timer.start(self.timer_interval_ms)
        self._update_start_button_state()

    def stop_simulation(self) -> None:
        self.timer.stop()
        self._update_start_button_state()

    def _tick(self) -> None:
        dt = self.system.backend.dt
        self._runtime_guard_active = False
        last_tracking_error = np.zeros(3, dtype=float)
        last_commanded_force = np.zeros(3, dtype=float)
        last_achieved_force = np.zeros(3, dtype=float)
        last_u = np.zeros(6, dtype=float)
        last_residual_norm = 0.0
        last_margin = self._display_feasibility_margin()
        last_saturated = False
        last_regime = self._display_regime("green")
        transfer_completed = False

        for _ in range(self.substeps):
            if self.controller is None:
                u = np.zeros(6, dtype=float)
                commanded_force = np.zeros(3, dtype=float)
                reference = self._display_target()
                tracking_error = self.state.position - reference
                regime = "green"
                saturated = False
                residual_norm = 0.0
                margin = 1.0
            else:
                reference, completed = self._reference_for_current_step()
                transfer_completed = transfer_completed or completed
                if hasattr(reference, "position"):
                    u, snapshot = self.controller.compute_tracking(
                        self.state.position,
                        self.state.velocity,
                        reference.position,
                        reference.velocity,
                        reference.acceleration,
                        dt,
                    )
                else:
                    u, snapshot = self.controller.compute_regulation(
                        self.state.position,
                        self.state.velocity,
                        np.asarray(reference, dtype=float),
                        dt,
                    )
                commanded_force = snapshot.commanded_force
                tracking_error = snapshot.error
                regime = snapshot.proof_regime
                saturated = snapshot.allocation.saturated
                residual_norm = snapshot.allocation.residual_norm
                margin = 1.0 - snapshot.allocation.saturation_fraction
                if isinstance(self.force_model, InducedDipoleForceModel):
                    guard = guard_induced_command(
                        self.force_model,
                        self.state.position,
                        self.state.velocity,
                        u,
                        self.last_applied_u,
                        dt,
                        self.system.ball.radius,
                        self.system.induced_mode,
                        exploratory_enabled=self._exploratory_enabled(),
                    )
                    if guard.aborted:
                        self._runtime_unsafe_reason = "Induced mode aborted: " + "; ".join(guard.reasons)
                        self.physical_targets.set_unsafe_abort(self._runtime_unsafe_reason)
                        self._sync_operating_assessment()
                        self.stop_simulation()
                        self._update_views(
                            tracking_error=tracking_error,
                            commanded_force=commanded_force,
                            achieved_force=np.zeros(3, dtype=float),
                            actuator_vector=np.zeros(6, dtype=float),
                            residual_norm=residual_norm,
                            feasibility_margin=0.0,
                            saturated=saturated,
                            proof_regime="red",
                        )
                        return
                    u = guard.u
                    self.last_applied_u = u.copy()
                    self.controller.set_last_input(u)
                    if guard.status != "ok":
                        self._runtime_guard_active = True
                        self._runtime_guard_events += 1
                else:
                    self.last_applied_u = u.copy()

            step = self.backend.step(self.state, self.force_model, u, dt)
            self.state = step.state
            last_u = u
            last_tracking_error = tracking_error
            last_commanded_force = commanded_force
            last_achieved_force = step.magnetic_force
            last_regime = regime
            last_saturated = saturated
            last_residual_norm = residual_norm
            last_margin = margin
            self.plots.append_sample(
                time=self.state.time,
                error_norm=float(np.linalg.norm(tracking_error)),
                z_value=float(self.state.position[2]),
                z_target=float(self._display_target()[2]),
                commanded_force_norm=float(np.linalg.norm(commanded_force)),
                achieved_force_norm=float(np.linalg.norm(step.magnetic_force)),
            )

        if transfer_completed:
            self._refresh_target_plan(clear_abort=False)
        else:
            self._complete_fallback_plan_if_settled()
            self._sync_operating_assessment()
        self._update_views(
            tracking_error=last_tracking_error,
            commanded_force=last_commanded_force,
            achieved_force=last_achieved_force,
            actuator_vector=last_u,
            residual_norm=last_residual_norm,
            feasibility_margin=last_margin,
            saturated=last_saturated,
            proof_regime=self._display_regime(last_regime),
        )

    def _update_views(
        self,
        tracking_error: np.ndarray,
        commanded_force: np.ndarray,
        achieved_force: np.ndarray,
        actuator_vector: np.ndarray,
        residual_norm: float,
        feasibility_margin: float,
        saturated: bool,
        proof_regime: str,
    ) -> None:
        assessment = self.current_assessment
        self.projections.set_state(
            current=self.state.position,
            start=self.start_position,
            target=self.target_position,
            active_target=self.physical_targets.active_goal(),
            last_feasible_target=self.physical_targets.last_feasible_target,
            target_status="idle" if assessment is None else assessment.target_status,
            ball_radius=self.system.ball.radius,
            side_length=self.system.cube.side_length,
        )
        self.controls.update_diagnostics(
            UIStateSnapshot(
                position=self.state.position,
                velocity=self.state.velocity,
                tracking_error=tracking_error,
                commanded_force=commanded_force,
                achieved_force=achieved_force,
                actuator_vector=actuator_vector,
                residual_norm=residual_norm,
                feasibility_margin=feasibility_margin,
                saturated=saturated,
                proof_regime=proof_regime,
                physical_feasibility=self._physical_feasibility_text,
                force_to_weight_ratio=self._force_to_weight_ratio,
                induced_stability=self._induced_stability_text,
                warning_text=self._warning_text,
                endpoint_status="-" if assessment is None else assessment.endpoint_status_text,
                path_status="-" if assessment is None else assessment.path_status_text,
                start_status="-" if assessment is None else assessment.start_status_text,
                execution_mode_text="-" if assessment is None else assessment.execution_mode_text,
                margin_text="-" if assessment is None else assessment.margin_text,
                operating_state_text="-" if assessment is None else assessment.physical_feasibility,
                active_target_text=self._fmt_vector(self.physical_targets.active_goal()),
                last_feasible_target_text=self._fmt_vector(self.physical_targets.last_feasible_target),
            )
        )

    def _handle_projection_pick(self, plane: str, role: str, a: float, b: float) -> None:
        point = self.target_position.copy() if role == "target" else self.start_position.copy()
        if plane == "xy":
            point[0], point[1] = a, b
        elif plane == "xz":
            point[0], point[2] = a, b
        else:
            point[1], point[2] = a, b
        lower, upper = self.system.cube.admissible_bounds(self.system.ball.radius)
        point = np.clip(point, lower, upper)
        if role == "target":
            self.target_position = point
            self.controls.set_target_position(point)
            self._refresh_target_plan(clear_abort=True)
        else:
            self.start_position = point
            self.controls.set_start_position(point)
            if not self.timer.isActive():
                self.state = RigidBodyState(self.start_position.copy(), np.zeros(3, dtype=float), self.state.time)
            self._refresh_target_plan(clear_abort=True)
        self._update_views(
            tracking_error=self.state.position - self._display_target(),
            commanded_force=np.zeros(3, dtype=float),
            achieved_force=np.zeros(3, dtype=float),
            actuator_vector=np.zeros(6, dtype=float),
            residual_norm=0.0,
            feasibility_margin=self._display_feasibility_margin(),
            saturated=False,
            proof_regime=self._display_regime("green"),
        )

    def _sync_start_from_controls(self) -> None:
        lower, upper = self.system.cube.admissible_bounds(self.system.ball.radius)
        self.start_position = np.clip(self.controls.start_position(), lower, upper)
        self.controls.set_start_position(self.start_position)
        if not self.timer.isActive():
            self.state = RigidBodyState(self.start_position.copy(), np.zeros(3, dtype=float), self.state.time)
        self._refresh_target_plan(clear_abort=True)
        self._update_views(
            tracking_error=self.state.position - self._display_target(),
            commanded_force=np.zeros(3, dtype=float),
            achieved_force=np.zeros(3, dtype=float),
            actuator_vector=np.zeros(6, dtype=float),
            residual_norm=0.0,
            feasibility_margin=self._display_feasibility_margin(),
            saturated=False,
            proof_regime=self._display_regime("green"),
        )

    def _sync_target_from_controls(self) -> None:
        lower, upper = self.system.cube.admissible_bounds(self.system.ball.radius)
        self.target_position = np.clip(self.controls.target_position(), lower, upper)
        self.controls.set_target_position(self.target_position)
        self._refresh_target_plan(clear_abort=True)
        self._update_views(
            tracking_error=self.state.position - self._display_target(),
            commanded_force=np.zeros(3, dtype=float),
            achieved_force=np.zeros(3, dtype=float),
            actuator_vector=np.zeros(6, dtype=float),
            residual_norm=0.0,
            feasibility_margin=self._display_feasibility_margin(),
            saturated=False,
            proof_regime=self._display_regime("green"),
        )

    def _handle_gain_change(self) -> None:
        self.controller = self._build_controller()
        if self.controller is not None:
            self.controller.set_last_input(self.last_applied_u)
        self._refresh_target_plan(clear_abort=False)

    def _handle_exploratory_toggle(self) -> None:
        self._refresh_target_plan(clear_abort=True)

    def _system_from_controls(self) -> SystemParameters:
        cube = CubeGeometry(side_length=self.base_system.cube.side_length, gravity=self.base_system.cube.gravity)
        ball = BallProperties(
            radius=float(self.controls.ball_radius_spin.value()),
            mass=float(self.controls.ball_mass_spin.value()),
            inertia=self.base_system.ball.inertia,
        )
        preset_name = self.controls.selected_physical_preset()
        preset = self.base_system.physical_presets[preset_name]
        mode_name = self.controls.selected_force_model()
        dt_key = {
            "Free Dynamics": "free",
            "Affine Surrogate": "affine",
            "Fixed Dipole": "fixed_dipole",
            "Induced Dipole": "induced_dipole",
        }[mode_name]
        u_min, u_max = self.controls.actuator_bounds()
        actuators = build_face_actuators(
            cube.side_length,
            u_min,
            u_max,
            self.base_system.actuators[0].affine_gain,
            self.base_system.actuators[0].decay_power,
            self.base_system.actuators[0].decay_offset,
            preset.dipole_strength,
        )
        backend = BackendParameters(
            name=self.base_system.backend.name,
            dt=self.base_system.backend.dt_by_mode.get(dt_key, self.base_system.backend.dt),
            dt_by_mode=dict(self.base_system.backend.dt_by_mode),
            damping=self.base_system.backend.damping,
            contact_mode=self.base_system.backend.contact_mode,
            restitution=self.base_system.backend.restitution,
        )
        coupling = MagneticCoupling(
            fixed_dipole_moment=preset.fixed_dipole_moment.copy(),
            induced_alpha=preset.induced_alpha,
            physical_preset=preset.name,
        )
        return SystemParameters(
            cube=cube,
            ball=ball,
            backend=backend,
            actuators=actuators,
            coupling=coupling,
            physical_presets=dict(self.base_system.physical_presets),
            induced_mode=self.base_system.induced_mode,
            physical_status=self.base_system.physical_status,
        )

    def _build_force_model(self) -> object:
        name = self.controls.selected_force_model()
        if name == "Free Dynamics":
            return ZeroMagneticForceModel()
        if name == "Affine Surrogate":
            return AffineFaceMagneticForceModel(self.system.cube, self.system.actuators)
        field = FaceDipoleFieldModel(self.system.cube, self.system.actuators)
        if name == "Fixed Dipole":
            return FixedDipoleForceModel(
                field,
                self.system.coupling.fixed_dipole_moment,
                diff=DifferenceSettings(position_eps=self.system.induced_mode.finite_difference_eps),
            )
        return InducedDipoleForceModel(
            field,
            self.system.coupling.induced_alpha,
            diff=DifferenceSettings(position_eps=self.system.induced_mode.finite_difference_eps),
        )

    def _build_controller(self) -> TwoLayerController | None:
        if self.controls.selected_force_model() == "Free Dynamics":
            return None
        kp, kd, ki = self.controls.gains()
        gains = DiagonalGains(kp=kp, kd=kd, ki=ki)
        return TwoLayerController(
            self.force_model,
            self.system.ball.mass,
            self.system.cube.gravity,
            gains,
            allocator_mode=allocator_mode_for_force_model(self.force_model),
            integral_limit=0.2,
        )

    @staticmethod
    def _build_backend(system: SystemParameters) -> RK4Backend:
        return RK4Backend(
            side_length=system.cube.side_length,
            ball_radius=system.ball.radius,
            mass=system.ball.mass,
            gravity=system.cube.gravity,
            damping=system.backend.damping,
            contact_mode=system.backend.contact_mode,
            restitution=system.backend.restitution,
            enable_contact=True,
        )

    def _refresh_target_plan(self, clear_abort: bool) -> None:
        if clear_abort:
            self._runtime_unsafe_reason = None
            self._runtime_guard_events = 0
            self.physical_targets.clear_unsafe_abort()
        self.preview_plan = None
        if self.controls.selected_force_model() == "Free Dynamics":
            self.current_assessment = None
            self._physical_feasibility_text = "not applicable"
            self._force_to_weight_ratio = None
            self._induced_stability_text = "not applicable"
            self._warning_text = "Free dynamics: no magnetic actuation."
            self._exploratory_text = "off"
            self.statusBar().showMessage(self._warning_text)
            self._update_start_button_state()
            return
        if not self._is_physical_mode():
            self.current_assessment = None
            self._physical_feasibility_text = "surrogate mode"
            self._force_to_weight_ratio = None
            self._induced_stability_text = "not applicable"
            self._warning_text = "Affine surrogate: controller prototype mode."
            self._exploratory_text = "off"
            self.statusBar().showMessage(self._warning_text)
            self._update_start_button_state()
            return

        self.preview_plan = plan_physical_target_request(
            self.force_model,
            self.state.position,
            self.target_position,
            self.system.ball.mass,
            self.system.cube.gravity,
            trajectory_duration=self.defaults.trajectory_duration,
            mode=allocator_mode_for_force_model(self.force_model),
            ball_radius=self.system.ball.radius,
            induced_safety=self.system.induced_mode,
            status_thresholds=self.system.physical_status,
            sample_count=self.system.physical_status.path_sample_count,
            exploratory_enabled=self._exploratory_enabled(),
            start_time=self.state.time,
            mode_name=self.controls.selected_force_model(),
        )
        self.physical_targets.apply_request(self.preview_plan, simulation_active=self.timer.isActive())
        self._sync_operating_assessment()

    def _sync_operating_assessment(self) -> None:
        if self.preview_plan is None:
            self._update_start_button_state()
            return
        self.current_assessment = classify_physical_plan(
            self.preview_plan.endpoint,
            self.preview_plan.path,
            self.system.physical_status,
            mode_name=self.controls.selected_force_model(),
            exploratory_enabled=self._exploratory_enabled(),
            exploratory_supported=isinstance(self.force_model, InducedDipoleForceModel),
            runtime_guard_active=self._runtime_guard_active,
            runtime_unsafe_reason=self._runtime_unsafe_reason,
        )
        self._physical_feasibility_text = self.current_assessment.physical_feasibility
        self._force_to_weight_ratio = self.current_assessment.force_to_weight_ratio
        self._induced_stability_text = self._induced_status_text()
        self._exploratory_text = "enabled" if self._exploratory_enabled() else "off"
        self._warning_text = self.current_assessment.message
        if self.current_assessment.state == START_ALLOWED_CAUTION and self._runtime_guard_events > 0:
            self._warning_text += f" Guard events: {self._runtime_guard_events}."
        self.statusBar().showMessage(self._warning_text)
        self._update_start_button_state()

    def _update_start_button_state(self) -> None:
        if self.timer.isActive():
            self.controls.start_button.setEnabled(False)
            self.controls.stop_button.setEnabled(True)
            return
        if self.controls.selected_force_model() in {"Free Dynamics", "Affine Surrogate"}:
            self.controls.start_button.setEnabled(True)
        elif self.current_assessment is not None:
            self.controls.start_button.setEnabled(self.current_assessment.start_allowed)
        else:
            self.controls.start_button.setEnabled(False)
        self.controls.stop_button.setEnabled(False)

    def _reference_for_current_step(self) -> tuple[np.ndarray | object, bool]:
        if not self._is_physical_mode():
            return self.target_position.copy(), False
        had_active = self.physical_targets.active_plan is not None
        reference = self.physical_targets.active_reference(self.state.time)
        completed = had_active and self.physical_targets.active_plan is None
        if reference is None:
            reference = self.physical_targets.active_goal()
        if reference is None:
            reference = self.target_position.copy()
        return reference, completed

    def _complete_fallback_plan_if_settled(self) -> None:
        active_plan = self.physical_targets.active_plan
        if active_plan is None:
            return
        if active_plan.assessment.execution_mode == EXECUTION_DIRECT_TRAJECTORY:
            return
        position_error = float(np.linalg.norm(self.state.position - active_plan.requested_target))
        velocity_norm = float(np.linalg.norm(self.state.velocity))
        if position_error <= 0.01 and velocity_norm <= 0.05:
            self.physical_targets.transfer_completed()

    def _display_target(self) -> np.ndarray:
        if self._is_physical_mode():
            goal = self.physical_targets.active_goal()
            if goal is not None:
                return goal.copy()
        return self.target_position.copy()

    def _display_feasibility_margin(self) -> float:
        if self.current_assessment is None:
            return 1.0
        if self.preview_plan is None:
            return 1.0
        path_margin = self.preview_plan.path.minimum_margin if self.preview_plan.path is not None else self.preview_plan.endpoint.margin_estimate
        return float(min(self.preview_plan.endpoint.margin_estimate, path_margin))

    def _display_regime(self, controller_regime: str) -> str:
        severity = {"green": 0, "yellow": 1, "red": 2}
        physical_regime = "green" if self.current_assessment is None else self.current_assessment.color
        return max((controller_regime, physical_regime), key=lambda key: severity[key])

    def _is_physical_mode(self) -> bool:
        return self.controls.selected_force_model() in {"Fixed Dipole", "Induced Dipole"}

    def _exploratory_enabled(self) -> bool:
        return self.controls.selected_force_model() == "Induced Dipole" and self.controls.exploratory_enabled()

    def _induced_status_text(self) -> str:
        if self.controls.selected_force_model() != "Induced Dipole":
            return "not applicable"
        if self.current_assessment is None:
            return "not applicable"
        if self.current_assessment.state == START_UNSAFE_ABORT:
            return "unsafe abort"
        if self.current_assessment.state == START_BLOCKED:
            return "blocked"
        if self.current_assessment.execution_mode == "guarded_fallback" or self.current_assessment.state == START_ALLOWED_CAUTION:
            return "guarded / caution"
        return "nominal"

    @staticmethod
    def _fmt_vector(vector: np.ndarray | None) -> str:
        if vector is None:
            return "-"
        values = np.asarray(vector, dtype=float)
        return "[" + ", ".join(f"{value:+.3f}" for value in values) + "]"
