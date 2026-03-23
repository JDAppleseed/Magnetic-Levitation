"""Control and diagnostics panel."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


@dataclass(frozen=True)
class UIStateSnapshot:
    """Formatted diagnostics for the UI."""

    position: np.ndarray
    velocity: np.ndarray
    tracking_error: np.ndarray
    commanded_force: np.ndarray
    achieved_force: np.ndarray
    actuator_vector: np.ndarray
    residual_norm: float
    feasibility_margin: float
    saturated: bool
    proof_regime: str
    physical_feasibility: str = "-"
    force_to_weight_ratio: float | None = None
    induced_stability: str = "-"
    warning_text: str = "-"
    endpoint_status: str = "-"
    path_status: str = "-"
    start_status: str = "-"
    execution_mode_text: str = "-"
    margin_text: str = "-"
    operating_state_text: str = "-"
    active_target_text: str = "-"
    last_feasible_target_text: str = "-"


class ControlsPanel(QWidget):
    """Holds user inputs and live diagnostics."""

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumWidth(360)
        self.setMaximumWidth(430)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        setup_tab = QWidget()
        setup_layout = QVBoxLayout(setup_tab)
        setup_layout.setContentsMargins(4, 4, 4, 4)
        setup_layout.setSpacing(8)

        self.model_box = QGroupBox("Model")
        model_form = QFormLayout(self.model_box)
        self.force_model_combo = QComboBox()
        self.force_model_combo.addItems(["Free Dynamics", "Affine Surrogate", "Fixed Dipole", "Induced Dipole"])
        self.physical_preset_combo = QComboBox()
        self.physical_preset_combo.addItems(["demo_calibrated_physical", "conservative_physical"])
        self.physical_preset_combo.setCurrentText("demo_calibrated_physical")
        self.exploratory_check = QCheckBox("Allow guarded exploratory mode")
        self.exploratory_check.setChecked(False)
        self.ball_radius_spin = self._spin(0.01, 0.20, 0.05, step=0.005)
        self.ball_mass_spin = self._spin(0.05, 2.00, 0.25, step=0.01)
        self.u_min_spin = self._spin(-20.0, 20.0, 0.0, step=0.1)
        self.u_max_spin = self._spin(0.0, 50.0, 12.0, step=0.1)
        model_form.addRow("Force model", self.force_model_combo)
        model_form.addRow("Physical preset", self.physical_preset_combo)
        model_form.addRow("Ball radius [m]", self.ball_radius_spin)
        model_form.addRow("Ball mass [kg]", self.ball_mass_spin)
        model_form.addRow("u_min", self.u_min_spin)
        model_form.addRow("u_max", self.u_max_spin)
        model_form.addRow("", self.exploratory_check)
        setup_layout.addWidget(self.model_box)

        self.gains_box = QGroupBox("Controller Gains")
        gains_form = QFormLayout(self.gains_box)
        self.kp_xy_spin = self._spin(0.0, 100.0, 18.0, step=0.5)
        self.kp_z_spin = self._spin(0.0, 100.0, 26.0, step=0.5)
        self.kd_xy_spin = self._spin(0.0, 50.0, 8.0, step=0.25)
        self.kd_z_spin = self._spin(0.0, 50.0, 10.0, step=0.25)
        self.ki_spin = self._spin(0.0, 20.0, 0.0, step=0.1)
        gains_form.addRow("Kp xy", self.kp_xy_spin)
        gains_form.addRow("Kp z", self.kp_z_spin)
        gains_form.addRow("Kd xy", self.kd_xy_spin)
        gains_form.addRow("Kd z", self.kd_z_spin)
        gains_form.addRow("Ki", self.ki_spin)
        setup_layout.addWidget(self.gains_box)

        self.points_box = QGroupBox("Start / Target")
        points_grid = QGridLayout(self.points_box)
        self.start_spins = [self._spin(0.0, 1.0, value, step=0.01) for value in (0.30, 0.30, 0.75)]
        self.target_spins = [self._spin(0.0, 1.0, value, step=0.01) for value in (0.70, 0.65, 0.55)]
        for col, label in enumerate(("x", "y", "z"), start=1):
            points_grid.addWidget(QLabel(label), 0, col)
        points_grid.addWidget(QLabel("start"), 1, 0)
        points_grid.addWidget(QLabel("target"), 2, 0)
        for col, spin in enumerate(self.start_spins, start=1):
            points_grid.addWidget(spin, 1, col)
        for col, spin in enumerate(self.target_spins, start=1):
            points_grid.addWidget(spin, 2, col)
        hint = QLabel("Left-click projection: set target\nRight-click projection: set start")
        hint.setWordWrap(True)
        points_grid.addWidget(hint, 3, 0, 1, 4)
        setup_layout.addWidget(self.points_box)

        self.buttons_box = QGroupBox("Simulation")
        button_layout = QHBoxLayout(self.buttons_box)
        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.reset_button = QPushButton("Reset")
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.reset_button)
        setup_layout.addWidget(self.buttons_box)

        self.target_status_box = QGroupBox("Target Assessment")
        target_status_form = QFormLayout(self.target_status_box)
        self.endpoint_status_label = self._value_label()
        self.path_status_label = self._value_label()
        self.margin_status_label = self._value_label()
        self.start_status_label = self._value_label()
        self.execution_mode_label = self._value_label()
        self.active_target_label = self._value_label()
        self.last_feasible_target_label = self._value_label()
        target_status_form.addRow("Start", self.start_status_label)
        target_status_form.addRow("Execution", self.execution_mode_label)
        target_status_form.addRow("Endpoint hover", self.endpoint_status_label)
        target_status_form.addRow("Path", self.path_status_label)
        target_status_form.addRow("Margins", self.margin_status_label)
        target_status_form.addRow("Active hold", self.active_target_label)
        target_status_form.addRow("Last feasible", self.last_feasible_target_label)
        setup_layout.addWidget(self.target_status_box)
        setup_layout.addStretch(1)

        status_tab = QWidget()
        status_layout = QVBoxLayout(status_tab)
        status_layout.setContentsMargins(4, 4, 4, 4)
        status_layout.setSpacing(8)

        self.diag_box = QGroupBox("Live Diagnostics")
        diag_layout = QFormLayout(self.diag_box)
        diag_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        diag_layout.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.regime_indicator = QLabel("green")
        self.regime_indicator.setAlignment(Qt.AlignCenter)
        self.physical_feasibility_label = self._value_label()
        self.force_weight_label = self._value_label()
        self.stability_label = self._value_label()
        self.warning_label = self._value_label()
        self.position_label = self._value_label()
        self.velocity_label = self._value_label()
        self.error_label = self._value_label()
        self.commanded_force_label = self._value_label()
        self.achieved_force_label = self._value_label()
        self.actuator_label = self._value_label()
        self.feasibility_label = self._value_label()
        self.saturation_label = self._value_label()
        diag_layout.addRow("Proof regime", self.regime_indicator)
        diag_layout.addRow("Physical feasibility", self.physical_feasibility_label)
        diag_layout.addRow("Force / weight", self.force_weight_label)
        diag_layout.addRow("Induced stability", self.stability_label)
        diag_layout.addRow("Position [m]", self.position_label)
        diag_layout.addRow("Velocity [m/s]", self.velocity_label)
        diag_layout.addRow("Tracking error [m]", self.error_label)
        diag_layout.addRow("Commanded F [N]", self.commanded_force_label)
        diag_layout.addRow("Achieved F [N]", self.achieved_force_label)
        diag_layout.addRow("Actuators", self.actuator_label)
        diag_layout.addRow("Residual / margin", self.feasibility_label)
        diag_layout.addRow("Saturation", self.saturation_label)
        diag_layout.addRow("Warnings", self.warning_label)
        status_layout.addWidget(self.diag_box)
        status_layout.addStretch(1)
        self.tabs.addTab(setup_tab, "Setup")
        self.tabs.addTab(status_tab, "Status")

    @staticmethod
    def _spin(minimum: float, maximum: float, value: float, step: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setDecimals(4)
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setValue(value)
        return spin

    @staticmethod
    def _value_label() -> QLabel:
        label = QLabel("-")
        label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        label.setWordWrap(True)
        return label

    def start_position(self) -> np.ndarray:
        return np.asarray([spin.value() for spin in self.start_spins], dtype=float)

    def target_position(self) -> np.ndarray:
        return np.asarray([spin.value() for spin in self.target_spins], dtype=float)

    def set_start_position(self, position: np.ndarray) -> None:
        for spin, value in zip(self.start_spins, position):
            spin.setValue(float(value))

    def set_target_position(self, position: np.ndarray) -> None:
        for spin, value in zip(self.target_spins, position):
            spin.setValue(float(value))

    def gains(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        kp = np.array([self.kp_xy_spin.value(), self.kp_xy_spin.value(), self.kp_z_spin.value()], dtype=float)
        kd = np.array([self.kd_xy_spin.value(), self.kd_xy_spin.value(), self.kd_z_spin.value()], dtype=float)
        ki = np.full(3, self.ki_spin.value(), dtype=float)
        return kp, kd, ki

    def actuator_bounds(self) -> tuple[float, float]:
        return float(self.u_min_spin.value()), float(self.u_max_spin.value())

    def selected_force_model(self) -> str:
        return str(self.force_model_combo.currentText())

    def selected_physical_preset(self) -> str:
        return str(self.physical_preset_combo.currentText())

    def exploratory_enabled(self) -> bool:
        return bool(self.exploratory_check.isChecked())

    def update_diagnostics(self, snapshot: UIStateSnapshot) -> None:
        self.position_label.setText(self._fmt(snapshot.position))
        self.velocity_label.setText(self._fmt(snapshot.velocity))
        self.error_label.setText(self._fmt(snapshot.tracking_error))
        self.commanded_force_label.setText(self._fmt(snapshot.commanded_force))
        self.achieved_force_label.setText(self._fmt(snapshot.achieved_force))
        self.actuator_label.setText(self._fmt(snapshot.actuator_vector))
        self.feasibility_label.setText(
            f"residual={snapshot.residual_norm:.3e}, margin={snapshot.feasibility_margin:.3f}"
        )
        self.saturation_label.setText("yes" if snapshot.saturated else "no")
        self.physical_feasibility_label.setText(snapshot.physical_feasibility)
        if snapshot.force_to_weight_ratio is None:
            self.force_weight_label.setText("-")
        else:
            self.force_weight_label.setText(f"{snapshot.force_to_weight_ratio:.2f} x body weight")
        self.stability_label.setText(snapshot.induced_stability)
        self.warning_label.setText(snapshot.warning_text)
        self.start_status_label.setText(snapshot.start_status)
        self.execution_mode_label.setText(snapshot.execution_mode_text)
        self.endpoint_status_label.setText(snapshot.endpoint_status)
        self.path_status_label.setText(snapshot.path_status)
        self.margin_status_label.setText(snapshot.margin_text)
        self.active_target_label.setText(snapshot.active_target_text)
        self.last_feasible_target_label.setText(snapshot.last_feasible_target_text)
        self.regime_indicator.setText(snapshot.proof_regime)
        colors = {
            "green": "#2b8a57",
            "yellow": "#c89526",
            "red": "#bb4d4d",
        }
        color = colors.get(snapshot.proof_regime, "#4a5565")
        self.regime_indicator.setStyleSheet(f"background:{color}; border-radius: 6px; padding: 6px;")

    def set_controller_enabled(self, enabled: bool) -> None:
        self.gains_box.setEnabled(enabled)

    def set_physical_mode_enabled(self, enabled: bool) -> None:
        self.physical_preset_combo.setEnabled(enabled)
        self.exploratory_check.setEnabled(enabled)

    @staticmethod
    def _fmt(values: np.ndarray) -> str:
        return "[" + ", ".join(f"{value:+.3f}" for value in np.asarray(values, dtype=float)) + "]"
