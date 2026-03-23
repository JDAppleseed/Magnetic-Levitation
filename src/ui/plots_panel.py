"""Live plots for tracking error and force realization."""

from __future__ import annotations

from collections import deque

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QVBoxLayout, QWidget


class PlotsPanel(QWidget):
    """PyQtGraph plots for live simulation diagnostics."""

    def __init__(self, history_seconds: float = 8.0) -> None:
        super().__init__()
        self.history_seconds = float(history_seconds)
        self.times: deque[float] = deque()
        self.error_norms: deque[float] = deque()
        self.z_values: deque[float] = deque()
        self.z_targets: deque[float] = deque()
        self.cmd_force_norms: deque[float] = deque()
        self.ach_force_norms: deque[float] = deque()

        layout = QVBoxLayout(self)
        pg.setConfigOptions(antialias=True)

        self.error_plot = pg.PlotWidget(title="Tracking Error / Height")
        self.force_plot = pg.PlotWidget(title="Commanded vs Achieved Force Norm")
        self.error_plot.addLegend()
        self.force_plot.addLegend()
        self.error_curve = self.error_plot.plot(pen=pg.mkPen("#5bd1a5", width=2), name="||e||")
        self.z_curve = self.error_plot.plot(pen=pg.mkPen("#9ad1ff", width=2), name="z")
        self.z_target_curve = self.error_plot.plot(
            pen=pg.mkPen("#f2c14e", width=2, style=Qt.PenStyle.DashLine),
            name="z_target",
        )
        self.cmd_curve = self.force_plot.plot(pen=pg.mkPen("#ef6f6c", width=2), name="||F_cmd||")
        self.ach_curve = self.force_plot.plot(pen=pg.mkPen("#edf2f7", width=2), name="||F_ach||")
        layout.addWidget(self.error_plot, stretch=1)
        layout.addWidget(self.force_plot, stretch=1)

    def reset(self) -> None:
        for series in (
            self.times,
            self.error_norms,
            self.z_values,
            self.z_targets,
            self.cmd_force_norms,
            self.ach_force_norms,
        ):
            series.clear()
        self._refresh()

    def append_sample(
        self,
        time: float,
        error_norm: float,
        z_value: float,
        z_target: float,
        commanded_force_norm: float,
        achieved_force_norm: float,
    ) -> None:
        self.times.append(float(time))
        self.error_norms.append(float(error_norm))
        self.z_values.append(float(z_value))
        self.z_targets.append(float(z_target))
        self.cmd_force_norms.append(float(commanded_force_norm))
        self.ach_force_norms.append(float(achieved_force_norm))
        while self.times and self.times[-1] - self.times[0] > self.history_seconds:
            self.times.popleft()
            self.error_norms.popleft()
            self.z_values.popleft()
            self.z_targets.popleft()
            self.cmd_force_norms.popleft()
            self.ach_force_norms.popleft()
        self._refresh()

    def _refresh(self) -> None:
        times = np.asarray(self.times, dtype=float)
        self.error_curve.setData(times, np.asarray(self.error_norms, dtype=float))
        self.z_curve.setData(times, np.asarray(self.z_values, dtype=float))
        self.z_target_curve.setData(times, np.asarray(self.z_targets, dtype=float))
        self.cmd_curve.setData(times, np.asarray(self.cmd_force_norms, dtype=float))
        self.ach_curve.setData(times, np.asarray(self.ach_force_norms, dtype=float))
