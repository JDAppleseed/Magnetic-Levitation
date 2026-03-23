"""Orthographic projection widgets with click-based point picking."""

from __future__ import annotations

from typing import Literal

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPen
from PySide6.QtWidgets import QGridLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget


PlaneName = Literal["xy", "xz", "yz"]


class ProjectionView(QWidget):
    """Single 2D orthographic projection."""

    pointSelected = Signal(str, str, float, float)

    def __init__(self, plane: PlaneName) -> None:
        super().__init__()
        self.plane = plane
        self.current = np.array([0.5, 0.5, 0.5], dtype=float)
        self.start = self.current.copy()
        self.target = self.current.copy()
        self.active_target: np.ndarray | None = self.current.copy()
        self.last_feasible_target: np.ndarray | None = self.current.copy()
        self.target_status = "endpoint_exact_feasible"
        self.ball_radius = 0.05
        self.side_length = 1.0
        self.setMinimumSize(220, 220)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_state(
        self,
        current: np.ndarray,
        start: np.ndarray,
        target: np.ndarray,
        active_target: np.ndarray | None,
        last_feasible_target: np.ndarray | None,
        target_status: str,
        ball_radius: float,
        side_length: float,
    ) -> None:
        self.current = np.asarray(current, dtype=float)
        self.start = np.asarray(start, dtype=float)
        self.target = np.asarray(target, dtype=float)
        self.active_target = None if active_target is None else np.asarray(active_target, dtype=float)
        self.last_feasible_target = None if last_feasible_target is None else np.asarray(last_feasible_target, dtype=float)
        self.target_status = str(target_status)
        self.ball_radius = float(ball_radius)
        self.side_length = float(side_length)
        self.update()

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        lower = self.ball_radius
        upper = self.side_length - self.ball_radius
        rect = self.rect().adjusted(18, 18, -18, -18)
        x0, x1 = self._screen_to_plane(event.position(), rect, lower, upper)
        role = "target" if event.button() == Qt.LeftButton else "start"
        self.pointSelected.emit(self.plane, role, x0, x1)
        super().mousePressEvent(event)

    def paintEvent(self, event) -> None:  # type: ignore[override]
        del event
        lower = self.ball_radius
        upper = self.side_length - self.ball_radius
        rect = self.rect().adjusted(18, 18, -18, -18)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#1d2430"))
        painter.setPen(QPen(QColor("#465165"), 1.0))
        painter.drawRect(rect)

        for values, color in ((self.start, QColor("#f2c14e")),):
            point = self._plane_to_screen(values, rect, lower, upper)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(color))
            painter.drawEllipse(point, 6.0, 6.0)

        if self.last_feasible_target is not None:
            point = self._plane_to_screen(self.last_feasible_target, rect, lower, upper)
            painter.setPen(QPen(QColor("#b38cff"), 2.0))
            painter.drawLine(point.x() - 6.0, point.y() - 6.0, point.x() + 6.0, point.y() + 6.0)
            painter.drawLine(point.x() - 6.0, point.y() + 6.0, point.x() + 6.0, point.y() - 6.0)

        if self.active_target is not None:
            point = self._plane_to_screen(self.active_target, rect, lower, upper)
            painter.setPen(QPen(QColor("#58a6ff"), 2.0))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(QRectF(point.x() - 7.0, point.y() - 7.0, 14.0, 14.0))

        target_color = {
            "endpoint_exact_feasible": QColor("#5bd1a5"),
            "endpoint_marginal": QColor("#e0b34f"),
            "endpoint_blocked": QColor("#df6b6b"),
            "idle": QColor("#5bd1a5"),
        }.get(self.target_status, QColor("#5bd1a5"))
        target_point = self._plane_to_screen(self.target, rect, lower, upper)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(target_color))
        painter.drawEllipse(target_point, 6.0, 6.0)

        point = self._plane_to_screen(self.current, rect, lower, upper)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor("#edf2f7")))
        painter.drawEllipse(point, 6.0, 6.0)

    def _plane_components(self, vector: np.ndarray) -> tuple[float, float]:
        if self.plane == "xy":
            return float(vector[0]), float(vector[1])
        if self.plane == "xz":
            return float(vector[0]), float(vector[2])
        return float(vector[1]), float(vector[2])

    def _plane_to_screen(self, vector: np.ndarray, rect: QRectF, lower: float, upper: float) -> QPointF:
        x0, x1 = self._plane_components(vector)
        sx = rect.left() + (x0 - lower) / (upper - lower) * rect.width()
        sy = rect.bottom() - (x1 - lower) / (upper - lower) * rect.height()
        return QPointF(sx, sy)

    @staticmethod
    def _screen_to_plane(position: QPointF, rect: QRectF, lower: float, upper: float) -> tuple[float, float]:
        px = np.clip((position.x() - rect.left()) / rect.width(), 0.0, 1.0)
        py = np.clip((rect.bottom() - position.y()) / rect.height(), 0.0, 1.0)
        return lower + px * (upper - lower), lower + py * (upper - lower)


class ProjectionsPanel(QWidget):
    """Three orthographic projections."""

    pointSelected = Signal(str, str, float, float)

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        title = QLabel("Orthographic Projections")
        layout.addWidget(title)
        grid = QGridLayout()
        self.views = {
            "xy": ProjectionView("xy"),
            "xz": ProjectionView("xz"),
            "yz": ProjectionView("yz"),
        }
        labels = {"xy": "XY", "xz": "XZ", "yz": "YZ"}
        for index, plane in enumerate(("xy", "xz", "yz")):
            panel = QWidget()
            panel_layout = QVBoxLayout(panel)
            panel_layout.setContentsMargins(0, 0, 0, 0)
            label = QLabel(labels[plane])
            panel_layout.addWidget(label)
            panel_layout.addWidget(self.views[plane], stretch=1)
            grid.addWidget(panel, 0, index)
            self.views[plane].pointSelected.connect(self.pointSelected)
        layout.addLayout(grid)

    def set_state(
        self,
        current: np.ndarray,
        start: np.ndarray,
        target: np.ndarray,
        active_target: np.ndarray | None,
        last_feasible_target: np.ndarray | None,
        target_status: str,
        ball_radius: float,
        side_length: float,
    ) -> None:
        for view in self.views.values():
            view.set_state(
                current,
                start,
                target,
                active_target,
                last_feasible_target,
                target_status,
                ball_radius,
                side_length,
            )
