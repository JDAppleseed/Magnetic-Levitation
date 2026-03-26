"""Toggleable 3D scene viewer backed by pyqtgraph with a VisPy fallback."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QGuiApplication, QVector3D
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


TARGET_COLORS = {
    "endpoint_exact_feasible": "#5bd1a5",
    "endpoint_marginal": "#f2c14e",
    "endpoint_blocked": "#ef6f6c",
    "idle": "#5bd1a5",
}
PANEL_BACKGROUND = "#14181f"
PANEL_SURFACE = "#1d2430"
TEXT_COLOR = "#edf2f7"
BALL_COLOR = "#edf2f7"
ACTIVE_TARGET_COLOR = "#58a6ff"
LAST_FEASIBLE_COLOR = "#b38cff"
CUBE_EDGE_COLOR = "#465165"


@dataclass(frozen=True)
class Scene3DState:
    """Rendering-only snapshot for the 3D scene."""

    current: np.ndarray
    target: np.ndarray
    active_target: np.ndarray | None
    last_feasible_target: np.ndarray | None
    target_status: str
    ball_radius: float
    side_length: float


def _rgba(color: str, alpha: float = 1.0) -> tuple[float, float, float, float]:
    rgba = QColor(color).getRgbF()
    return rgba[0], rgba[1], rgba[2], alpha


def _cube_segments(side_length: float) -> np.ndarray:
    side = float(side_length)
    corners = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [side, 0.0, 0.0],
            [side, side, 0.0],
            [0.0, side, 0.0],
            [0.0, 0.0, side],
            [side, 0.0, side],
            [side, side, side],
            [0.0, side, side],
        ],
        dtype=float,
    )
    edges = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    )
    return np.asarray([corners[index] for edge in edges for index in edge], dtype=float)


class _Unavailable3DWidget(QWidget):
    """Fallback placeholder when no interactive 3D backend is available."""

    def __init__(self, message: str) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        label = QLabel(message)
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet(
            f"background:{PANEL_SURFACE}; border:1px solid #2b3442; border-radius:8px; padding:16px;"
        )
        layout.addStretch(1)
        layout.addWidget(label)
        layout.addStretch(1)

    def set_scene_state(self, state: Scene3DState) -> None:
        del state


class _PyQtGraph3DWidget(QWidget):
    """Interactive 3D scene rendered via pyqtgraph.opengl."""

    def __init__(self) -> None:
        super().__init__()
        import pyqtgraph.opengl as gl

        class OrbitPanView(gl.GLViewWidget):
            def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
                position = event.position() if hasattr(event, "position") else event.localPos()
                if not hasattr(self, "mousePos"):
                    self.mousePos = position
                delta = position - self.mousePos
                self.mousePos = position
                buttons = event.buttons()
                if buttons & Qt.LeftButton:
                    if event.modifiers() & Qt.ControlModifier:
                        self.pan(delta.x(), delta.y(), 0, relative="view")
                    else:
                        self.orbit(-delta.x(), delta.y())
                    return
                if buttons & (Qt.RightButton | Qt.MiddleButton):
                    self.pan(delta.x(), delta.y(), 0, relative="view-upright")
                    return
                super().mouseMoveEvent(event)

        self._gl = gl
        self._ball_radius = 0.0
        self._side_length = 0.0
        self._marker_size = 0.035
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.view = OrbitPanView()
        self.view.setMinimumHeight(420)
        self.view.setBackgroundColor(QColor(PANEL_BACKGROUND))
        layout.addWidget(self.view)

        self.cube_item = gl.GLLinePlotItem(
            pos=np.empty((0, 3), dtype=float),
            color=_rgba(CUBE_EDGE_COLOR, alpha=0.85),
            width=1.5,
            antialias=True,
            mode="lines",
        )
        self.view.addItem(self.cube_item)

        self.ball_item = gl.GLMeshItem(
            meshdata=gl.MeshData.sphere(rows=16, cols=24, radius=1.0),
            color=_rgba(BALL_COLOR, alpha=0.92),
            smooth=True,
            drawFaces=True,
            drawEdges=False,
            glOptions="translucent",
        )
        self.view.addItem(self.ball_item)

        self.requested_target_item = gl.GLScatterPlotItem(
            pos=np.empty((0, 3), dtype=float),
            color=_rgba(TARGET_COLORS["idle"], alpha=0.95),
            size=self._marker_size,
            pxMode=False,
        )
        self.view.addItem(self.requested_target_item)

        self.active_target_item = gl.GLScatterPlotItem(
            pos=np.empty((0, 3), dtype=float),
            color=_rgba(ACTIVE_TARGET_COLOR, alpha=0.95),
            size=self._marker_size * 0.95,
            pxMode=False,
        )
        self.view.addItem(self.active_target_item)

        self.last_feasible_item = gl.GLScatterPlotItem(
            pos=np.empty((0, 3), dtype=float),
            color=_rgba(LAST_FEASIBLE_COLOR, alpha=0.95),
            size=self._marker_size * 0.95,
            pxMode=False,
        )
        self.view.addItem(self.last_feasible_item)

    def set_scene_state(self, state: Scene3DState) -> None:
        state = Scene3DState(
            current=np.asarray(state.current, dtype=float),
            target=np.asarray(state.target, dtype=float),
            active_target=None if state.active_target is None else np.asarray(state.active_target, dtype=float),
            last_feasible_target=None
            if state.last_feasible_target is None
            else np.asarray(state.last_feasible_target, dtype=float),
            target_status=str(state.target_status),
            ball_radius=float(state.ball_radius),
            side_length=float(state.side_length),
        )
        self._update_camera(state.side_length)
        self._update_cube(state.side_length)
        self._update_ball(state.current, state.ball_radius)
        self.requested_target_item.setData(
            pos=np.asarray([state.target], dtype=float),
            color=_rgba(TARGET_COLORS.get(state.target_status, TARGET_COLORS["idle"]), alpha=0.95),
            size=self._marker_size,
            pxMode=False,
        )
        self._set_optional_marker(self.active_target_item, state.active_target, ACTIVE_TARGET_COLOR)
        self._set_optional_marker(self.last_feasible_item, state.last_feasible_target, LAST_FEASIBLE_COLOR)

    def _update_camera(self, side_length: float) -> None:
        side = float(side_length)
        if abs(side - self._side_length) <= 1.0e-9:
            return
        self._side_length = side
        center = side / 2.0
        self.view.setCameraPosition(distance=max(1.8, side * 2.1), elevation=22.0, azimuth=34.0)
        self.view.opts["center"] = QVector3D(center, center, center)
        self.view.update()

    def _update_cube(self, side_length: float) -> None:
        self.cube_item.setData(
            pos=_cube_segments(side_length),
            color=_rgba(CUBE_EDGE_COLOR, alpha=0.85),
            width=1.5,
            antialias=True,
            mode="lines",
        )

    def _update_ball(self, position: np.ndarray, ball_radius: float) -> None:
        radius = float(ball_radius)
        if abs(radius - self._ball_radius) > 1.0e-9:
            self._ball_radius = radius
            self.ball_item.setMeshData(meshdata=self._gl.MeshData.sphere(rows=16, cols=24, radius=radius))
        self.ball_item.resetTransform()
        self.ball_item.translate(float(position[0]), float(position[1]), float(position[2]))

    def _set_optional_marker(self, item, point: np.ndarray | None, color: str) -> None:
        if point is None:
            item.setData(pos=np.empty((0, 3), dtype=float), color=_rgba(color, alpha=0.95), size=self._marker_size, pxMode=False)
            return
        item.setData(pos=np.asarray([point], dtype=float), color=_rgba(color, alpha=0.95), size=self._marker_size, pxMode=False)


class _VisPy3DWidget(QWidget):
    """Minimal VisPy fallback if pyqtgraph.opengl is unavailable."""

    def __init__(self) -> None:
        super().__init__()
        from vispy import scene
        from vispy.visuals.transforms import STTransform

        self._scene = scene
        self._transform_type = STTransform
        self._marker_size = 14.0
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.canvas = scene.SceneCanvas(keys=None, show=False, bgcolor=PANEL_BACKGROUND)
        self.canvas.native.setMinimumHeight(420)
        layout.addWidget(self.canvas.native)

        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.TurntableCamera(
            up="z",
            fov=45.0,
            center=(0.5, 0.5, 0.5),
            distance=2.1,
            azimuth=34.0,
            elevation=22.0,
        )

        self.cube_item = scene.visuals.Line(
            pos=np.empty((0, 3), dtype=float),
            color=_rgba(CUBE_EDGE_COLOR, alpha=0.9),
            width=1.0,
            connect="segments",
            method="gl",
            antialias=True,
            parent=self.view.scene,
        )
        self.ball_item = scene.visuals.Sphere(
            radius=1.0,
            rows=16,
            cols=24,
            method="latitude",
            color=_rgba(BALL_COLOR, alpha=0.92),
            parent=self.view.scene,
        )
        self.ball_item.transform = STTransform(scale=(1.0, 1.0, 1.0), translate=(0.5, 0.5, 0.5))
        self.requested_target_item = scene.visuals.Markers(parent=self.view.scene)
        self.active_target_item = scene.visuals.Markers(parent=self.view.scene)
        self.last_feasible_item = scene.visuals.Markers(parent=self.view.scene)

    def set_scene_state(self, state: Scene3DState) -> None:
        self.view.camera.center = tuple(np.asarray([state.side_length / 2.0] * 3, dtype=float))
        self.view.camera.distance = max(1.8, float(state.side_length) * 2.1)
        self.cube_item.set_data(pos=_cube_segments(state.side_length), color=_rgba(CUBE_EDGE_COLOR, alpha=0.9))
        self.ball_item.transform = self._transform_type(
            scale=(float(state.ball_radius), float(state.ball_radius), float(state.ball_radius)),
            translate=tuple(np.asarray(state.current, dtype=float)),
        )
        self._set_marker(
            self.requested_target_item,
            np.asarray(state.target, dtype=float),
            TARGET_COLORS.get(state.target_status, TARGET_COLORS["idle"]),
        )
        self._set_marker(self.active_target_item, state.active_target, ACTIVE_TARGET_COLOR)
        self._set_marker(self.last_feasible_item, state.last_feasible_target, LAST_FEASIBLE_COLOR)
        self.canvas.update()

    def _set_marker(self, visual, point: np.ndarray | None, color: str) -> None:
        if point is None:
            visual.visible = False
            return
        visual.visible = True
        rgba = _rgba(color, alpha=0.95)
        visual.set_data(
            pos=np.asarray([point], dtype=float),
            face_color=rgba,
            edge_color=rgba,
            size=self._marker_size,
        )


class ThreeDViewPanel(QWidget):
    """3D panel wrapper with backend selection and shared scene-state updates."""

    def __init__(self) -> None:
        super().__init__()
        self.backend_name = "unavailable"
        self.backend_message = ""
        self.setObjectName("threeDViewPanel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        title = QLabel("3D View")
        title.setStyleSheet("font-weight: 600;")
        layout.addWidget(title)

        hint = QLabel("Left drag: orbit  |  Right drag: pan  |  Scroll: zoom")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #c9d2de;")
        layout.addWidget(hint)

        self.backend_label = QLabel("")
        self.backend_label.setWordWrap(True)
        self.backend_label.setStyleSheet(
            f"background:{PANEL_SURFACE}; border:1px solid #2b3442; border-radius:6px; padding:6px; color:{TEXT_COLOR};"
        )
        layout.addWidget(self.backend_label)

        self.scene_widget = self._build_backend()
        layout.addWidget(self.scene_widget, stretch=1)

    def _build_backend(self) -> QWidget:
        if self._is_headless_platform():
            self.backend_name = "headless"
            self.backend_message = "3D rendering is disabled on the Qt offscreen platform used for headless runs."
            self.backend_label.setText("Backend: headless placeholder")
            return _Unavailable3DWidget(self.backend_message)
        try:
            widget = _PyQtGraph3DWidget()
        except Exception as exc:
            pyqtgraph_error = exc
        else:
            self.backend_name = "pyqtgraph"
            self.backend_message = "Backend: pyqtgraph.opengl"
            self.backend_label.setText(self.backend_message)
            return widget

        try:
            widget = _VisPy3DWidget()
        except Exception as exc:
            self.backend_name = "unavailable"
            self.backend_message = (
                f"3D view unavailable. pyqtgraph.opengl failed with: {pyqtgraph_error}. "
                f"VisPy fallback failed with: {exc}."
            )
            self.backend_label.setText("Backend: unavailable")
            return _Unavailable3DWidget(self.backend_message)

        self.backend_name = "vispy"
        self.backend_message = f"Backend: VisPy fallback (pyqtgraph.opengl unavailable: {pyqtgraph_error})"
        self.backend_label.setText(self.backend_message)
        return widget

    def set_state(
        self,
        current: np.ndarray,
        target: np.ndarray,
        active_target: np.ndarray | None,
        last_feasible_target: np.ndarray | None,
        target_status: str,
        ball_radius: float,
        side_length: float,
    ) -> None:
        self.scene_widget.set_scene_state(
            Scene3DState(
                current=np.asarray(current, dtype=float),
                target=np.asarray(target, dtype=float),
                active_target=None if active_target is None else np.asarray(active_target, dtype=float),
                last_feasible_target=None
                if last_feasible_target is None
                else np.asarray(last_feasible_target, dtype=float),
                target_status=str(target_status),
                ball_radius=float(ball_radius),
                side_length=float(side_length),
            )
        )

    @staticmethod
    def _is_headless_platform() -> bool:
        platform_name = (QGuiApplication.platformName() or "").lower()
        env_platform = os.environ.get("QT_QPA_PLATFORM", "").lower()
        return any(name in {"offscreen", "minimal"} for name in (platform_name, env_platform))
