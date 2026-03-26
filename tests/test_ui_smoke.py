from __future__ import annotations

import os

from PySide6.QtWidgets import QApplication

from ui.main_window import MainWindow
from ui.theme import apply_theme, load_theme


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        app = QApplication([])
        apply_theme(app, load_theme())
    return app


def test_main_window_3d_toggle_smoke() -> None:
    app = _app()
    window = MainWindow()
    window.show()
    app.processEvents()

    assert window.view_tabs.indexOf(window.projections) != -1
    assert window.view_tabs.indexOf(window.view_3d_panel) == -1

    window.controls.show_3d_check.setChecked(True)
    app.processEvents()
    assert window.view_tabs.indexOf(window.view_3d_panel) != -1
    assert window.view_3d_panel.backend_name in {"headless", "pyqtgraph", "vispy", "unavailable"}

    window.controls.show_3d_check.setChecked(False)
    app.processEvents()
    assert window.view_tabs.indexOf(window.view_3d_panel) == -1

    window.close()
    app.processEvents()
