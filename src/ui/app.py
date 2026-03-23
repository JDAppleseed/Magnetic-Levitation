"""UI application entry point."""

from __future__ import annotations

import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from ui.main_window import MainWindow
from ui.theme import apply_theme, load_theme


def main(smoke_test: bool = False) -> int:
    app = QApplication(sys.argv)
    apply_theme(app, load_theme())
    window = MainWindow()
    window.show()
    if smoke_test:
        QTimer.singleShot(150, app.quit)
    return app.exec()
