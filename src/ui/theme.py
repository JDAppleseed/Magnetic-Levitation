"""Dark theme helpers for the PySide6 UI."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import QApplication

from analysis.config import load_yaml, repo_path


@dataclass(frozen=True)
class Theme:
    """UI color palette."""

    background: str
    panel: str
    accent: str
    warning: str
    danger: str
    text: str

    def qcolor(self, name: str) -> QColor:
        return QColor(getattr(self, name))


def load_theme() -> Theme:
    """Load the configured UI theme."""

    raw = load_yaml(repo_path("configs", "ui.yaml"))
    colors = raw["theme"]
    return Theme(
        background=str(colors["background"]),
        panel=str(colors["panel"]),
        accent=str(colors["accent"]),
        warning=str(colors["warning"]),
        danger=str(colors["danger"]),
        text=str(colors["text"]),
    )


def apply_theme(app: QApplication, theme: Theme) -> None:
    """Apply the dark stylesheet to the app."""

    app.setFont(QFont("Helvetica Neue", 12))
    app.setStyleSheet(
        f"""
        QWidget {{
            background: {theme.background};
            color: {theme.text};
            font-size: 12px;
            font-family: "Helvetica Neue";
        }}
        QMainWindow {{
            background: {theme.background};
        }}
        QGroupBox {{
            border: 1px solid {theme.panel};
            border-radius: 8px;
            margin-top: 10px;
            padding-top: 8px;
            background: {theme.panel};
            font-weight: 600;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 12px;
            padding: 0 4px;
        }}
        QPushButton {{
            background: {theme.accent};
            color: #11161d;
            border: none;
            border-radius: 6px;
            padding: 8px 12px;
            font-weight: 700;
        }}
        QPushButton:disabled {{
            background: #4a5565;
            color: #c9d2de;
        }}
        QComboBox, QDoubleSpinBox, QLabel[frameShape="4"] {{
            background: #11161d;
            border: 1px solid #2b3442;
            border-radius: 6px;
            padding: 4px 6px;
        }}
        QComboBox QAbstractItemView {{
            background: #11161d;
            color: {theme.text};
        }}
        """
    )
