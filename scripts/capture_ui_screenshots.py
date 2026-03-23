#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _framework_python_candidate() -> Path | None:
    base_prefix = Path(sys.base_prefix)
    candidate = base_prefix / "Resources" / "Python.app" / "Contents" / "MacOS" / "Python"
    return candidate if candidate.exists() else None


def _should_relaunch_via_python_app() -> bool:
    if sys.platform != "darwin":
        return False
    if os.environ.get("MAGLEV_UI_SKIP_RELAUNCH") == "1":
        return False
    executable = Path(sys.executable).resolve()
    return "Python.app/Contents/MacOS/Python" not in str(executable)


def _relaunch_via_python_app() -> None:
    candidate = _framework_python_candidate()
    if candidate is None:
        return
    env = os.environ.copy()
    entries = [str(ROOT / "src")]
    current_pythonpath = env.get("PYTHONPATH", "")
    if current_pythonpath:
        entries.extend(part for part in current_pythonpath.split(os.pathsep) if part)
    deduped: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        if entry not in seen:
            seen.add(entry)
            deduped.append(entry)
    env["PYTHONPATH"] = os.pathsep.join(deduped)
    env["MAGLEV_UI_SKIP_RELAUNCH"] = "1"
    os.execve(str(candidate), [str(candidate), str(Path(__file__).resolve()), *sys.argv[1:]], env)


if __name__ == "__main__":
    _relaunch_via_python_app() if _should_relaunch_via_python_app() else None
    sys.path.insert(0, str(ROOT / "src"))

    from PySide6.QtWidgets import QApplication

    from ui.main_window import MainWindow
    from ui.theme import apply_theme, load_theme

    app = QApplication(sys.argv)
    apply_theme(app, load_theme())
    window = MainWindow()
    window.show()
    app.processEvents()

    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)

    scenarios = [
        {
            "name": "ui_fixed_feasible",
            "force_model": "Fixed Dipole",
            "preset": "demo_calibrated_physical",
            "exploratory": False,
            "start": [0.50, 0.50, 0.55],
            "target": [0.50, 0.50, 0.55],
            "tab_index": 0,
        },
        {
            "name": "ui_fixed_fallback",
            "force_model": "Fixed Dipole",
            "preset": "demo_calibrated_physical",
            "exploratory": False,
            "start": [0.20, 0.50, 0.50],
            "target": [0.80, 0.50, 0.50],
            "tab_index": 0,
        },
        {
            "name": "ui_fixed_blocked",
            "force_model": "Fixed Dipole",
            "preset": "demo_calibrated_physical",
            "exploratory": False,
            "start": [0.50, 0.50, 0.55],
            "target": [0.50, 0.50, 0.80],
            "tab_index": 0,
        },
        {
            "name": "ui_induced_guarded",
            "force_model": "Induced Dipole",
            "preset": "demo_calibrated_physical",
            "exploratory": False,
            "start": [0.20, 0.50, 0.50],
            "target": [0.80, 0.50, 0.50],
            "tab_index": 1,
        },
    ]

    summary: list[dict[str, object]] = []

    for scenario in scenarios:
        window.controls.force_model_combo.setCurrentText(str(scenario["force_model"]))
        window.controls.physical_preset_combo.setCurrentText(str(scenario["preset"]))
        window.controls.exploratory_check.setChecked(bool(scenario["exploratory"]))
        window.controls.set_start_position(scenario["start"])
        window.controls.set_target_position(scenario["target"])
        window.controls.tabs.setCurrentIndex(int(scenario["tab_index"]))
        window.reset_simulation()
        app.processEvents()
        assessment = window.current_assessment
        image_path = output_dir / f"{scenario['name']}.png"
        pixmap = window.grab()
        pixmap.save(str(image_path))
        summary.append(
            {
                "name": scenario["name"],
                "image": str(image_path),
                "force_model": scenario["force_model"],
                "preset": scenario["preset"],
                "exploratory": scenario["exploratory"],
                "start": scenario["start"],
                "target": scenario["target"],
                "target_status": None if assessment is None else assessment.target_status,
                "path_status": None if assessment is None else assessment.path_status,
                "start_status": None if assessment is None else assessment.state,
                "execution_mode": None if assessment is None else assessment.execution_mode,
                "message": None if assessment is None else assessment.message,
            }
        )

    summary_path = output_dir / "ui_screenshots_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved UI screenshots to {output_dir}")
    print(f"Saved summary to {summary_path}")
