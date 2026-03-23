"""CSV and JSON export helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from analysis.logs import SimulationLog


def export_log_csv(log: SimulationLog, path: str | Path) -> None:
    """Write the simulation log to CSV."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "time",
                "x",
                "y",
                "z",
                "vx",
                "vy",
                "vz",
                "ax",
                "ay",
                "az",
                "ux0",
                "ux1",
                "uy0",
                "uy1",
                "uz0",
                "uz1",
                "cmd_fx",
                "cmd_fy",
                "cmd_fz",
                "ach_fx",
                "ach_fy",
                "ach_fz",
                "err_x",
                "err_y",
                "err_z",
                "residual_norm",
                "proof_regime",
            ]
        )
        for index, time in enumerate(log.times):
            writer.writerow(
                [
                    time,
                    *log.positions[index],
                    *log.velocities[index],
                    *log.accelerations[index],
                    *log.control_inputs[index],
                    *log.commanded_forces[index],
                    *log.achieved_forces[index],
                    *log.tracking_errors[index],
                    log.residual_norms[index],
                    log.proof_regimes[index],
                ]
            )


def export_summary_json(summary: dict[str, Any], path: str | Path) -> None:
    """Write a summary dictionary to JSON."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

