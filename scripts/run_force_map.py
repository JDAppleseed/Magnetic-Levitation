#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis.runtime import build_affine_model, load_system_parameters
from physics.force_feasibility import evaluate_hover_feasibility


def main() -> None:
    system = load_system_parameters()
    model = build_affine_model(system)
    resolution = 31
    lower, upper = system.cube.admissible_bounds(system.ball.radius)
    xs = np.linspace(lower[0], upper[0], resolution)
    ys = np.linspace(lower[1], upper[1], resolution)
    z = 0.5 * (lower[2] + upper[2])
    residuals = np.zeros((resolution, resolution), dtype=float)
    authority = np.zeros((resolution, resolution), dtype=float)
    feasible = np.zeros((resolution, resolution), dtype=float)
    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            point = np.array([x, y, z], dtype=float)
            hover = evaluate_hover_feasibility(model, point, system.ball.mass, system.cube.gravity)
            residuals[iy, ix] = hover.allocation.residual_norm
            authority[iy, ix] = model.authority(point).singular_values[-1]
            feasible[iy, ix] = 1.0 if hover.allocation.feasible else 0.0

    output_dir = ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "force_map.png"

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for axis, data, title in (
        (axes[0], residuals, "Hover residual norm"),
        (axes[1], authority, "Smallest singular value of G(x)"),
        (axes[2], feasible, "Hover feasible set"),
    ):
        image = axis.imshow(
            data,
            origin="lower",
            extent=[xs[0], xs[-1], ys[0], ys[-1]],
            aspect="equal",
            cmap="viridis",
        )
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        axis.set_title(f"{title}\n(z = {z:.2f} m)")
        fig.colorbar(image, ax=axis, shrink=0.82)
    fig.savefig(figure_path, dpi=160)
    print(f"Saved force-feasibility slice to {figure_path}")


if __name__ == "__main__":
    main()

