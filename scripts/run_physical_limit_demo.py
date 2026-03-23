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

from analysis.export import export_summary_json
from analysis.runtime import build_physical_models, load_system_parameters
from physics.decay_analysis import analyze_decay_limit
from physics.earnshaw_demos import analyze_fixed_input_potential


def main() -> None:
    system = load_system_parameters()
    fixed_model, _ = build_physical_models(system)
    lower, upper = system.cube.admissible_bounds(system.ball.radius)
    u_star = np.ones(6, dtype=float)
    earnshaw = analyze_fixed_input_potential(
        lambda x: fixed_model.scalar_potential(x, u_star),
        lower + 0.05,
        upper - 0.05,
        samples_per_axis=5,
    )
    decay = analyze_decay_limit(
        side_length=1.1,
        mass=system.ball.mass,
        gravity=system.cube.gravity,
        u_max=system.actuators[0].u_max,
        decay_constant=system.actuators[0].affine_gain,
        alpha=system.actuators[0].decay_power,
    )

    output_dir = ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "physical_limit_demo.png"
    export_summary_json(
        {
            "earnshaw_strict_minimum_count": earnshaw.strict_minimum_count,
            "earnshaw_sample_count": earnshaw.sample_count,
            "earnshaw_max_abs_trace": earnshaw.max_abs_trace,
            "decay_center_force_bound": decay.center_force_bound,
            "decay_required_force": decay.required_force,
            "decay_critical_side_length": decay.critical_side_length,
            "decay_center_hover_feasible": decay.center_hover_feasible,
        },
        output_dir / "physical_limit_demo_summary.json",
    )

    center_y = 0.5 * (lower[1] + upper[1])
    xs = np.linspace(lower[0] + 0.05, upper[0] - 0.05, 80)
    zs = np.linspace(lower[2] + 0.05, upper[2] - 0.05, 80)
    grid = np.zeros((zs.size, xs.size), dtype=float)
    for ix, x in enumerate(xs):
        for iz, z in enumerate(zs):
            point = np.array([x, center_y, z], dtype=float)
            grid[iz, ix] = fixed_model.scalar_potential(point, u_star)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    image = axes[0].imshow(
        grid,
        origin="lower",
        extent=[xs[0], xs[-1], zs[0], zs[-1]],
        aspect="equal",
        cmap="magma",
    )
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("z [m]")
    axes[0].set_title("Fixed-input potential slice")
    fig.colorbar(image, ax=axes[0], shrink=0.82)
    traces = [sample.trace for sample in earnshaw.samples]
    minima = [np.min(sample.eigenvalues) for sample in earnshaw.samples]
    axes[1].scatter(traces, minima, s=18, alpha=0.8)
    axes[1].axvline(0.0, color="black", linestyle="--", alpha=0.5)
    axes[1].axhline(0.0, color="black", linestyle="--", alpha=0.5)
    axes[1].set_xlabel("trace(H)")
    axes[1].set_ylabel("min eigenvalue(H)")
    axes[1].set_title("Earnshaw diagnostic samples")
    fig.savefig(figure_path, dpi=160)

    print(f"Earnshaw strict minima found: {earnshaw.strict_minimum_count} / {earnshaw.sample_count}")
    print(f"Max |trace(H)| over samples: {earnshaw.max_abs_trace:.3e}")
    print(f"Decay critical side length L*: {decay.critical_side_length:.3f} m")
    print(f"Saved outputs to {figure_path}")


if __name__ == "__main__":
    main()

