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

from analysis.export import export_log_csv, export_summary_json
from analysis.runtime import build_affine_model, build_backend, load_controller_config, load_sim_defaults, load_system_parameters
from control.pid import TwoLayerController
from control.trajectory_generator import QuinticTrajectory
from physics.force_feasibility import evaluate_trajectory_force_feasibility
from sim.state import RigidBodyState


def main() -> None:
    system = load_system_parameters()
    defaults = load_sim_defaults()
    controller_cfg = load_controller_config()
    backend = build_backend(system, enable_contact=True)
    model = build_affine_model(system)
    trajectory = QuinticTrajectory(defaults.start, defaults.target, defaults.trajectory_duration)
    feasibility = evaluate_trajectory_force_feasibility(model, trajectory, system.ball.mass, system.cube.gravity)
    controller = TwoLayerController(
        model,
        system.ball.mass,
        system.cube.gravity,
        controller_cfg.gains,
        allocator_mode=controller_cfg.allocator_mode,
        integral_limit=controller_cfg.integral_limit,
    )
    state0 = RigidBodyState(defaults.start.copy(), np.zeros(3, dtype=float), 0.0)
    log = backend.simulate(
        state0,
        model,
        duration=defaults.tracking_duration,
        dt=system.backend.dt,
        controller=controller,
        target_provider=trajectory.evaluate,
    )
    final_error = float(np.linalg.norm(log.tracking_errors[-1]))
    rms_error = log.rms_tracking_error()
    output_dir = ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "transfer_demo.png"
    export_log_csv(log, output_dir / "transfer_demo.csv")
    export_summary_json(
        {
            "trajectory_force_feasible": feasibility.all_feasible,
            "trajectory_max_residual_norm": feasibility.max_residual_norm,
            "final_error_norm": final_error,
            "rms_error_norm": rms_error,
        },
        output_dir / "transfer_demo_summary.json",
    )

    times = np.asarray(log.times)
    positions = np.asarray(log.positions)
    refs = np.asarray([trajectory.evaluate(t).position for t in times])
    errors = np.linalg.norm(np.asarray(log.tracking_errors), axis=1)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=False)
    axes[0].plot(positions[:, 0], positions[:, 1], lw=2.0, label="actual")
    axes[0].plot(refs[:, 0], refs[:, 1], "--", lw=2.0, label="reference")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    axes[0].set_title("XY transfer path")
    axes[0].legend()
    axes[1].plot(times, positions[:, 2], label="actual z")
    axes[1].plot(times, refs[:, 2], "--", label="reference z")
    axes[1].set_ylabel("z [m]")
    axes[1].legend()
    axes[2].plot(times, errors, color="black", lw=2.0)
    axes[2].set_ylabel("||e|| [m]")
    axes[2].set_xlabel("time [s]")
    axes[2].set_title("Tracking error")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)

    print(f"Pointwise force-feasible: {feasibility.all_feasible}")
    print(f"Max force-allocation residual: {feasibility.max_residual_norm:.3e}")
    print(f"RMS tracking error: {rms_error:.6e} m")
    print(f"Final tracking error: {final_error:.6e} m")
    print(f"Saved outputs to {figure_path}")


if __name__ == "__main__":
    main()

