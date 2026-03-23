#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
from physics.force_feasibility import evaluate_hover_feasibility
from physics.zero_force_model import ZeroMagneticForceModel
from sim.state import RigidBodyState


def run_free_fall() -> None:
    system = load_system_parameters()
    backend = build_backend(system, enable_contact=False)
    model = ZeroMagneticForceModel()
    defaults = load_sim_defaults()
    state0 = RigidBodyState(np.array([0.5, 0.5, 0.5], dtype=float), np.zeros(3, dtype=float), 0.0)
    log = backend.simulate(
        state0,
        model,
        duration=defaults.free_fall_duration,
        dt=system.backend.dt,
        control_input_fn=lambda _t, _s: np.zeros(6, dtype=float),
    )
    accelerations = np.asarray(log.accelerations)
    rel_error = abs(np.mean(accelerations[:, 2]) + system.cube.gravity) / system.cube.gravity
    output_dir = ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "free_fall.png"
    export_log_csv(log, output_dir / "free_fall.csv")
    export_summary_json(
        {
            "mode": "free",
            "mean_z_acceleration": float(np.mean(accelerations[:, 2])),
            "gravity": -system.cube.gravity,
            "relative_error": float(rel_error),
        },
        output_dir / "free_fall_summary.json",
    )
    times = np.asarray(log.times)
    positions = np.asarray(log.positions)
    velocities = np.asarray(log.velocities)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(times, positions[:, 2], lw=2.0)
    axes[0].set_ylabel("z [m]")
    axes[0].set_title("Free-fall sanity check")
    axes[1].plot(times, velocities[:, 2], lw=2.0)
    axes[1].set_ylabel("vz [m/s]")
    axes[1].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    print(f"Free-fall mean z-acceleration: {np.mean(accelerations[:, 2]):.6f} m/s^2")
    print(f"Relative error vs -g: {rel_error:.6%}")
    print(f"Saved outputs to {figure_path}")


def run_hover() -> None:
    system = load_system_parameters()
    defaults = load_sim_defaults()
    controller_cfg = load_controller_config()
    backend = build_backend(system, enable_contact=True)
    model = build_affine_model(system)
    hover = evaluate_hover_feasibility(model, defaults.hover_point, system.ball.mass, system.cube.gravity)
    controller = TwoLayerController(
        model,
        system.ball.mass,
        system.cube.gravity,
        controller_cfg.gains,
        allocator_mode=controller_cfg.allocator_mode,
        integral_limit=controller_cfg.integral_limit,
    )
    state0 = RigidBodyState(defaults.hover_point + np.array([0.05, 0.0, 0.0], dtype=float), np.zeros(3, dtype=float), 0.0)
    log = backend.simulate(
        state0,
        model,
        duration=defaults.regulation_duration,
        dt=system.backend.dt,
        controller=controller,
        target_provider=lambda _t: defaults.hover_point,
    )
    errors = np.linalg.norm(np.asarray(log.tracking_errors), axis=1)
    output_dir = ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "hover_demo.png"
    export_log_csv(log, output_dir / "hover_demo.csv")
    export_summary_json(
        {
            "mode": "hover",
            "hover_residual_norm": hover.allocation.residual_norm,
            "hover_u": hover.allocation.u.tolist(),
            "final_error_norm": float(errors[-1]),
            "min_error_norm": float(np.min(errors)),
        },
        output_dir / "hover_demo_summary.json",
    )
    times = np.asarray(log.times)
    positions = np.asarray(log.positions)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(times, positions[:, 0], label="x")
    axes[0].plot(times, positions[:, 1], label="y")
    axes[0].plot(times, positions[:, 2], label="z")
    axes[0].axhline(defaults.hover_point[0], color="C0", linestyle="--", alpha=0.4)
    axes[0].axhline(defaults.hover_point[1], color="C1", linestyle="--", alpha=0.4)
    axes[0].axhline(defaults.hover_point[2], color="C2", linestyle="--", alpha=0.4)
    axes[0].set_ylabel("position [m]")
    axes[0].legend()
    axes[1].plot(times, errors, color="black", lw=2.0)
    axes[1].set_ylabel("||e|| [m]")
    axes[1].set_xlabel("time [s]")
    axes[1].set_title("Local regulation error")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    print(f"Hover feasibility residual: {hover.allocation.residual_norm:.3e}")
    print(f"Final regulation error: {errors[-1]:.6e} m")
    print(f"Saved outputs to {figure_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run free-fall or hover demo.")
    parser.add_argument("--mode", choices=("free", "hover"), default="hover")
    args = parser.parse_args()
    if args.mode == "free":
        run_free_fall()
    else:
        run_hover()


if __name__ == "__main__":
    main()

