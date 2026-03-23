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
from analysis.runtime import build_backend, load_controller_config, load_system_parameters
from control.physical_mode_guard import assess_induced_hover, corner_sampled_max_force_norm
from control.pid import TwoLayerController
from physics.induced_dipole_force import InducedDipoleForceModel
from physics.magnetic_field_model import FaceDipoleFieldModel
from sim.state import RigidBodyState


def _induced_model(preset: str) -> tuple[object, InducedDipoleForceModel]:
    system = load_system_parameters(physical_preset=preset)
    field = FaceDipoleFieldModel(system.cube, system.actuators)
    return system, InducedDipoleForceModel(field, system.coupling.induced_alpha)


def main() -> None:
    output_dir = ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    demo_system, demo_model = _induced_model("demo_calibrated_physical")
    conservative_system, conservative_model = _induced_model("conservative_physical")
    hover_point = np.array([0.5, 0.5, 0.55], dtype=float)
    near_boundary_point = np.array([0.5, 0.5, demo_system.ball.radius + 0.03], dtype=float)
    mg = demo_system.ball.mass * demo_system.cube.gravity

    demo_assessment = assess_induced_hover(
        demo_model,
        hover_point,
        demo_system.ball.mass,
        demo_system.cube.gravity,
        demo_system.ball.radius,
        demo_system.induced_mode,
    )
    conservative_assessment = assess_induced_hover(
        conservative_model,
        hover_point,
        conservative_system.ball.mass,
        conservative_system.cube.gravity,
        conservative_system.ball.radius,
        conservative_system.induced_mode,
    )

    alpha_values = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0], dtype=float)
    alpha_force_ratios = []
    alpha_hover_feasible = []
    for alpha in alpha_values:
        probe_model = InducedDipoleForceModel(
            FaceDipoleFieldModel(demo_system.cube, demo_system.actuators),
            alpha,
            diff=demo_model.diff,
        )
        alpha_force_ratios.append(corner_sampled_max_force_norm(probe_model, hover_point) / mg)
        alpha_hover_feasible.append(
            assess_induced_hover(
                probe_model,
                hover_point,
                demo_system.ball.mass,
                demo_system.cube.gravity,
                demo_system.ball.radius,
                demo_system.induced_mode,
            ).hover.allocation.feasible
        )
    alpha_force_ratios = np.asarray(alpha_force_ratios, dtype=float)
    first_authoritative_alpha = next((float(alpha) for alpha, ratio in zip(alpha_values, alpha_force_ratios) if ratio >= 1.0), None)
    first_hover_feasible_alpha = next((float(alpha) for alpha, feasible in zip(alpha_values, alpha_hover_feasible) if feasible), None)

    hover_u = demo_assessment.hover.allocation.u
    h_values = np.array([1.0e-6, 2.0e-6, 5.0e-6, 1.0e-5, 2.0e-5, 5.0e-5, 1.0e-4], dtype=float)
    analytic_force = demo_model.force(hover_point, hover_u)
    fd_errors = np.array(
        [
            np.linalg.norm(demo_model.force_from_gradient_fd(hover_point, hover_u, h) - analytic_force)
            for h in h_values
        ],
        dtype=float,
    )
    fd_rel_errors = fd_errors / max(np.linalg.norm(analytic_force), 1.0e-12)

    near_boundary_diag = demo_model.diagnostics(near_boundary_point, hover_u, demo_system.ball.radius)
    hover_diag = demo_model.diagnostics(hover_point, hover_u, demo_system.ball.radius)

    controller_cfg = load_controller_config()
    dt_values = np.array([0.005, 0.0025, 0.001, 0.0005], dtype=float)
    dt_final_errors: list[float] = []
    dt_rms_errors: list[float] = []
    dt_max_forces: list[float] = []
    dt_inside_flags: list[bool] = []
    dt_stage_projection_counts: list[int] = []
    for dt in dt_values:
        backend = build_backend(demo_system, enable_contact=True)
        controller = TwoLayerController(
            demo_model,
            demo_system.ball.mass,
            demo_system.cube.gravity,
            controller_cfg.gains,
            allocator_mode="nonlinear",
            integral_limit=controller_cfg.integral_limit,
        )
        state0 = RigidBodyState(hover_point + np.array([0.02, 0.0, 0.01], dtype=float), np.zeros(3, dtype=float), 0.0)
        log = backend.simulate(
            state0,
            demo_model,
            duration=1.0,
            dt=float(dt),
            controller=controller,
            target_provider=lambda _t: hover_point,
        )
        errors = np.linalg.norm(np.asarray(log.tracking_errors), axis=1)
        achieved_forces = np.linalg.norm(np.asarray(log.achieved_forces), axis=1)
        dt_final_errors.append(float(errors[-1]))
        dt_rms_errors.append(float(log.rms_tracking_error()))
        dt_max_forces.append(float(np.max(achieved_forces)))
        dt_inside_flags.append(bool(all(log.inside_admissible_flags)))
        dt_stage_projection_counts.append(int(max(log.stage_projection_counts)))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].loglog(h_values, np.maximum(fd_rel_errors, 1.0e-16), marker="o", lw=2.0)
    axes[0].set_xlabel("finite-difference step h [m]")
    axes[0].set_ylabel("relative force error")
    axes[0].set_title("Induced-force FD sensitivity")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[1].plot(dt_values, dt_final_errors, marker="o", label="final error")
    axes[1].plot(dt_values, dt_rms_errors, marker="s", label="RMS error")
    axes[1].set_xlabel("integration step dt [s]")
    axes[1].set_ylabel("tracking error [m]")
    axes[1].set_title("Induced-mode timestep sweep")
    axes[1].invert_xaxis()
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    figure_path = output_dir / "induced_diagnose.png"
    fig.savefig(figure_path, dpi=160)

    summary = {
        "demo_hover_feasible": demo_assessment.hover.allocation.feasible,
        "demo_hover_residual_norm": demo_assessment.hover.allocation.residual_norm,
        "demo_force_to_weight_ratio": demo_assessment.force_to_weight_ratio,
        "conservative_hover_feasible": conservative_assessment.hover.allocation.feasible,
        "conservative_hover_residual_norm": conservative_assessment.hover.allocation.residual_norm,
        "conservative_force_to_weight_ratio": conservative_assessment.force_to_weight_ratio,
        "alpha_values": alpha_values.tolist(),
        "alpha_force_to_weight_ratios": alpha_force_ratios.tolist(),
        "alpha_hover_feasible": alpha_hover_feasible,
        "first_alpha_with_weight_support": first_authoritative_alpha,
        "first_alpha_with_hover_feasibility": first_hover_feasible_alpha,
        "fd_h_values": h_values.tolist(),
        "fd_relative_errors": fd_rel_errors.tolist(),
        "hover_field_norm": hover_diag.field_norm,
        "hover_grad_norm_sq_norm": hover_diag.grad_norm_sq_norm,
        "boundary_field_norm": near_boundary_diag.field_norm,
        "boundary_grad_norm_sq_norm": near_boundary_diag.grad_norm_sq_norm,
        "dt_values": dt_values.tolist(),
        "dt_final_errors": dt_final_errors,
        "dt_rms_errors": dt_rms_errors,
        "dt_max_forces": dt_max_forces,
        "dt_inside_flags": dt_inside_flags,
        "dt_stage_projection_counts": dt_stage_projection_counts,
    }
    export_summary_json(summary, output_dir / "induced_diagnose_summary.json")

    print("Induced-mode diagnosis")
    print(f"Demo preset hover feasible: {demo_assessment.hover.allocation.feasible} (residual {demo_assessment.hover.allocation.residual_norm:.3e})")
    print(
        f"Conservative preset hover feasible: {conservative_assessment.hover.allocation.feasible} "
        f"(residual {conservative_assessment.hover.allocation.residual_norm:.3e})"
    )
    if first_authoritative_alpha is None:
        print("Alpha sweep: no tested alpha achieved corner-sampled body-weight support at the hover point.")
    else:
        print(f"Alpha sweep: corner-sampled force-to-weight exceeds 1.0 starting near alpha={first_authoritative_alpha:.2f}.")
    if first_hover_feasible_alpha is None:
        print("Alpha sweep: no tested alpha achieved hover feasibility at the nominal hover point.")
    else:
        print(f"Alpha sweep: nominal hover becomes feasible near alpha={first_hover_feasible_alpha:.2f}.")
    print(
        f"FD sensitivity at h={h_values[np.argmin(fd_rel_errors)]:.1e}: relative force error {np.min(fd_rel_errors):.3e}; "
        f"at h={h_values[-1]:.1e}: {fd_rel_errors[-1]:.3e}"
    )
    print(
        f"Boundary probe grad||B||^2 norm: {near_boundary_diag.grad_norm_sq_norm:.3e} "
        f"vs hover {hover_diag.grad_norm_sq_norm:.3e}"
    )
    for dt, final_error, rms_error, max_force, inside, stage_projections in zip(
        dt_values,
        dt_final_errors,
        dt_rms_errors,
        dt_max_forces,
        dt_inside_flags,
        dt_stage_projection_counts,
    ):
        print(
            f"dt={dt:.4f}s -> final error {final_error:.3e} m, RMS {rms_error:.3e} m, "
            f"max |F| {max_force:.3f} N, inside={inside}, max stage projections={stage_projections}"
        )
    print(f"Saved outputs to {figure_path}")


if __name__ == "__main__":
    main()
