from __future__ import annotations

import numpy as np

from control.pid import TwoLayerController
from control.trajectory_generator import QuinticTrajectory
from physics.force_feasibility import evaluate_trajectory_force_feasibility
from sim.state import RigidBodyState


def test_problem4_trajectory_force_feasibility(affine_model, system, sim_defaults):
    trajectory = QuinticTrajectory(sim_defaults.start, sim_defaults.target, sim_defaults.trajectory_duration)
    summary = evaluate_trajectory_force_feasibility(
        affine_model,
        trajectory,
        system.ball.mass,
        system.cube.gravity,
        sample_count=101,
        mode="bounded_ls",
    )
    assert summary.all_feasible
    assert summary.max_residual_norm < 1.0e-6


def test_problem4_tracking_controller(affine_model, backend, system, controller_config, sim_defaults):
    controller = TwoLayerController(
        affine_model,
        system.ball.mass,
        system.cube.gravity,
        controller_config.gains,
        allocator_mode=controller_config.allocator_mode,
        integral_limit=controller_config.integral_limit,
    )
    trajectory = QuinticTrajectory(sim_defaults.start, sim_defaults.target, sim_defaults.trajectory_duration)
    state0 = RigidBodyState(sim_defaults.start.copy(), np.zeros(3), 0.0)
    log = backend.simulate(
        state0,
        affine_model,
        duration=sim_defaults.tracking_duration,
        dt=system.backend.dt,
        controller=controller,
        target_provider=trajectory.evaluate,
    )
    final_error = np.linalg.norm(log.tracking_errors[-1])
    assert log.rms_tracking_error() < 0.02
    assert final_error < 0.01

