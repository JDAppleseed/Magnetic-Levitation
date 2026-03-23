from __future__ import annotations

import numpy as np

from analysis.runtime import build_backend
from control.pid import TwoLayerController
from physics.zero_force_model import ZeroMagneticForceModel
from sim.state import RigidBodyState


def test_phase1_free_fall_sanity(system):
    free_backend = build_backend(system, enable_contact=False)
    zero_model = ZeroMagneticForceModel()
    state0 = RigidBodyState(np.array([0.5, 0.5, 0.5]), np.zeros(3), 0.0)
    log = free_backend.simulate(
        state0,
        zero_model,
        duration=1.0,
        dt=system.backend.dt,
        control_input_fn=lambda _t, _s: np.zeros(6, dtype=float),
    )
    accelerations = np.asarray(log.accelerations)
    relative_error = abs(np.mean(accelerations[:, 2]) + system.cube.gravity) / system.cube.gravity
    assert relative_error < 0.01
    assert np.allclose(accelerations[:, :2], 0.0, atol=1.0e-12)


def test_problem3_regulation_converges(affine_model, backend, system, controller_config, sim_defaults):
    controller = TwoLayerController(
        affine_model,
        system.ball.mass,
        system.cube.gravity,
        controller_config.gains,
        allocator_mode=controller_config.allocator_mode,
        integral_limit=controller_config.integral_limit,
    )
    target = sim_defaults.hover_point
    state0 = RigidBodyState(target + np.array([0.05, 0.0, 0.0]), np.zeros(3), 0.0)
    log = backend.simulate(
        state0,
        affine_model,
        duration=5.0,
        dt=system.backend.dt,
        controller=controller,
        target_provider=lambda _t: target,
    )
    errors = np.linalg.norm(np.asarray(log.tracking_errors), axis=1)
    assert np.min(errors) < 0.01
    assert errors[-1] < 0.01
