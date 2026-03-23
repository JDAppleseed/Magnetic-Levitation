# Architecture

## Source of truth

The local proof files in `Mathematical-Proof/` define the mathematical obligations. The simulation does not introduce hidden stabilizers or silent force overrides. All control decisions flow through:

1. an outer-loop force command,
2. an allocator that maps the force request into actuator commands,
3. an explicit force model,
4. a translational ODE backend.

## Core modules

- `src/sim/`: state representation, contact handling, and the RK4 backend
- `src/physics/`: affine surrogate force model, dipole-based magnetic field model, feasibility analysis, decay analysis, and Earnshaw diagnostics
- `src/control/`: force-space outer loops, actuator allocation, PID regulation, trajectory generation, and stabilizability checks
- `src/analysis/`: linearization utilities, reachable-set summaries, domain sampling, logging, and export helpers
- `src/ui/`: minimal PySide6 desktop front end with orthographic projections and live diagnostics

## Backend boundary

`RK4Backend` is the source of truth for Problems 1 through 5. It integrates:

`m x_ddot = F_mag(x,u) - m g e_z + F_contact + F_damping`

with explicit, configurable contact projection and optional explicit damping. The force model is injected; the backend is agnostic to whether it is affine or physical.

## Control boundary

The proof-backed control structure is always:

`state -> desired force f_cmd -> allocator -> u -> realized force`

The UI, scripts, and tests all report the requested force, achieved force, residual, and actuator saturation so the simulation never hides force infeasibility.

