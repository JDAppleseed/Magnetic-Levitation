# Magnetic Levitation Simulation

This repository implements a modular simulation and control stack for a magnetically actuated spherical robot moving inside a 1 m cube. The mathematical source of truth is the local proof set in [`Mathematical-Proof/`](./Mathematical-Proof), with the newest authoritative solution files selected per problem:

- Problem statement: `Mathematical-Proof/magnetic-fields-problem.tex`
- Problem 1: `Mathematical-Proof/Answer-Keys/problem1_solution.tex`
- Problem 2: `Mathematical-Proof/Answer-Keys/problem2_solution_revised.tex`
- Problem 3: `Mathematical-Proof/Answer-Keys/problem3_solution_relocked.tex`
- Problem 4: `Mathematical-Proof/Answer-Keys/problem4_solution.tex`
- Problem 5: `Mathematical-Proof/Answer-Keys/problem5_solution_patched.tex`

The implementation is organized around the proof-backed phase structure:

- `Phase 0`: local proof-file scan, authority map, and docs
- `Phase 1`: RK4 translational dynamics with explicit contact handling
- `Phase 2`: control-affine surrogate magnetic force model and hover feasibility
- `Phase 3`: analytic and finite-difference linearization plus stabilizability checks
- `Phase 4`: two-layer regulation controller with allocation and local validation
- `Phase 5`: quintic `C^2` trajectory generation and tracking validation
- `Phase 6`: physical dipole-based field model, decay-limit analysis, and Earnshaw-style demos
- `Phase 7`: minimal dark-mode PySide6 desktop UI

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Validation

Run the full validation suite:

```bash
python3 -m pytest
```

## Main Demos

Free-fall and backend sanity:

```bash
python scripts/run_hover_demo.py --mode free
```

Static hover:

```bash
python scripts/run_hover_demo.py --mode hover
```

Point-to-point transfer:

```bash
python scripts/run_transfer_demo.py
```

Reachable-force and feasibility sampling:

```bash
python scripts/run_force_map.py
```

Physical-limit demo:

```bash
python scripts/run_physical_limit_demo.py
```

Desktop UI:

```bash
python scripts/run_ui.py
```

The Setup tab includes a `Show 3D view` toggle. Enable it to expose a `3D View` tab beside the existing 2D projections. The preferred backend is `pyqtgraph.opengl` via `PyOpenGL`; if that backend cannot initialize, the UI falls back to a minimal `VisPy` scene. Controls are `left drag` to orbit, `right drag` to pan, and `scroll` to zoom.

## Documentation

- [`docs/architecture.md`](./docs/architecture.md)
- [`docs/math_mapping.md`](./docs/math_mapping.md)
- [`docs/simulator_assumptions.md`](./docs/simulator_assumptions.md)
- [`docs/real_world_bridge.md`](./docs/real_world_bridge.md)
