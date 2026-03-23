# Math Mapping

## Authoritative sources selected in Phase 0

Newest local authoritative files by problem:

- Problem statement: `Mathematical-Proof/magnetic-fields-problem.tex`
- Problem 1: `Mathematical-Proof/Answer-Keys/problem1_solution.tex`
- Problem 2: `Mathematical-Proof/Answer-Keys/problem2_solution_revised.tex`
- Problem 3: `Mathematical-Proof/Answer-Keys/problem3_solution_relocked.tex`
- Problem 4: `Mathematical-Proof/Answer-Keys/problem4_solution.tex`
- Problem 5: `Mathematical-Proof/Answer-Keys/problem5_solution_patched.tex`

No newer competing local files were found for Problems 1 and 4. Problems 2, 3, and 5 explicitly use the newer `revised`, `relocked`, and `patched` variants.

## Assumptions carried into code

- The admissible center-of-mass region is `Omega_r = [r, L-r]^3`.
- Gravity acts as `-m g e_z` with `e_z = (0,0,1)^T`.
- Problems 1 through 5 use translation-only dynamics.
- No damping is active unless explicitly configured.
- Actuator limits are explicit box bounds.
- The affine model is for theorem-backed control prototyping; the physical dipole models are labeled exploratory where the proofs no longer guarantee realizability.

## Problem-to-code traceability

### Problem 1: Equilibrium feasibility and reachable force set

- Force matrix `G(x)`: `src/physics/affine_force_model.py::AffineFaceMagneticForceModel.force_matrix`
- Reachable force set diagnostics: `src/analysis/reachable_set.py`
- Hover feasibility and bounded solve: `src/physics/force_feasibility.py::evaluate_force_request`
- Validation: `tests/test_problem1_mapping.py`

### Problem 2: Linearization and stabilizability

- Analytic linearization `A_x`, `B_x`: `src/analysis/linearization.py`
- Finite-difference cross-checks: `src/analysis/linearization.py::finite_difference_linearization`
- Rank and PBH stabilizability diagnostics: `src/control/stabilizability_checks.py`
- Validation: `tests/test_problem2_linearization.py`

### Problem 3: Local regulation and bounded local allocator

- Outer-loop force law `f_cmd = m g e_z - K_p e - K_d e_dot - K_i eta`: `src/control/outer_loop.py`
- Pseudoinverse and bounded least-squares allocators: `src/control/allocator.py`
- Closed-loop PID regulation: `src/control/pid.py`
- Validation: `tests/test_problem3_regulation.py`

### Problem 4: Tracking and pointwise force-feasibility

- Quintic `C^2` reference generation: `src/control/trajectory_generator.py`
- Force-feasibility along a path: `src/physics/force_feasibility.py::evaluate_trajectory_force_feasibility`
- Tracking controller: `src/control/pid.py`
- Validation: `tests/test_problem4_tracking.py`

### Problem 5: Physical limitations

- Decay-threshold computation and sampling: `src/physics/decay_analysis.py`
- Dipole-field potential and Hessian analysis: `src/physics/earnshaw_demos.py`
- Physical magnetic models: `src/physics/dipole_force.py`, `src/physics/induced_dipole_force.py`
- Validation: `tests/test_problem5_limits.py`

## Missing material

No local files were missing from the expected Problem 1 through Problem 5 set. The implementation therefore uses the complete local proof set.

