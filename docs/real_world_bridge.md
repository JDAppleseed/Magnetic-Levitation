# Real-World Bridge

## What is theorem-backed

- Reachable-force feasibility in the affine surrogate
- Local linearization and stabilizability checks
- Two-layer regulation and tracking controller structure
- Pointwise force-feasibility along a `C^2` reference trajectory
- Decay-based loss of hover authority and the Earnshaw-style obstruction in the static conservative regime

## What is an engineering approximation

- The affine face-force surrogate
- The point-dipole face actuator model
- Numerical field gradients and Hessians
- Explicit hard-wall contact projection
- Any damping term used for exploratory numerical conditioning

## What changes for hardware

- Replace the surrogate `G(x)` basis with calibrated coil-to-force or coil-to-field maps
- Model current drivers, voltage limits, and coil thermal limits
- Add field calibration, coil inductance, rate limits, and sensor latency
- Resolve robot magnetic coupling experimentally, including dipole orientation dynamics and torque
- Add state estimation from vision, magnetic sensing, or both

## Actuator realizability issues

- The affine model can request force directions that static source-free fields cannot realize.
- The physical model still omits winding geometry, eddy-current effects, material hysteresis, and thermal saturation.
- Coil current limits and field gradients, not field magnitudes alone, set the usable force envelope.

## Sensing requirements

- Full 3D position at control rate
- Velocity estimate or filtered differentiation
- Optional attitude sensing if torque and dipole orientation are introduced
- Field or current sensing for calibration and fault detection

## Why the abstract surrogate is not a realizable magnetic design

The surrogate directly maps actuator commands to forces. Real magnetic actuation produces fields first, and force arises from spatial gradients plus the robot's coupling law. This means the surrogate is valid for control-design reasoning but not sufficient as a physical hardware claim.

## Hybrid paths around static-field limits

- Time-varying magnetic fields
- Onboard reaction or thrust devices
- Contact-assisted or guide-rail designs
- Diamagnetic or superconducting material regimes outside the baseline static conservative assumptions

