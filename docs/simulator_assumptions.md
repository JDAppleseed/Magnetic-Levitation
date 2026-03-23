# Simulator Assumptions

## Global assumptions

- Units are SI throughout.
- The default cube interior side length is `L = 1.0 m`.
- The ball state is the center-of-mass position and velocity only.
- Contact is explicit and configurable. The default contact handler clips the center back into `Omega_r` and removes the outward velocity component with configurable restitution.
- Damping is disabled by default. If enabled, it is linear viscous damping reported as an explicit force term.

## Affine magnetic surrogate

- Each face actuator contributes a force basis vector aligned with its inward face normal.
- Magnitude decays with distance from the corresponding face plane using a configurable inverse-power law.
- This model is intentionally control-oriented and may realize force fields not compatible with static source-free magnetostatics.

## Physical dipole layer

- Each face actuator is modeled as a point dipole placed at the center of a cube face and oriented along the inward normal.
- The field is quasi-static and computed from superposition of dipole fields.
- Fixed-dipole and induced-dipole couplings are supported through numerical gradients of scalar potentials.
- The default physical coefficients are effective calibrated demo values so the desktop UI can realize visible motion on a 1 m domain. They should be interpreted as control-oriented stand-ins, not hardware-ready coil parameters.
- This layer is useful for exposing physical limitations, not as a final hardware-fidelity solver.

## Numerical assumptions

- The RK4 integrator uses zero-order-hold actuator inputs over each timestep.
- Controller integral state is updated explicitly between integration steps.
- Physical-model force gradients and Hessians are evaluated with centered finite differences.
