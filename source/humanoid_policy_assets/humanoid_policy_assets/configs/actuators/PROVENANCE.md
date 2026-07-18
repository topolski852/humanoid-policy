# Actuator model provenance

These two JSONs are the **validated predictive motor+gearbox models** produced by the
bench system-ID in the sibling repo `humanoid-tuner`, vendored here so walk-policy
training is self-contained and reproducible.

| file | motor | joint group | source (tuner) |
|---|---|---|---|
| `m6c12_pitch.json`  | MAD M6C12, 150 KV | legs: hip_roll, hip_yaw, hip_pitch, knee_pitch | `sim/fitted_models/m6c12_pitch.json` |
| `mad5010_roll.json` | MAD 5010, 200 KV  | ankles: ankle_pitch, ankle_roll                | `sim/fitted_models/mad5010_roll.json` |

**Tuner commit:** `bd7c613` ("5010 geared cross-check + hardware benchmark of sim-tuned gains").

Model semantics are defined by `humanoid-tuner/sim/motor_model.py` (`MotorModel` +
`FrictionModel` stick-slip) and applied to Isaac exactly as in
`humanoid-tuner/sim/isaac/substrate.py` (`effort = tau_ctrl - FrictionModel.torque(vel,
drive=tau_ctrl, load)`). The torch port lives in
`humanoid_policy_assets/actuators/stickslip_actuator.py`.

## What we take from these JSONs, and what we deliberately override

- **inertia** (`inertia`) -> Isaac `armature` (reflected motor+gearbox inertia; URDF link
  masses provide the load inertia on top). legs 0.025, ankles 0.01367.
- **friction** (`coulomb`, `breakaway`, `stribeck_vel`, `viscous`, `stick_vel`) -> the
  stick-slip `FrictionModel`. Applied per-joint within the group.

Overridden on purpose (do **not** read these two fields from the JSON):

- **`latency_s` (JSON reads 0)** — the gentle fit steps did not excite transport delay.
  We use the **measured** command->response latency from the tuner's
  `sim/isaac/motor.py` MotorSpec: **7.2 ms (M6C12 / legs)**, **12 ms (5010 / ankles)**.
- **`torque_limit` (JSON reads 3.0)** — this is a **bench safety cap**, not a real motor
  limit. Training keeps the real per-joint effort limits already in the walk cfg
  (`_CONTRACT_EFFORT`) and the per-joint contract PD gains. The 3.0 is ignored.

The policy<->robot contract (45-dim obs, 12 actions, action scale, contract PD gains) is
**unchanged** — these models only alter the sim *plant* (inertia, friction, latency).

## Robot mass correction (same sim2real fidelity pass)

The measured robot is **27.8 lb = 12.61 kg** (two legs + torso, no arms). The CAD-derived
biped asset totalled only **11.34 kg (25.0 lb)** — ~10% light.

Cross-check against the bench-measured assembled actuator weights (motor + 15:1 gearbox +
ESC): M6C12 **0.680 kg** ×8 hip/knee joints + MAD5010 **0.425 kg** ×4 ankle joints =
**7.14 kg (57% of the robot)**. The sim's 12 leg links total 8.18 kg, i.e. those 7.14 kg of
motors + ~1.0 kg of 3D-printed structure — so **the motor mass is already represented
correctly** in the leg links (the per-joint URDF distribution mounts each motor on the link
it bolts to, so only the leg-group total is meaningful; it checks out). The ankle group is
~75 g light — negligible.

The entire **1.27 kg deficit is torso/structure** (battery, compute, wiring — absent from
CAD). Fix: base/torso link mass **3.16 -> 4.43 kg**, diagonal + product inertia scaled
x1.4016 (added mass assumed distributed like the torso; CoM kept — if the battery is mounted
low, lowering base CoM z would further help walking stability). Applied in BOTH:
  - `usd/humanoid_biped.usd` — root-layer override on /humanoid/base (Isaac loads this).
  - `urdf/humanoid_biped.urdf` — source of truth kept consistent.
The full-humanoid asset and the shared physics config are untouched.
