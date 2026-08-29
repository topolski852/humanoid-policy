# Archived walk policy — smoothness preset A, full profile

Snapshot of the **A-full** walk policy that was live in `deploy/walk/` from 2026-08-26.

| field | value |
|---|---|
| run | `logs/rsl_rl/biped/2026-08-25_02-13-41_smoothA-full` |
| checkpoint | `model_5999.pt` (final of 6000, continuous run) |
| gains | kp=45.0 / kd=1.5 (`HUMANOID_GAIN_PRESET=tuned`) |
| rewards | `HUMANOID_SMOOTH_PRESET=A` — -0.05 / -0.002 / -1e-4 |
| plant | post-flash effort limits, knees 11.0 Nm |
| sim | fwd 0.485 m/s, action_rate_rms 0.357, falls 0.141/min, gait 1.56 Hz |

## ⚠️ Why it was replaced — HARDWARE FAILURE
Run on the robot 2026-08-29: **the data was poor and the encoders faulted across the whole leg.**
That is worse than the single `ERROR_ENCODER_FAULT` seen earlier with the kp45 baseline, which
affected one joint.

This is the important data point: A-full was **measurably smoother than the baseline in sim**
(action_rate_rms 0.357 vs 0.480, joint_vel_rms 1.840 vs 2.145, fall rate 0.141 vs 0.234 /min) and
still made the hardware symptom worse. Sim smoothness is not predicting hardware behaviour — see
`docs/walk-smoothness-sweep.md` sec 10.

Superseded by the preset-B full run (`2026-08-28_00-09-42_smoothB-full-resumed`), which is smoother
again in sim (action_rate_rms 0.316). If B faults the encoders the same way, the sim-side
smoothness lever should be considered exhausted and the cause sought on the hardware side
(encoder mounting, I2C integrity, torque limits, or the sim-to-real contract itself).

Kept for rollback / comparison.
