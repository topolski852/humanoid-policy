# Archived walk policy — actuator-model retrain, per-joint asymmetric gains

Snapshot of the walk policy that was live in `deploy/walk/` up to 2026-08-20.

| field | value |
|---|---|
| run | `logs/rsl_rl/biped/2026-07-18_14-18-40` |
| checkpoint | `model_best.pt` (iter 3087, plateau peak) |
| plant | bench actuator models (stick-slip friction + latency + measured armature), 12.61 kg |
| **gains** | **per-joint asymmetric** — kp 10.5–68.4, kd 0.5–9.8 |

Superseded by the bench-tuned uniform-gain retrain (`logs/rsl_rl/biped/2026-08-18_20-45-17`,
`model_5999.pt`). The asymmetric gains here came from the sim-side contract; bench calibration
on the real motors (humanoid-tuner) found **kp=45 / kd=1.5** uniform gave markedly better
tracking on both motor types, so the successor was trained on those gains instead.

This is the policy that was live during the "robot kicks wildly / won't balance" sim2real
sessions. Kept for rollback / comparison.

Pair `policy_latest.yaml` + `leg_policy_contract.json` with `policy.onnx` when restoring —
the gains and defaults are self-contained here.
