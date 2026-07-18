# Archived walk policy — Eureka g2c3, full profile

Snapshot of the walk policy that was live in `deploy/walk/` up to 2026-07-18.

| field | value |
|---|---|
| run | `logs/rsl_rl/biped/2026-07-17_12-28-34` |
| checkpoint | `model_best.pt` (iter 4593, plateau peak) |
| reward | Eureka **g2c3** (fitness 0.6926) on the Berkeley-Humanoid-Lite base |
| profile | full (16k+ envs) |
| **plant** | **friction-free implicit actuators, old robot mass 11.34 kg** |

Superseded by the actuator-model retrain (bench-validated stick-slip friction + latency +
measured armature, corrected 12.61 kg mass; branch `actuator-model-integration`). On the
bench-modeled plant the successor cut the fall rate ~45% vs this policy — this one was the
friction-free-trained baseline that destabilized on the real robot.

Kept for rollback / comparison. Pair `policy_latest.yaml` + `leg_policy_contract.json` with
`policy.onnx` when restoring — the defaults/gains are self-contained here.
