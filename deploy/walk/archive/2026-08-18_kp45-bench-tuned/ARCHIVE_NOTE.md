# Archived walk policy — kp=45 / kd=1.5 (bench-tuned)

Snapshot of the **kp45** walk policy that was live in `deploy/walk/` from
2026-08-20 to 2026-08-24.

| field | value |
|---|---|
| run | `logs/rsl_rl/biped/2026-08-18_20-45-17` |
| checkpoint | `model_5999.pt` (final of 6000; no early stop) |
| **gains** | **kp=45.0 / kd=1.5 uniform** (bench-calibrated, humanoid-tuner 2026-08) |
| plant | StickSlipDelayedPDActuator — measured armature, stick-slip friction, latency |
| training | mean_reward 14.90, mean_episode_length 469.5/500 |

## Why it was replaced
**Jittery on the real robot.** Damping ratio is `zeta = kd / (2*sqrt(kp*J))`. The bench fit
only saw reflected motor inertia (J~0.025), where 45/1.5 gives zeta~0.71. On the assembled
robot each joint also carries its link and everything below it — several times that inertia —
so at J~0.10 these gains fall to zeta~0.35 (underdamped). Combined with the 7-12 ms command
latency, that is the expected jitter mechanism.

Superseded by the kp=20 / kd=2.0 policy trained on Berkeley Humanoid Lite's upstream default
gains (`logs/rsl_rl/biped/2026-08-21_21-20-23_kp20-berkeley`), which sits near zeta~0.71 under
the same loaded-inertia estimate. Same plant, same reward, same 6000-iteration budget — the
runs differ **only** in gains (`HUMANOID_GAIN_PRESET`).

Kept for the head-to-head hardware comparison and for rollback. Pair `policy_latest.yaml` +
`leg_policy_contract.json` with `policy.onnx` when restoring — the gains are self-contained
here, and restoring this policy also means re-writing kp=45/kd=1.5 to the ESCs.
