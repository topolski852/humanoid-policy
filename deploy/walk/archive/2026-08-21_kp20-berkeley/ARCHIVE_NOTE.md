# Archived walk policy — kp=20 / kd=2.0 (Berkeley defaults)

Snapshot of the **kp20** walk policy that was live in `deploy/walk/` from 2026-08-24.

| field | value |
|---|---|
| run | `logs/rsl_rl/biped/2026-08-21_21-20-23_kp20-berkeley` |
| checkpoint | `model_5999.pt` (final of 6000) |
| **gains** | **kp=20.0 / kd=2.0 uniform** (`HUMANOID_GAIN_PRESET=berkeley`) |
| plant | StickSlipDelayedPDActuator; effort limits **pre-flash** (knees 6.0 Nm) |
| training | mean_reward 12.41, mean_episode_length 423.4/500 |

## Why it was replaced
**The kp20 A/B failed on hardware** (see humanoid-control commit `0e74ba1`, which reverted the
robot-side bundle to kp45). It was trained to test whether Berkeley's upstream default gains
would cure the jitter seen with the bench-tuned kp=45 / kd=1.5 policy; on the robot it did not.

Superseded by the smoothness-penalty retrain (`2026-08-25_02-13-41_smoothA-full`), which keeps
kp=45 / kd=1.5 and instead attacks smoothness through the reward, on the post-flash plant
(knees 11.0 Nm).

Note this bundle predates the effort-limit flash: its `effort_limits` still read 6.0 Nm for the
knees. Restoring it means restoring **both** kp=20/kd=2.0 to the ESCs and accepting the older
torque caps in the contract.

Kept for rollback / comparison.
