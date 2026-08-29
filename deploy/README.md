# Deployable policies (biped, legs-only)

Exported policies for `humanoid-control` to load on the robot. Three policies make up the
**stand → walk → squat** sequence, each in its own folder:

| folder | policy | motion | trained from | checkpoint |
|---|---|---|---|---|
| `standup/` | squat → stand | rise to standing | `standup_biped/2026-07-08_23-43-42` | model_700 |
| `walk/` | velocity-tracking walk | omnidirectional walk (spawns from stand) | `biped/2026-08-28_00-09-42_smoothB-full-resumed` | model_5999 (final of 6000) |
| `squat/` | stand → squat | controlled descent to squat | `squat_biped/2026-07-09_17-11-40` | model_500 |

Each folder contains:
- `policy.onnx` (+ `policy.onnx.data`) — the ONNX policy for inference.
- `policy.pt` — TorchScript/JIT version (alternative).
- `policy_latest.yaml` — deploy params: joint order, per-joint PD gains, **default joint positions**, obs/action layout. `policy_checkpoint_path` is repo-relative (`deploy/<name>/policy.onnx`).
- `leg_policy_contract.json` — the sim↔real contract (canonical joint order, gains, limits, obs/action spec).

## Archiving previous policies (`<folder>/archive/`)
Each deploy folder keeps the **current** policy at its top level and previous ones under
`<folder>/archive/<YYYY-MM-DD_short-name>/`. Before replacing a live policy, snapshot the five
current files into a new dated subfolder there (with a short `ARCHIVE_NOTE.md` giving the run,
checkpoint, and what changed) so any deployed policy can be rolled back or compared.

Archived so far:
- `walk/archive/2026-08-25_smoothA-full/` — the **smoothness preset A** full-profile policy.
  Live 2026-08-26; replaced after it **faulted the encoders across the whole leg** on the robot
  despite being measurably smoother than the baseline in sim.
- `walk/archive/2026-08-21_kp20-berkeley/` — the **kp=20 / kd=2.0** Berkeley-default policy.
  Live 2026-08-24; the gain A/B failed on hardware (humanoid-control `0e74ba1`). Predates the
  effort-limit flash, so its contract still has 6.0 Nm knees.
- `walk/archive/2026-08-18_kp45-bench-tuned/` — the **kp=45 / kd=1.5** bench-tuned policy
  (run 2026-08-18_20-45-17). Live 2026-08-20 to 2026-08-24; replaced because it was jittery
  on the real robot (underdamped once real link inertia is accounted for).
- `walk/archive/2026-07-18_actuator-model-asymmetric-gains/` — the actuator-model retrain
  with per-joint asymmetric gains (kp 10.5-68.4 / kd 0.5-9.8); superseded by the bench-tuned
  uniform kp=45 / kd=1.5 retrain.
- `walk/archive/2026-07-17_eureka-g2c3-fullprofile/` — the friction-free-trained Eureka g2c3
  walk policy (old 11.34 kg mass); the pre-actuator-model baseline.

## ⚠️ Each policy has its own `default_joint_positions` — pair policy + contract on every switch
All three share the same 45-dim obs / 12-dim action layout, but the **action offset and observation
reference differ per policy**:
- action: `target = action × action_scale + default_joint_positions`
- observation: uses `joint_pos − default_joint_positions`

The defaults are **not the same** across policies:
- `standup` default = the **squat** pose (it starts in the squat).
- `walk` and `squat` defaults = the **stand** pose (they start standing).

So when the sequencer switches policies, it **must** switch to that policy's `default_joint_positions`
(and gains) from the matching folder. Using the wrong default = the robot snaps to the wrong
reference on switch.

## Sequence / handoff
`standup` (squat→stand) → `walk` (spawns from stand, so it takes over cleanly) → `squat` (stand→squat).
Handoff points are the **stand** pose. The walk policy was trained spawning from the stand pose for a
clean standup→walk handoff; for walk→squat, bring the robot to a settled stand before switching (the
squat policy expects to start from ~stand).

## Notes
- Legs-only (12 DoF, no arms). `squat` is the least-trained (500 iters) — validate it carefully.
- Regenerate any policy by re-running `scripts/rsl_rl/play.py --variant <standup|walk|squat>-biped
  --load_run <run> --checkpoint <model>.pt` and copying `exported/` + the two `configs/` files here.
