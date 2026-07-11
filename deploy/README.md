# Deployable policies (biped, legs-only)

Exported policies for `humanoid-control` to load on the robot. Three policies make up the
**stand → walk → squat** sequence, each in its own folder:

| folder | policy | motion | trained from | checkpoint |
|---|---|---|---|---|
| `standup/` | squat → stand | rise to standing | `standup_biped/2026-07-08_23-43-42` | model_700 |
| `walk/` | velocity-tracking walk | omnidirectional walk (spawns from stand) | `biped/2026-07-09_21-33-18` | model_3499 |
| `squat/` | stand → squat | controlled descent to squat | `squat_biped/2026-07-09_17-11-40` | model_500 |

Each folder contains:
- `policy.onnx` (+ `policy.onnx.data`) — the ONNX policy for inference.
- `policy.pt` — TorchScript/JIT version (alternative).
- `policy_latest.yaml` — deploy params: joint order, per-joint PD gains, **default joint positions**, obs/action layout. `policy_checkpoint_path` is repo-relative (`deploy/<name>/policy.onnx`).
- `leg_policy_contract.json` — the sim↔real contract (canonical joint order, gains, limits, obs/action spec).

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
