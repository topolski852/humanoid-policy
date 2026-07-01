# humanoid-policy

RL policy-training repo for the humanoid robot, extracted from Berkeley Humanoid Lite and
kept in sync with the runtime control code in
[`humanoid-control`](https://github.com/topolski852/humanoid-control).

This repo is **training only** — hardware interface, firmware, teleop, and sim2real
deployment code live elsewhere. It trains policies in Isaac Lab (rsl_rl / PPO), exports them
to ONNX/JIT, and emits a deployment contract that `humanoid-control` loads.

## Training variants

Training type is selected with a `--variant` flag:

| variant | task | robot | notes |
|---|---|---|---|
| `standup-biped` | squat → stand | legs (12 DoF) | starts from the control repo's squat pose |
| `walk-biped` | velocity tracking | legs (12 DoF) | matches the current legs-only control contract |
| `walk-humanoid` | velocity tracking | full body (22 DoF) | arms included (future) |
| `standup-humanoid` | squat → stand | full body (22 DoF) | future |

```bash
# train — full run (tuned 16k-env setup)
python scripts/rsl_rl/train.py --variant standup-biped --profile full --headless

# train — fast dev run (4k envs, sub-3h)
python scripts/rsl_rl/train.py --variant standup-biped --profile fast --headless

# play / export policy + deployment contract
python scripts/rsl_rl/play.py --variant standup-biped
```

`--profile` sets the training scale (`full` = 16384 envs, `fast` = 4096 envs targeted sub-3h);
explicit `--num_envs` / `--max_iterations` override it. See `scripts/rsl_rl/profiles.py`
(tune `FAST_MAX_ITERATIONS` to hit your wall-clock budget after checking the reported sec/iter).

## Sim ↔ real contract

The biped observation/action layout matches `humanoid-control`'s
`configs/leg_policy_params.json` exactly: 45-dim observation
(`command(3) · base_ang_vel(3) · projected_gravity(3) · (joint_pos−default)(12) · joint_vel(12) · prev_action(12)`),
12-dim action with `action_scale = 0.25`, 25 Hz policy, canonical left→right joint order
`[hip_roll, hip_yaw, hip_pitch, knee_pitch, ankle_pitch, ankle_roll]`. Per-joint PD gains and
position limits in sim are set to the ESC device-truth values from that contract.

## Layout

```
scripts/rsl_rl/      train.py, play.py, cli_args.py, variants.py
source/humanoid_policy/           Isaac Lab tasks (locomotion/velocity, locomotion/standup)
source/humanoid_policy_assets/    robot model: USD / URDF / meshes
```

## Attribution

Derived from [Berkeley Humanoid Lite](https://github.com/HybridRobotics/Berkeley-Humanoid-Lite)
(MIT). The physical robot model retains its original `berkeley_humanoid_lite` identifiers.
