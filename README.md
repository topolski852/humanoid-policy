# humanoid-policy

RL policy-training repo for the humanoid robot, extracted from Berkeley Humanoid Lite and
kept in sync with the runtime control code in
[`humanoid-control`](https://github.com/topolski852/humanoid-control).

This repo is **training only** — hardware interface, firmware, teleop, and sim2real
deployment code live elsewhere. It trains policies in Isaac Lab (rsl_rl / PPO), exports them
to ONNX/JIT, and emits a deployment contract that `humanoid-control` loads.

## Setup

Self-contained via [`uv`](https://docs.astral.sh/uv/) — no external Isaac/Berkeley install needed.
From the repo root:

```bash
uv sync                       # creates .venv with Isaac Sim / Isaac Lab / torch (pinned via uv.lock)
export OMNI_KIT_ACCEPT_EULA=YES   # one-time: accept the Omniverse Kit EULA for headless runs
```

Then run scripts with the repo-local interpreter, e.g. `.venv/bin/python scripts/rsl_rl/train.py …`
(or `uv run python scripts/rsl_rl/train.py …`). See [COMMANDS.md](COMMANDS.md) for full recipes.

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
`configs/leg_policy_params.json`: 45-dim observation
(`command(3) · base_ang_vel(3) · projected_gravity(3) · (joint_pos−default)(12) · joint_vel(12) · prev_action(12)`),
12-dim action with `action_scale = 0.25`, 25 Hz policy (`policy_dt = 0.04`), canonical left→right
joint order `[hip_roll, hip_yaw, hip_pitch, knee_pitch, ankle_pitch, ankle_roll]`.

### ⚠️ Joint SIGN / frame convention — the robot runs in the *device* frame, not the URDF frame

This is the one place the current `walk` export does **not** match the robot, and it must be fixed
before a deploy will drive the legs correctly. The control code reads joint positions from, and
sends targets to, the **ESC/device frame** (the studio `humanoid_lite.json` config). The ESC gear
sign already un-mirrors the right leg, so **both legs use the same numeric sign convention** — a
symmetric stance is the *same* sign on left and right, not opposite.

The sim/URDF is left↔right mirror-symmetric, so its exported contract mirrors the right leg. Verified
against the device (position limits are frame-authoritative), **exactly these joints are sign-flipped**
and must be **negated** so the export comes out in the device frame:

| joint | device frame (correct) | current sim/URDF export | action |
|---|---|---|---|
| `right_hip_roll` | limits `[-0.175, +1.571]`, default same sign as left | `[-1.571, +0.175]` (mirrored) | **negate** |
| `right_hip_yaw`  | limits `[-0.983, +0.590]`, default same sign as left | `[-0.589, +0.982]` (mirrored) | **negate** |
| `right_ankle_roll` | limits symmetric `±0.262` | also URDF-mirrored | **negate** (see note) |

All six left joints and the right pitch joints (`hip_pitch`, `knee_pitch`, `ankle_pitch`) already match —
do **not** touch them. After the fix, the exported `leg_policy_contract.json` right-leg limits and
`default_joint_positions` must read the same sign as the left leg (e.g. `right_hip_roll` default `> 0`,
matching `left_hip_roll`, not `< 0`).

> Note on `right_ankle_roll`: its limits are symmetric so the sign can't be proven from limits, and its
> physical encoder is broken on the robot (mirror the left value for pose). Negate it for consistency
> with the URDF mirror, but it can't be validated on hardware yet.

**Full device-frame position limits** (source of truth — set the sim joint limits to match):

| joint | lower | upper |  | joint | lower | upper |
|---|---|---|---|---|---|---|
| left_hip_roll | -0.175 | +1.571 | | right_hip_roll | -0.175 | +1.571 |
| left_hip_yaw | -0.983 | +0.590 | | right_hip_yaw | -0.983 | +0.590 |
| left_hip_pitch | -1.899 | +0.983 | | right_hip_pitch | -1.899 | +0.983 |
| left_knee_pitch | 0.000 | +2.443 | | right_knee_pitch | 0.000 | +2.443 |
| left_ankle_pitch | -0.785 | +0.785 | | right_ankle_pitch | -0.785 | +0.785 |
| left_ankle_roll | -0.262 | +0.262 | | right_ankle_roll | -0.262 | +0.262 |

### PD gains

The `walk` policy was trained (and deployed) with **uniform `kp = 20`, `kd = 2`** on all 12 joints
(the Berkeley default). The robot's ESCs have been set to match, so train with the same flat gains for
now. This is NOT the robot's per-joint tuned set — a later gain-matched retrain should use these
device gains instead (`humanoid_lite.json` / `leg_policy_params.json`):

`hip_roll 20/4 · hip_yaw(L 10.5/0.5, R 20/1) · hip_pitch 68.4/9.8 · knee(L 27/2.45, R 30/1.22) ·
ankle_pitch(L 18/2, R 20/0.5) · ankle_roll(L 23.3/4, R 20/2)`  (kp/kd).

## Layout

```
scripts/rsl_rl/      train.py, play.py, cli_args.py, variants.py
source/humanoid_policy/           Isaac Lab tasks (locomotion/velocity, locomotion/standup)
source/humanoid_policy_assets/    robot model: USD / URDF / meshes
```

## Attribution

The robot model is loosely derived from
[Berkeley Humanoid Lite](https://github.com/HybridRobotics/Berkeley-Humanoid-Lite) (MIT); the
original copyright notice is retained in the source headers per the license. Identifiers in this
repo have since been renamed to the generic `humanoid` / `HUMANOID_*` scheme.
