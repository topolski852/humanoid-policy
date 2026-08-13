# G1 squat ↔ stand — training plan (dev machine)

Picking up `xr_teleoperate` branch `teleop-safety`, `docs/HANDOFF.md` §6, on the GPU box
(`nse-yki-issac-sim`, RTX 5080 16 GB). The handoff splits the work: this machine trains, the WSL
laptop verifies against the robot. Everything below is the training half.

Written 2026-08-10. Sections marked **[unverified]** have not been run.

---

## 1. Two things the handoff could not have known

### Isaac Gym is not an option on this GPU

The handoff says to clone `~/unitree_rl_gym` and `~/isaacgym` here as well. Isaac Gym Preview 4 is
built against CUDA 11 / PyTorch 1.x and has no `sm_120` (Blackwell) kernels, so it cannot run on an
RTX 5080. Neither repo is cloned on this machine, and cloning them will not help. *(A compatibility
statement, not a measurement — no install was attempted.)*

That loss is smaller than it looks. What unitree_rl_gym contributed was: a G1 env (12 leg DoF, flat
velocity tracking — the handoff already rejected it as the wrong task), a pre-trained walk policy,
and `deploy_real.py` publishing to `rt/lowcmd`. Only the last one still matters, and it matters as a
*template for the deploy side*, not as a training dependency.

**Use this repo instead.** `humanoid-policy` already runs Isaac Sim 6.0 + Isaac Lab 3.0b2 +
rsl_rl 5.0.1 on torch 2.11/cu128 (see `pyproject.toml`), which is Blackwell-capable, and it already
contains a **squat ↔ stand task pair for the Berkeley biped** — `Standup-Humanoid-Policy-Biped-v0`
and `Squat-Humanoid-Policy-Biped-v0` — with a pose library, terminal-pose reward, and a contact
schedule. The G1 task is a port into that structure, not a new stack.

Isaac Lab also ships the robot: `isaaclab_assets.robots.unitree.G1_29DOF_CFG` (and
`G1_INSPIRE_FTP_CFG`, the same 29-DoF G1 with Inspire hands the robot actually wears).

### The captured safety squat rides — and in two places exceeds — the joint limits

Checked `teleop/safety/poses/safety_squat.yaml` against the joint ranges in
`xr_teleoperate/assets/g1/g1_body29_hand14.xml` (the model `robot_arm_ik.py` uses):

| joint | captured | hard limit | headroom [rad] | |
| :-- | --: | --: | --: | :-- |
| `left_knee` | 2.8988 | 2.8798 | **−0.019** | past the limit |
| `right_knee` | 2.9107 | 2.8798 | **−0.031** | past the limit |
| `left_hip_pitch` | −2.5260 | −2.5307 | 0.0047 | on the limit |
| `right_hip_pitch` | −2.5279 | −2.5307 | 0.0028 | on the limit |
| `left_ankle_pitch` | −0.8709 | −0.8727 | 0.0018 | on the limit |
| `right_ankle_pitch` | −0.8690 | −0.8727 | 0.0037 | on the limit |
| `waist_pitch` | 0.4986 | 0.5200 | 0.0214 | on the limit |

Also outside Isaac's *soft* limits (`soft_joint_pos_limit_factor = 0.9`): both hip pitches, both
ankle pitches, `waist_pitch`, `left_wrist_pitch`, `right_wrist_yaw` — nine joints in total sit
outside the soft band, seven of them in the legs.

This is what a maximal hand-posed fold looks like, and it is not a problem with the pose. It is a
problem for three specific things in training, all of which the scaffolding already handles:

1. **An unreachable target caps the reward.** PhysX clamps the PD target to the joint limit, so a
   knee goal of 2.899 can never be reached and `track_joint_pose_exp` never returns 1 on knees.
   → the importer clamps into the limits with a 0.01 rad margin and prints exactly what it changed.
2. **`dof_pos_limits` would fight the objective.** With the stock 0.9 soft factor, the goal pose
   sits outside the soft band on seven leg/waist joints, so the penalty saturates *at the target*.
   → `G1_SQUAT_CFG` sets `soft_joint_pos_limit_factor = 1.0`.
3. **Sim2real margin is ~0.** The encoders read past the URDF limit, so either the URDF is
   conservative by ~2° or the zero calibration differs. Either way a policy that parks joints on
   the limit is one calibration drift away from a different pose on hardware. Worth a laptop-side
   check: read the robot's own limits over the SDK and compare against the URDF.

---

## 2. What is scaffolded

All in `humanoid-policy`. It imports, registers, and constructs **[verified]**; nothing has been
trained or stepped through physics **[unverified]**.

| file | what it is |
| :-- | :-- |
| `source/humanoid_policy_assets/.../robots/g1.py` | `G1_SQUAT_CFG` — `G1_29DOF_CFG` + contact sensors + the SDK deploy gains + soft-limit fix. Joint groups in `rt/lowcmd` index order. |
| `scripts/rsl_rl/import_g1_pose.py` | capture → pose library: deg→rad, `_joint` suffix, clamp-into-limits with a report. Runs today, no Isaac needed. |
| `.../standup/mdp/arm_trajectory.py` | `ArmTrajectoryCommand` — scripted arm motion, emits `[phase, phase_rate]` as the command. |
| `.../standup/mdp/rewards.py` | `+ scheduled_contact` — contact penalty gated on base height (the shin release). |
| `.../standup/config/g1/env_cfg.py` | `Standup-Humanoid-Policy-G1-v0` — squat → stand. |
| `.../standup/config/g1/squat_env_cfg.py` | `Squat-Humanoid-Policy-G1-v0` — stand → squat, the safe-stop descent. |
| `.../standup/config/g1/agents/rsl_rl_ppo_cfg.py` | PPO, identical to the biped standup baseline. |
| `scripts/rsl_rl/variants.py` | `--variant standup-g1` / `squat-g1`. |

`configs/poses.yaml` gained a `g1_squat` entry (additive; the biped's `squat`/`stand` are
untouched). Its base pose is a **placeholder** — see §4.

### The design, and why

**13-DoF action space** — 12 legs + `waist_pitch`, ordered to match `rt/lowcmd` indices 0–11 and 14,
with `preserve_order=True`. This is the handoff's central scoping result: the hands-down contact is
incidental, the arms carry no load, so they leave the action space and self-collision leaves the
reward.

**Arms as a phase-observed disturbance.** `ArmTrajectoryCommand` interpolates all 14 arm joints
along a smoothstep from the squat configuration to the stand configuration and writes their position
targets directly; the `JointPositionAction` term only touches the 13 policy joints, so the two never
contend. The command vector is `[phase, phase_rate]`, and `phase_rate` (peaked mid-motion) is the
part that tells the policy *when the shove arrives* — phase alone conveys that only implicitly. The
trajectory is deterministic joint-space interpolation, so the deploy side can replay it open-loop.

**Contact schedule via `scheduled_contact`.** The squat rests on feet *and* shins (`*_knee_link`).
A flat `undesired_contacts` on the shins would punish the start pose; ignoring them lets the policy
drag its shins up. So shin contact is free below a release height (a third of the way up the rise)
and charged above it.

**Terminal pose, not velocity.** `track_joint_pose_exp` toward the stand pose, per-joint normalized
by the squat→stand travel — the biped's term, unchanged. Plus a pelvis-height term.

**Gains that match deployment.** Legs Kp 60/60/60/100/40/40, Kd 1/1/1/2/1/1; waist 60/40/40; arms
Kp 40 Kd 1 (handoff §4). The stock Isaac config uses Kp 100–200 on the legs and 3000 on the arms;
training at 3000 would hide the arm dynamics the legs are supposed to reject.

**`flat_orientation_l2` is a hazard here.** The deep fold pitches the pelvis hard, so an
upright-pelvis penalty opposes the pose itself. It is reduced to −1.0 for the rise (where upright
*is* the destination) and **zeroed** for the descent. If the rise stalls, this is the first weight
to look at.

---

## 3. What is deliberately *not* in the env

**No self-collision reward, and `enabled_self_collisions` stays `False`.** Per handoff §5, the only
contact worth preventing is hand-vs-knee; calf-on-thigh and arm-on-chest are wanted. Hand contact is
a fixed-geometry property of the arm trajectory, so it is an offline check
(`xr_teleoperate/tools/check_self_collision.py --from/--to`) run once before training, not a term
paid for on every step of every env.

### Status: endpoint cleared, and the path needed a waypoint

The check now runs on this machine (`~/.venvs/xr-tools`, mujoco 3.11). Its self-test passes — the
forced arm-into-torso probe reports 25 contacts — so a clean result means something.

**Endpoint.** Reproduced the handoff's finding exactly: `left_hand_thumb_1/2` ↔ `left_knee`, 2.7 mm
and 1.0 mm. Applied **−2.0°** to `left_shoulder_pitch` rather than the handoff's −1.0°: measured
against a 20 mm detection band, −1.0° clears by only 0.80 mm, which is under the 2.7 mm
mesh-approximation scale the handoff itself flags. −2.0° clears by 4.22 mm at the same cost. The
right hand's 9.4 mm clearance is the ceiling, so there is nothing to gain past ≈−3.5°. Applied to
both `safety_squat.yaml` and its radians twin `safety_squat.json`, and re-imported into `g1_squat`.

**Path — this is the one that mattered.** Both endpoints being clean says nothing about the motion
between them. A straight joint-space line from the stand pose to the corrected squat puts **both
hands through the thighs and knees**: 22 of 61 samples in collision, up to 50 mm penetration,
`*_hand_thumb_0`/`middle_0`/`index_0` against `*_hip_yaw_link` and `*_knee_link` over t = 0.63–0.98.
The first version of `ArmTrajectoryCommand` was exactly that straight line.

Simple fixes do not work — legs-first, arms-first, and a pure roll abduction all still collide
(10–25 samples). A parameter sweep found the shape that does: **hold the arms until the legs are
into the fold, then move them while the shoulders swing out and back.** With
`shoulder_pitch −60°`, `shoulder_roll ±20°` as a half-sine waypoint (zero at both ends, so the
endpoints are untouched), the tightest point on the entire 240-sample path is the *endpoint* — the
same 4.22 mm. This is now `_ARM_WAYPOINT` in `env_cfg.py` and `waypoint_offset` on the command term.

**Caveat that matters:** this was validated against the *placeholder* stand pose. The waypoint
numbers are tuned to an endpoint that will change. **Re-run the path check once the real `g1_stand`
is captured** — the mechanism will still be right, the constants may not be.

Remaining non-hand contacts in the pose (`--all`) are the wanted ones: thigh-on-torso,
forearm-on-knee, calf-on-thigh. Per §5 of the handoff only hand contacts are worth preventing.

---

## 4. Order of work

Blocking items first — the two poses are prerequisites for everything.

1. ~~**Clear the hand/knee contact.**~~ **Done** — see §3. `left_shoulder_pitch −2.0°`, and a
   waypoint on the arm trajectory to clear the *path*, which was the larger problem.
2. **Floor-snap `g1_squat`.** The importer writes joint angles but a placeholder base
   (`z = 0.30`, identity rotation) — a joint capture has no base pose. Until it is snapped, the
   robot spawns hovering or intersecting the floor and every reset is a drop.
   `pose_editor.py --variant standup-g1 --pose g1_squat --viz kit`, place, snap, save.
   *(The editor is written against the biped; a G1 variant may need small fixes. **[unverified]**)*
3. **Author `g1_stand`.** Until it exists the env falls back to the G1's nominal init pose
   (hip −0.10, knee 0.30, ankle −0.20) and a guessed 0.75 m pelvis height, which makes the
   height reward meaningless. This should be the FSM-4 stand as the robot actually holds it —
   with the caveat from handoff §3 that FSM 4 leaves the ankles unpowered, so capture the joint
   angles, not the behaviour.
4. **Re-validate the arm trajectory** against the real `g1_stand` —
   `check_self_collision.py --from <stand>.yaml --to teleop/safety/poses/safety_squat.yaml`.
   Done once already against the placeholder stand (§3); the waypoint constants are tuned to that
   placeholder and must be re-checked, and re-swept if they no longer clear.
5. **Smoke-run the descent first**, not the rise: `--variant squat-g1`, small `--num_envs`, few
   iterations. The descent is the safety-critical direction (a safe stop descends before
   de-energizing, handoff §7) and it starts from the easy, well-conditioned pose. If the spawn is
   wrong, this is where it shows up cheapest.
6. **Then the rise**, `--variant standup-g1`.

### Tuning levers, in the order they are likely to be needed

1. `flat_orientation_l2` on the rise (−1.0) — the pose/orientation conflict described above.
2. `base_height_bonus` on the descent — the biped needed weight 2.0 / std 0.10 to stop settling for
   a shallow crouch; the G1's fold is deeper, so expect to need at least that.
3. `_SHIN_RELEASE_HEIGHT` — currently a third of the way up, and a guess.
4. `decimation` — set to 4 (50 Hz) rather than the biped's 8 (25 Hz). This must match whatever the
   deploy runner streams to `rt/lowcmd`; if the laptop side settles on 25 Hz, change it here.

---

## 5. Cross-machine dependencies

The laptop owns the robot, so these cannot be resolved here:

- **The real `g1_stand` joint angles** — capture with `tools/capture_pose.py` from an FSM-4 stand.
- **The robot's own joint limits** over the SDK, versus the URDF (see §1).
- **The deploy loop rate**, which fixes `decimation`.
- **Any hardware validation of a safety action** — still unvalidated per handoff §7, and a mujoco
  pass cannot supply it (unitree_mujoco bridges low-level DDS only, no `LocoClient` services).

Nothing in the training scaffolding depends on the safety framework's runtime; the two meet only at
the exported policy and the pose.
