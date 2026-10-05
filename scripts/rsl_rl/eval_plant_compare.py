"""Cross-plant robustness eval: replay ONE trained walk policy under a chosen actuator plant.

Motivation: the deployed walk policy was trained on the friction-free implicit plant and
"freaks out" on the real robot. This harness replays that SAME checkpoint headless under
either plant and reports quantitative stability/tracking metrics, so we can see whether the
bench-validated modeled plant (armature + stick-slip friction + latency) reproduces the
on-robot instability — validating the actuator models and giving a baseline the retrain
must beat.

Runs ONE plant per process (the plant toggle is read once at import). Use ``--plant
baseline`` vs ``--plant modeled`` in two runs and diff the JSON. ``eval_plant_compare.sh``
(or just running this twice) does both. Nothing is trained or exported.

Metrics (aggregated over all envs × measured steps, after a warmup window):
  * forward_speed_mean  — base-frame x velocity (m/s); tracking of the fixed forward command.
  * base_accel_rms      — RMS horizontal base linear accel (m/s²); the IMU "freak out" signal.
  * rocking_rms         — RMS roll/pitch base angular velocity (rad/s).
  * joint_vel_rms       — RMS leg joint velocity (rad/s); the on-robot thrash was ~12 rad/s.
  * action_rate_rms     — RMS step-to-step change in raw action; chatter/runaway signal.
  * fall_rate_per_min   — non-timeout terminations per env-minute (higher = less stable).
  * mean_episode_len_s  — mean time between resets (s); shorter = falling more.
"""

from __future__ import annotations

import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Cross-plant robustness eval for a walk policy.")
parser.add_argument("--plant", choices=["baseline", "modeled"], required=True,
                    help="baseline = friction-free implicit; modeled = bench actuator models.")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=1000, help="measured policy steps (after warmup).")
parser.add_argument("--warmup", type=int, default=50, help="discarded settle steps after boot.")
parser.add_argument("--cmd_vx", type=float, default=0.3, help="fixed forward command (m/s).")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--task", type=str, default=None, help="gym task id (usually set via --variant).")
parser.add_argument("--out", type=str, default=None, help="write metrics JSON here.")
# Plant-identification overrides. Replaying a FROZEN network on a perturbed plant asks: which plant
# change makes this policy behave in sim the way it behaves on the robot? Hardware runs the same
# network, so a perturbation that reproduces the hardware signature is a candidate sim2real
# mismatch. None of these change training; they apply to this eval only.
parser.add_argument("--foot_friction", type=float, default=None,
                    help="fix robot-body static+dynamic friction (terrain is 1.0, combine=multiply, "
                         "so this IS the foot-ground coefficient). Default: training DR (0.4-1.2).")
parser.add_argument("--foot_dyn_friction", type=float, default=None,
                    help="fix robot-body DYNAMIC friction separately (with --foot_friction as static). "
                         "Models a floor whose kinetic friction sits below its static friction.")
parser.add_argument("--floor_stiffness", type=float, default=None,
                    help="PhysX compliant-contact spring stiffness on the ground plane (N/m per contact). "
                         "Models a soft floor such as a rug; sim's default ground is rigid.")
parser.add_argument("--floor_damping", type=float, default=None,
                    help="PhysX compliant-contact damping on the ground plane (N*s/m); needs --floor_stiffness.")
parser.add_argument("--sag_effort_scale", type=float, default=None,
                    help="scale the DELIVERED torque cap on hip_pitch/knee_pitch only. Models a joint that "
                         "delivers less torque under gait than commanded. Stalls are still scored against "
                         "the contract cap, the way the hardware reconstruction scores them.")
parser.add_argument("--sag_viscous_scale", type=float, default=None,
                    help="scale stick-slip viscous friction on hip_pitch and knee_pitch only.")
parser.add_argument("--base_force_x", type=float, default=None,
                    help="constant BODY-frame x force on the base, N (negative = holding the robot back). "
                         "Models the operator's supporting hand, which every hardware walk has and sim does not.")
parser.add_argument("--base_force_z", type=float, default=None,
                    help="constant BODY-frame z force on the base, N (positive = lifting / carrying weight).")
parser.add_argument("--sag_armature_scale", type=float, default=None,
                    help="scale armature (reflected inertia) on hip_pitch and knee_pitch only.")
parser.add_argument("--touchdown", action="store_true",
                    help="record feet + ankles at the PHYSICS rate and report foot-strike metrics: landing "
                         "speed, foot tilt and rotation rate just before contact, and the ankle_pitch velocity "
                         "peak that follows. The hardware ankle spikes (24-29 rad/s) are impact transients a "
                         "few ms long, which the 40 ms policy-rate sample can miss.")
# --variant / --task and rsl-rl args (--load_run / --checkpoint) come from the shared helpers.
import sys
sys.path.insert(0, os.path.dirname(__file__))
import variants  # noqa: E402
import cli_args  # noqa: E402

variants.add_variant_arg(parser)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.task is None and getattr(args_cli, "variant", None) is None:
    args_cli.variant = "walk-biped"
variants.resolve_variant(args_cli)

# Select the actuator plant BEFORE any task/robot-cfg import reads the toggle.
os.environ["HUMANOID_ACTUATOR_MODEL"] = "1" if args_cli.plant == "modeled" else "0"

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402

from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg  # noqa: E402
from importlib.metadata import version as _pkg_version  # noqa: E402

import humanoid_policy.tasks  # noqa: F401,E402
from humanoid_policy_assets.actuators.limits import joint_effort_limits  # noqa: E402


# EVAL_GAIT=0 / "false" mean OFF; a bare truthiness check enabled them for any non-empty value.
_KNEE_TRAJ = [] if os.environ.get("EVAL_GAIT", "").strip().lower() not in ("", "0", "false") else None
_KNEE_IDX = None
_FOOT_Z = [] if _KNEE_TRAJ is not None else None
_FOOT_IDX = None


def _stall_and_posture(tau_hist, jvel_hist, tau_limit, height_hist, pitch_hist, names, dt):
    """Stall fraction defined IDENTICALLY to humanoid-control scripts/measure/gait_contact.py
    (REPORT_2026-09-29_contact_B_C.md): a stall is the SAME hip_pitch or knee_pitch joint at
    >= 0.98 * cap for >= 0.3 s, or both knees under 0.5 rad/s for >= 0.3 s. Runs are found per
    joint and then combined -- OR-ing joints tick by tick first chains normal left/right stance
    saturation into fake multi-second stalls. Hardware: measC-full 17% of walking, smoothA 36%.
    """
    import numpy as _np
    out = {}
    tau = torch.stack(tau_hist).numpy()                 # (T, N, J)
    lim = tau_limit[0].cpu().numpy()                    # (J,)
    jv = torch.stack(jvel_hist).numpy()                 # (T, N, J)
    T, N, _ = tau.shape
    min_run = max(1, int(round(0.3 / dt)))
    sag = [i for i, n in enumerate(names) if n.endswith("hip_pitch_joint") or n.endswith("knee_pitch_joint")]
    knees = [i for i, n in enumerate(names) if n.endswith("knee_pitch_joint")]

    def runs(mask_1d):                                  # bool (T,) -> covered bool (T,) of runs >= min_run
        cov = _np.zeros_like(mask_1d)
        edges = _np.diff(_np.concatenate(([0], mask_1d.astype(_np.int8), [0])))
        for a, b in zip(_np.flatnonzero(edges == 1), _np.flatnonzero(edges == -1)):
            if b - a >= min_run:
                cov[a:b] = True
        return cov

    covered, n_stalls = 0, 0
    per_joint = {names[j]: 0 for j in sag}
    for e in range(N):
        union = _np.zeros(T, dtype=bool)
        for j in sag:
            r = runs(tau[:, e, j] >= 0.98 * lim[j])
            per_joint[names[j]] += int(r.sum())
            union |= r
        union |= runs(_np.all(jv[:, e, :][:, knees] < 0.5, axis=1))
        covered += int(union.sum())
        n_stalls += int(_np.sum(_np.diff(_np.concatenate(([0], union.astype(_np.int8)))) == 1))
    out["stall_fraction"] = covered / float(T * N)
    out["stalls_per_min"] = n_stalls / (T * N * dt / 60.0)
    out["stall_time_frac_per_joint"] = {k: v / float(T * N) for k, v in per_joint.items()}
    if height_hist:
        out["base_height_mean_m"] = float(torch.stack(height_hist).mean().item())
    if pitch_hist:
        p = torch.stack(pitch_hist) * 180.0 / 3.141592653589793
        out["torso_pitch_mean_deg"] = float(p.mean().item())
        out["torso_pitch_p95_deg"] = float(torch.quantile(p.flatten(), 0.95).item())
    return out


def _touchdown_metrics(rec, dt_phys, warm_substeps):
    """Foot-strike statistics from physics-rate records (see --touchdown).

    A touchdown is a foot whose contact force rises past 5 N after >= 20 ms below 1 N. For each one:
      pre-impact (the last substep before contact): foot vertical speed, horizontal speed, |angular
      velocity|, and tilt -- the angle between the foot's gravity vector and its median STANCE
      gravity vector, so it is independent of the link-frame convention.
      post-impact (the 100 ms after): peak |ankle_pitch velocity| on that leg, and peak contact force.
    Spearman correlations of the ankle peak against each pre-impact variable show which one an
    impact penalty should target.
    """
    import numpy as _np
    F = torch.stack(rec["force"]).numpy()[warm_substeps:]          # (S, N, 2)  |contact force|
    vz = torch.stack(rec["vz"]).numpy()[warm_substeps:]            # (S, N, 2)  foot vertical speed
    vxy = torch.stack(rec["vxy"]).numpy()[warm_substeps:]          # (S, N, 2)
    w = torch.stack(rec["w"]).numpy()[warm_substeps:]              # (S, N, 2)  |foot ang vel|
    g = torch.stack(rec["g"]).numpy()[warm_substeps:]              # (S, N, 2, 3) gravity in foot frame
    av = torch.stack(rec["ank"]).numpy()[warm_substeps:]           # (S, N, 2)  |ankle_pitch vel|
    S, N, _ = F.shape
    air_need = max(1, int(round(0.020 / dt_phys)))
    win = max(1, int(round(0.100 / dt_phys)))
    # stance reference orientation per foot: median gravity vector while loaded and still
    still = (F > 20.0) & (w < 0.5)
    ref = []
    for s in range(2):
        gs = g[:, :, s][still[:, :, s]]
        r = _np.median(gs, axis=0) if len(gs) else _np.array([0.0, 0.0, -1.0])
        ref.append(r / (_np.linalg.norm(r) + 1e-9))
    rows = []                                                      # (vz, vxy, w, tilt_deg, ank_peak, f_peak)
    for e in range(N):
        for s in range(2):
            f = F[:, e, s]
            on = f > 5.0
            off = f < 1.0
            for k in _np.flatnonzero(on[1:] & ~on[:-1]) + 1:
                if k < air_need or k + win > S or not off[k - air_need:k].all():
                    continue
                gp = g[k - 1, e, s] / (_np.linalg.norm(g[k - 1, e, s]) + 1e-9)
                tilt = float(_np.degrees(_np.arccos(_np.clip(gp @ ref[s], -1.0, 1.0))))
                rows.append((-vz[k - 1, e, s], vxy[k - 1, e, s], w[k - 1, e, s], tilt,
                             av[k:k + win, e, s].max(), f[k:k + win].max()))
    out = {"touchdown_events": len(rows)}
    a = _np.abs(av).reshape(-1)
    out["ankle_pitch_vel_physics_p99_rad_s"] = float(_np.percentile(a, 99))
    out["ankle_pitch_vel_physics_max_rad_s"] = float(a.max())
    if not rows:
        return out
    R = _np.array(rows)
    names = ["landing_vz_m_s", "landing_vxy_m_s", "landing_foot_angvel_rad_s", "landing_tilt_deg",
             "post_ankle_peak_rad_s", "post_force_peak_n"]
    for i, n in enumerate(names):
        out[f"td_{n}_p50"] = float(_np.median(R[:, i]))
        out[f"td_{n}_p95"] = float(_np.percentile(R[:, i], 95))

    def _rank(x):
        o = _np.argsort(x)
        r = _np.empty(len(x))
        r[o] = _np.arange(len(x))
        return r
    ry = _rank(R[:, 4])
    out["td_spearman_vs_ankle_peak"] = {
        n: float(_np.corrcoef(_rank(R[:, i]), ry)[0, 1]) for i, n in enumerate(names) if i != 4}
    out["td_frac_ankle_peak_over_13"] = float((R[:, 4] > 13.0).mean())
    return out


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed

    # Fixed forward command so both plants get identical, visible commands (same as play.py).
    r = env_cfg.commands.base_velocity.ranges
    r.lin_vel_x = (args_cli.cmd_vx, args_cli.cmd_vx)
    r.lin_vel_y = (0.0, 0.0)
    r.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.commands.base_velocity.heading_command = False

    overrides = {}
    if args_cli.foot_friction is not None:
        f = float(args_cli.foot_friction)
        p = env_cfg.events.physics_material.params
        p["static_friction_range"] = (f, f)
        p["dynamic_friction_range"] = (f, f)
        overrides["foot_friction"] = f
    if args_cli.foot_dyn_friction is not None:
        fd = float(args_cli.foot_dyn_friction)
        env_cfg.events.physics_material.params["dynamic_friction_range"] = (fd, fd)
        overrides["foot_dyn_friction"] = fd
    if args_cli.floor_stiffness is not None:
        mat = env_cfg.scene.terrain.physics_material
        mat.compliant_contact_stiffness = float(args_cli.floor_stiffness)
        mat.compliant_contact_damping = float(args_cli.floor_damping or 0.0)
        overrides["floor_stiffness"] = mat.compliant_contact_stiffness
        overrides["floor_damping"] = mat.compliant_contact_damping
    # hip_pitch / knee_pitch are the joints whose hardware torque runs 2-3x sim; hip_roll / hip_yaw
    # share the same M6C12 actuator model and match. Scale only the sagittal pair so the eval can
    # tell a sagittal-specific plant error from an actuator-wide one.
    sag = ("leg_.*_hip_pitch_joint", "leg_.*_knee_pitch_joint")
    rest = ("leg_.*_hip_roll_joint", "leg_.*_hip_yaw_joint")
    legs = env_cfg.scene.robot.actuators["legs"]
    for arg, field in (("sag_viscous_scale", "viscous"), ("sag_armature_scale", "armature")):
        k = getattr(args_cli, arg)
        if k is None:
            continue
        base = getattr(legs, field)
        if isinstance(base, dict):
            raise RuntimeError(f"{field} is already per-joint; override expects a group scalar")
        setattr(legs, field, {**{j: base * k for j in sag}, **{j: base for j in rest}})
        overrides[arg] = k
    if args_cli.base_force_x is not None or args_cli.base_force_z is not None:
        # The reset-time random push would wipe a permanent wrench on every reset; drop it so the
        # hand force is the only external force in play. Re-applied every step below as well.
        env_cfg.events.base_external_force_torque = None
        overrides["base_force_x"] = args_cli.base_force_x or 0.0
        overrides["base_force_z"] = args_cli.base_force_z or 0.0
    if args_cli.sag_effort_scale is not None:
        k = float(args_cli.sag_effort_scale)
        eff = legs.effort_limit
        if not isinstance(eff, dict):
            raise RuntimeError("expected per-joint effort_limit dict on the legs group")
        legs.effort_limit = {j: (v * k if ("hip_pitch" in j or "knee_pitch" in j) else v) for j, v in eff.items()}
        overrides["sag_effort_scale"] = k
    if overrides:
        print(f"[eval] PLANT OVERRIDES (eval only): {overrides}")

    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = get_checkpoint_path(
        log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint,
        preferred_checkpoint=("model_best.pt" if args_cli.checkpoint is None else None),
    )
    print(f"[eval] plant={args_cli.plant}  checkpoint={resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if overrides:  # prove the override reached the simulated actuators, not just the cfg
        _r = env.unwrapped.scene["robot"]
        _a = _r.actuators["legs"]
        _names = [_r.data.joint_names[i] for i in (_a.joint_indices.tolist() if hasattr(_a.joint_indices, "tolist") else range(len(_r.data.joint_names)))]
        print("[eval] legs viscous :", {n: round(float(v), 4) for n, v in zip(_names, _a.viscous[0])})
        print("[eval] legs armature:", {n: round(float(v), 4) for n, v in zip(_names, _a.armature[0])})
    env = RslRlVecEnvWrapper(env)

    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, _pkg_version("rsl-rl-lib"))
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    # checkpoints were saved from torch.compile-wrapped actor/critic (keys prefixed `_orig_mod.`),
    # so compile before load to match — same as scripts/rsl_rl/play.py.
    runner.alg.actor = torch.compile(runner.alg.actor, mode="default")
    runner.alg.critic = torch.compile(runner.alg.critic, mode="default")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    uenv = env.unwrapped
    robot = uenv.scene["robot"]
    global _KNEE_IDX, _FOOT_IDX
    _names = robot.data.joint_names
    _KNEE_IDX = [_names.index("leg_left_knee_pitch_joint"), _names.index("leg_right_knee_pitch_joint")]
    _bn = robot.data.body_names
    _FOOT_IDX = [next(i for i, b in enumerate(_bn) if side in b and b.endswith("ankle_roll"))
                 for side in ("left", "right")]
    dt = float(uenv.step_dt)  # policy step (s)
    dev = uenv.device
    N = uenv.num_envs

    # accumulators (sums over measured steps × envs)
    n_steps = 0
    fwd_sum = torch.zeros((), device=dev)
    accel_sq = torch.zeros((), device=dev)
    rock_sq = torch.zeros((), device=dev)
    jvel_sq = torch.zeros((), device=dev)
    actrate_sq = torch.zeros((), device=dev)
    falls = torch.zeros((), device=dev)
    timeouts = torch.zeros((), device=dev)
    prev_action = None

    # Torque saturation, PER JOINT so it lines up with the hardware table in humanoid-control
    # docs/measurements/REPORT_2026-09-23_smoothA.md sec 2 (knees 52.1% / 33.3% of ticks at cap,
    # |tau| p95 28.0 Nm). Everything else in this file reduces to a scalar; these deliberately
    # do not, because "which joint runs out of headroom" is the whole question.
    #
    # Measured against computed_torque (the pre-clip PD demand), NOT applied_torque: on the
    # modeled plant StickSlipDelayedPDActuator rewrites applied_effort as clipped_PD - friction,
    # so applied_torque there is a net figure that conflates saturation with friction.
    tau_limit = joint_effort_limits(robot)          # (N, J), from the actuator groups
    n_joints = robot.data.joint_pos.shape[1]
    sat_count = torch.zeros(n_joints, device=dev)   # joint-steps at/over the cap
    tau_peak = torch.zeros(n_joints, device=dev)
    tau_hist = []                                   # |tau| per measured step, for the p95

    # The two hardware discriminators, so a bundle can be screened in sim before it costs a robot
    # session. Both come from humanoid-control docs/measurements/REPORT_2026-09-25_measA.md:
    #   tilt rate  separated smoothA (0.4 deg/s) from measA (20.6) where tilt MAGNITUDE did not,
    #              and matched the operator's ranking. Hardware target p95 < 1.0 deg/s.
    #   joint vel  is a hard ESC limit -- encoder fault near 13 rad/s. Target p99 < 1.5 rad/s.
    # Buffered rather than accumulated because both targets are tail statistics, not means, and
    # an rms hides exactly the excursions that matter.
    tilt_hist = []                                  # |omega_xy| per measured step (rad/s)
    yaw_hist = []                                   # |omega_z| per measured step (rad/s)
    height_hist, pitch_hist = [], []                # base height (m) and torso pitch (rad) per step
    jvel_hist = []                                  # |joint_vel| per measured step (rad/s)

    _hand = None
    if "base_force_x" in overrides:
        _base_id = [robot.data.body_names.index("base")]
        _f = torch.zeros((N, 1, 3), device=dev)
        _f[:, 0, 0] = overrides["base_force_x"]
        _f[:, 0, 2] = overrides["base_force_z"]
        _hand = (_f, torch.zeros_like(_f), _base_id)

    def _apply_hand():
        if _hand is not None:
            robot.permanent_wrench_composer.reset()
            robot.permanent_wrench_composer.add_forces_and_torques(
                _hand[0], _hand[1], body_ids=_hand[2], is_global=False)

    _td = None
    if args_cli.touchdown:
        # Wrap scene.update, which env.step calls once per PHYSICS substep (PhysX decimation runs
        # in Python), so feet and ankles are sampled every 5 ms rather than every 40 ms.
        from isaaclab.utils.math import quat_apply_inverse
        _cs = uenv.scene.sensors["contact_forces"]
        _cs_feet = [next(i for i, b in enumerate(_cs.body_names) if side in b and b.endswith("ankle_roll"))
                    for side in ("left", "right")]
        _ank_idx = [robot.data.joint_names.index(f"leg_{s}_ankle_pitch_joint") for s in ("left", "right")]
        _gdown = torch.tensor([0.0, 0.0, -1.0], device=dev).repeat(N * 2, 1)
        _td = {"on": False, "calls": 0, "force": [], "vz": [], "vxy": [], "w": [], "g": [], "ank": []}
        _orig_update = uenv.scene.update

        def _rec_update(dt):
            _orig_update(dt)
            if not _td["on"]:
                return
            d = robot.data
            lv = d.body_lin_vel_w.torch[:, _FOOT_IDX, :]
            q = d.body_quat_w.torch[:, _FOOT_IDX, :].reshape(-1, 4)
            _td["force"].append(_cs.data.net_forces_w.torch[:, _cs_feet, :].norm(dim=-1).cpu())
            _td["vz"].append(lv[..., 2].cpu())
            _td["vxy"].append(lv[..., :2].norm(dim=-1).cpu())
            _td["w"].append(d.body_ang_vel_w.torch[:, _FOOT_IDX, :].norm(dim=-1).cpu())
            _td["g"].append(quat_apply_inverse(q, _gdown).reshape(N, 2, 3).cpu())
            _td["ank"].append(d.joint_vel.torch[:, _ank_idx].abs().cpu())
            _td["calls"] += 1

        uenv.scene.update = _rec_update

    _apply_hand()
    obs = env.get_observations()
    total = args_cli.warmup + args_cli.steps
    with torch.inference_mode():
        policy(obs)  # warm up torch.compile / lazy init
        for i in range(total):
            if _td is not None and i == args_cli.warmup:
                _td["on"] = True
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
            _apply_hand()
            measuring = i >= args_cli.warmup
            if measuring:
                data = robot.data
                fwd = data.root_lin_vel_b.torch[:, 0]
                acc = data.body_lin_acc_w.torch[:, 0, :2]
                rock = data.root_ang_vel_b.torch[:, :2]
                jvel = data.joint_vel.torch
                fwd_sum += fwd.sum()
                accel_sq += acc.square().sum()
                rock_sq += rock.square().sum()
                jvel_sq += jvel.square().mean(dim=1).sum()   # per-env mean over joints, summed over envs
                if prev_action is not None:
                    actrate_sq += (actions - prev_action).square().mean(dim=1).sum()
                # terminations this step (falls = non-timeout dones)
                tm = uenv.termination_manager
                dones = tm.dones
                touts = tm.time_outs
                falls += (dones & ~touts).sum()
                timeouts += touts.sum()
                tau_abs = data.computed_torque.torch.abs()
                sat_count += (tau_abs >= 0.98 * tau_limit).sum(dim=0).float()
                tau_peak = torch.maximum(tau_peak, tau_abs.amax(dim=0))
                tau_hist.append(tau_abs.cpu())
                tilt_hist.append(rock.norm(dim=1).cpu())   # |omega_xy| per env
                yaw_hist.append(data.root_ang_vel_b.torch[:, 2].abs().cpu())
                height_hist.append(data.root_pos_w.torch[:, 2].cpu())
                # forward pitch from the base-frame gravity vector: +ve = nose down, as the IMU reports
                pitch_hist.append(torch.asin(torch.clamp(-data.projected_gravity_b.torch[:, 0], -1, 1)).cpu())
                jvel_hist.append(jvel.abs().cpu())
                n_steps += 1
                if _KNEE_TRAJ is not None:
                    _KNEE_TRAJ.append(robot.data.joint_pos.torch[:, _KNEE_IDX].clone().cpu())
                    _FOOT_Z.append(robot.data.body_pos_w.torch[:, _FOOT_IDX, 2].clone().cpu())
            prev_action = actions

    denom = max(n_steps * N, 1)
    steps_denom = max(n_steps - 1, 1) * N  # action_rate has one fewer sample
    resets = float((falls + timeouts).item())
    env_seconds = n_steps * N * dt
    metrics = {
        "plant": args_cli.plant,
        "plant_overrides": overrides,
        "checkpoint": os.path.basename(resume_path),
        "num_envs": N,
        "measured_steps": n_steps,
        "cmd_vx": args_cli.cmd_vx,
        "policy_dt_s": dt,
        "forward_speed_mean": float(fwd_sum.item() / denom),
        "base_accel_rms": float((accel_sq.item() / denom) ** 0.5),
        "rocking_rms": float((rock_sq.item() / denom) ** 0.5),
        "joint_vel_rms": float((jvel_sq.item() / denom) ** 0.5),
        "action_rate_rms": float((actrate_sq.item() / steps_denom) ** 0.5),
        "falls": float(falls.item()),
        "timeouts": float(timeouts.item()),
        "fall_rate_per_min": float(falls.item() / env_seconds * 60.0),
        "mean_episode_len_s": float(env_seconds / resets) if resets > 0 else float("inf"),
    }

    # Per-joint torque headroom. Compare torque_sat_frac directly against the hardware
    # saturation fractions; if sim shows ~52% at the knees too, the reward is the lever. If sim
    # shows a few percent while hardware shows 52%, the sim plant is too easy and no reward
    # change will close that gap.
    tau_all = torch.cat(tau_hist, dim=0) if tau_hist else torch.zeros((1, n_joints))
    joint_steps = max(n_steps * N, 1)
    metrics["torque_sat_frac"] = {
        name: float(sat_count[j].item() / joint_steps)
        for j, name in enumerate(robot.data.joint_names)
    }
    metrics["torque_p95_nm"] = {
        name: float(torch.quantile(tau_all[:, j], 0.95).item())
        for j, name in enumerate(robot.data.joint_names)
    }
    metrics["torque_peak_nm"] = {
        name: float(tau_peak[j].item()) for j, name in enumerate(robot.data.joint_names)
    }
    metrics["torque_limit_nm"] = {
        name: float(tau_limit[0, j].item()) for j, name in enumerate(robot.data.joint_names)
    }

    # Hardware-comparable balance + safety tails. Compare directly against the hardware table in
    # REPORT_2026-09-25_measA.md; these are the numbers that ranked the bundles correctly when
    # fall_rate/min did not (sim ranked measA best on falls and it is the worst on the robot).
    if tilt_hist:
        tilt = torch.cat(tilt_hist) * 180.0 / 3.141592653589793   # rad/s -> deg/s
        metrics["tilt_rate_p95_deg_s"] = float(torch.quantile(tilt, 0.95).item())
        metrics["tilt_rate_max_deg_s"] = float(tilt.max().item())
    if yaw_hist:
        # Body-frame yaw rate, comparable to the IMU gyro-z the heading report measured while
        # walking: |omega_z| p95 ~1.4-2.5 rad/s on hardware (humanoid-control
        # REPORT_2026-09-29_heading.md section 2) -- a 12-31 deg torso twist at the gait frequency.
        yaw = torch.cat(yaw_hist)
        metrics["yaw_rate_p95_rad_s"] = float(torch.quantile(yaw, 0.95).item())
        metrics["yaw_rate_p50_rad_s"] = float(torch.quantile(yaw, 0.50).item())
    if jvel_hist:
        jv = torch.cat(jvel_hist)                                 # (steps*envs, J)
        flat = jv.flatten()
        metrics["joint_vel_p99_rad_s"] = float(torch.quantile(flat, 0.99).item())
        metrics["joint_vel_max_rad_s"] = float(flat.max().item())
        metrics["joint_vel_p99_per_joint"] = {
            name: float(torch.quantile(jv[:, j], 0.99).item())
            for j, name in enumerate(robot.data.joint_names)
        }
        # Fraction of joint-steps past the reward hinge (2.0) and the termination ceiling (10.0).
        metrics["joint_vel_over_2_frac"] = float((flat > 2.0).float().mean().item())
        metrics["joint_vel_over_10_frac"] = float((flat > 10.0).float().mean().item())
    if _KNEE_TRAJ:
        import numpy as _np
        tr = torch.stack(_KNEE_TRAJ).numpy()          # (T, N, 2) left/right knee
        f = _np.fft.rfftfreq(tr.shape[0], dt)
        hzs, cors = [], []
        for e in range(tr.shape[1]):
            l, r = tr[:, e, 0], tr[:, e, 1]
            if _np.ptp(l) < 0.02:
                continue
            P = _np.abs(_np.fft.rfft(l - l.mean())) ** 2
            hzs.append(float(f[1:][_np.argmax(P[1:])]))
            lc, rc = l - l.mean(), r - r.mean()
            cors.append(float(lc @ rc / (_np.linalg.norm(lc) * _np.linalg.norm(rc) + 1e-9)))
        metrics["gait_hz_median"] = float(_np.median(hzs)) if hzs else 0.0
        metrics["knee_corr_median"] = float(_np.median(cors)) if cors else 0.0
        metrics["knee_swing_median"] = float(_np.median(_np.ptp(tr, axis=0)))
        metrics["envs_stepping_pct"] = 100.0 * len(hzs) / tr.shape[1]
    try:
        _stall_lim = tau_limit
        if "sag_effort_scale" in overrides:
            _stall_lim = tau_limit.clone()
            for _j, _n in enumerate(robot.data.joint_names):
                if "hip_pitch" in _n or "knee_pitch" in _n:
                    _stall_lim[:, _j] = tau_limit[:, _j] / overrides["sag_effort_scale"]
        metrics.update(_stall_and_posture(tau_hist, jvel_hist, _stall_lim, height_hist, pitch_hist,
                                          robot.data.joint_names, dt))
    except Exception as exc:  # a metric bug must never cost a pipeline its eval
        metrics["stall_metrics_error"] = repr(exc)
    if _FOOT_Z:
        # Swing-foot clearance, defined IDENTICALLY to humanoid-control ROBOT_PC_BRIEF item C so the
        # hardware FK number and this one are directly comparable: d = z_left - z_right at the
        # *_ankle_roll link origins; a step is the span between sign changes of d; its clearance is
        # max|d| over that span (swing foot's peak height above the stance foot); spans < 0.15 s
        # are crossing jitter and are dropped. The 10th percentile is the scuffing signal.
        import numpy as _np
        fz = torch.stack(_FOOT_Z).numpy()             # (T, N, 2)
        min_len = max(1, int(round(0.15 / dt)))
        clr = []
        for e in range(fz.shape[1]):
            d = fz[:, e, 0] - fz[:, e, 1]
            cross = _np.flatnonzero(_np.diff(_np.signbit(d)))  # index before each sign change
            for a, b in zip(cross[:-1], cross[1:]):
                if b - a >= min_len:
                    clr.append(float(_np.abs(d[a + 1:b + 1]).max()))
        if clr:
            metrics["swing_clearance_median_m"] = float(_np.median(clr))
            metrics["swing_clearance_p10_m"] = float(_np.percentile(clr, 10))
            metrics["swing_clearance_steps"] = len(clr)
    if _td is not None:
        metrics["touchdown_substeps_recorded"] = _td["calls"]   # expect steps * decimation
        try:
            metrics.update(_touchdown_metrics(_td, float(uenv.physics_dt), 0))
        except Exception as exc:  # a metric bug must never cost a pipeline its eval
            metrics["touchdown_metrics_error"] = repr(exc)
    print("[eval] RESULT " + json.dumps(metrics))
    if args_cli.out:
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
        with open(args_cli.out, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[eval] wrote {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
