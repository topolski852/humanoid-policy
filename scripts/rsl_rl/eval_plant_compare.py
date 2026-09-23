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

    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = get_checkpoint_path(
        log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint,
        preferred_checkpoint=("model_best.pt" if args_cli.checkpoint is None else None),
    )
    print(f"[eval] plant={args_cli.plant}  checkpoint={resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
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
    global _KNEE_IDX
    _names = robot.data.joint_names
    _KNEE_IDX = [_names.index("leg_left_knee_pitch_joint"), _names.index("leg_right_knee_pitch_joint")]
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

    obs = env.get_observations()
    total = args_cli.warmup + args_cli.steps
    with torch.inference_mode():
        policy(obs)  # warm up torch.compile / lazy init
        for i in range(total):
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
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
                n_steps += 1
                if _KNEE_TRAJ is not None:
                    _KNEE_TRAJ.append(robot.data.joint_pos.torch[:, _KNEE_IDX].clone().cpu())
            prev_action = actions

    denom = max(n_steps * N, 1)
    steps_denom = max(n_steps - 1, 1) * N  # action_rate has one fewer sample
    resets = float((falls + timeouts).item())
    env_seconds = n_steps * N * dt
    metrics = {
        "plant": args_cli.plant,
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
