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
                n_steps += 1
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
