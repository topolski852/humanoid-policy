"""Smoothness eval for a TD-MPC2 checkpoint — the "payoff measurement" vs the reactive PPO policy.

A fork of scripts/rsl_rl/eval_plant_compare.py: identical plant toggle, fixed-command setup, and
metric accumulators (forward_speed, base_accel_rms, rocking_rms, joint_vel_rms, action_rate_rms,
fall_rate, mean_episode_len) emitting the SAME JSON schema — so results diff directly against the
PPO baseline's eval_plant_compare output. The only change is the policy: a TD-MPC2 agent run with
the MPPI planner (`--plan`) or the bare policy prior (default).

Success target: lower action_rate_rms / joint_vel_rms (the "over-reaction" signals) than the PPO
policy at comparable forward_speed_mean and fall_rate_per_min, on `--plant modeled`.

Action-rate is measured on the ENV-raw action (agent [-1,1] × act_env_scale), matching the units
the PPO eval used, so the two are comparable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rsl_rl"))
import variants  # noqa: E402

parser = argparse.ArgumentParser(description="TD-MPC2 smoothness eval vs the reactive baseline.")
parser.add_argument("--plant", choices=["baseline", "modeled"], default="modeled",
                    help="baseline = friction-free implicit; modeled = bench actuator models.")
parser.add_argument("--checkpoint", type=str, required=True,
                    help="path to a TD-MPC2 .pt (or a run dir -> uses model_best.pt).")
parser.add_argument("--plan", action="store_true", help="use the MPPI planner (else the policy prior).")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=1000, help="measured policy steps (after warmup).")
parser.add_argument("--warmup", type=int, default=50)
parser.add_argument("--cmd_vx", type=float, default=0.3, help="fixed forward command (m/s).")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--out", type=str, default=None, help="write metrics JSON here.")
variants.add_variant_arg(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.task is None and getattr(args_cli, "variant", None) is None:
    args_cli.variant = "walk-biped"
variants.resolve_variant(args_cli)
os.environ["HUMANOID_ACTUATOR_MODEL"] = "1" if args_cli.plant == "modeled" else "0"

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402

from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402
import humanoid_policy.tasks  # noqa: F401,E402

from humanoid_policy.tdmpc.agent import TDMPC2  # noqa: E402
from env_adapter import TdmpcVecEnv  # noqa: E402


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    r = env_cfg.commands.base_velocity.ranges
    r.lin_vel_x = (args_cli.cmd_vx, args_cli.cmd_vx)
    r.lin_vel_y = (0.0, 0.0)
    r.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.commands.base_velocity.heading_command = False

    agent_cfg = load_cfg_from_registry(args_cli.task, "tdmpc_cfg_entry_point")
    agent_cfg.num_envs = args_cli.num_envs
    device = args_cli.device or env_cfg.sim.device

    ckpt = args_cli.checkpoint
    if os.path.isdir(ckpt):
        ckpt = os.path.join(ckpt, "model_best.pt")
    print(f"[eval-smooth] plant={args_cli.plant} planner={'mppi' if args_cli.plan else 'prior'} ckpt={ckpt}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = TdmpcVecEnv(env)
    agent = TDMPC2(agent_cfg, env.num_obs, env.num_actions, device)
    agent.load(ckpt)
    act_scale = float(agent_cfg.act_env_scale)

    uenv = env.uenv
    robot = uenv.scene["robot"]
    dt = float(uenv.step_dt)
    dev = uenv.device
    N = env.num_envs

    n_steps = 0
    fwd_sum = torch.zeros((), device=dev)
    accel_sq = torch.zeros((), device=dev)
    rock_sq = torch.zeros((), device=dev)
    jvel_sq = torch.zeros((), device=dev)
    actrate_sq = torch.zeros((), device=dev)
    falls = torch.zeros((), device=dev)
    timeouts = torch.zeros((), device=dev)
    prev_action = None

    obs_p, obs_c = env.reset()
    total = args_cli.warmup + args_cli.steps
    with torch.no_grad():
        for i in range(total):
            if args_cli.plan:
                a = agent.plan_batch(obs_p, eval_mode=True)
            else:
                a = agent.act_pi(obs_p, eval_mode=True)
            env_action = a * act_scale
            obs_p, obs_c, reward, terminated, time_out, _ = env.step(env_action)
            if i >= args_cli.warmup:
                data = robot.data
                fwd_sum += data.root_lin_vel_b.torch[:, 0].sum()
                accel_sq += data.body_lin_acc_w.torch[:, 0, :2].square().sum()
                rock_sq += data.root_ang_vel_b.torch[:, :2].square().sum()
                jvel_sq += data.joint_vel.torch.square().mean(dim=1).sum()
                if prev_action is not None:
                    actrate_sq += (env_action - prev_action).square().mean(dim=1).sum()
                tm = uenv.termination_manager
                falls += (tm.dones & ~tm.time_outs).sum()
                timeouts += tm.time_outs.sum()
                n_steps += 1
            prev_action = env_action

    denom = max(n_steps * N, 1)
    steps_denom = max(n_steps - 1, 1) * N
    resets = float((falls + timeouts).item())
    env_seconds = n_steps * N * dt
    metrics = {
        "policy": "tdmpc2",
        "planner": "mppi" if args_cli.plan else "prior",
        "plant": args_cli.plant,
        "checkpoint": os.path.basename(ckpt),
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
    print("[eval-smooth] RESULT " + json.dumps(metrics))
    if args_cli.out:
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
        with open(args_cli.out, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[eval-smooth] wrote {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
