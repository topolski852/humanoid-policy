"""Honest fall diagnostics for a walk policy — TD-MPC2 checkpoint or the PPO TorchScript baseline.

WHY THIS EXISTS. `eval_smoothness.py` reports `falls` as `termination_manager.dones & ~time_outs`,
which in the TD-MPC2 walk env is the SINGLE `hard_collapse` term: base height 0.30 m below standing.
That env deliberately dropped the PPO env's 45-degree `base_orientation` termination so a stumble
would not end the episode. The consequence nobody priced: a policy can topple, lie over, scrape
along and get back up without ever tripping that one deep-height threshold, so the eval prints
`falls: 0.0` for a gait that visibly falls. Every "0 falls" claim in this project's journal is
measured against that blind criterion.

This script instead measures falls the way a person watching the viewer would count them:
  * fall event   = torso tilt crosses ABOVE `--fall_deg` (default 45, the PPO env's own bar)
  * severe fall  = tilt crosses above `--severe_deg` (default 80 = past horizontal, definitely down)
  * recovery     = tilt returns below `--recover_deg` (default 25) before the episode ends
and reports the number that matches the human question "what fraction of episodes fall?".

Also reports the legacy hard_collapse count side by side, so the gap between the two is explicit.

Usage (GPU, headless, ~3 min):
    .venv/bin/python scripts/tdmpc/diagnose_falls.py --checkpoint <ckpt.pt> --plan --headless
    .venv/bin/python scripts/tdmpc/diagnose_falls.py --ppo_policy deploy/walk/policy.pt --headless
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rsl_rl"))
import variants  # noqa: E402

parser = argparse.ArgumentParser(description="Measure real fall rates for a walk policy.")
parser.add_argument("--checkpoint", type=str, default=None, help="TD-MPC2 .pt (or a run dir).")
parser.add_argument("--ppo_policy", type=str, default=None, help="TorchScript PPO policy instead.")
parser.add_argument("--plan", action="store_true", help="TD-MPC2: use MPPI (else the policy prior).")
parser.add_argument("--plant", choices=["baseline", "modeled"], default="baseline")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=1500, help="measured steps (3 episodes of 500).")
parser.add_argument("--warmup", type=int, default=50)
parser.add_argument("--cmd_vx", type=float, default=0.3)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--fall_deg", type=float, default=45.0)
parser.add_argument("--severe_deg", type=float, default=80.0)
parser.add_argument("--recover_deg", type=float, default=25.0)
parser.add_argument("--out", type=str, default=None)
parser.add_argument("--label", type=str, default=None)
parser.add_argument("--horizon", type=int, default=None,
                    help="override the MPPI planning horizon at INFERENCE. The world model is a "
                         "one-step dynamics model and none of the loaded weights depend on horizon, "
                         "so this probes D5 (is the model's rollout usable past 0.12 s?) on an "
                         "existing checkpoint without retraining.")
parser.add_argument("--warm_start", action="store_true",
                    help="enable the MPPI warm start (agent.plan_batch). Inference-time only, so this "
                         "A/Bs on an EXISTING checkpoint with no retraining.")
variants.add_variant_arg(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.task is None and getattr(args_cli, "variant", None) is None:
    args_cli.variant = "walk-biped-tdmpc"
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

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAD2DEG = 57.29577951308232


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
    agent_cfg.mppi_warm_start = bool(args_cli.warm_start)
    if args_cli.horizon is not None:
        agent_cfg.horizon = int(args_cli.horizon)   # inference-only; weights are horizon-independent
    device = args_cli.device or env_cfg.sim.device

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = TdmpcVecEnv(env)
    uenv, robot = env.uenv, env.uenv.scene["robot"]
    dt, dev, N = float(uenv.step_dt), uenv.device, env.num_envs
    act_scale = float(agent_cfg.act_env_scale)

    if args_cli.ppo_policy:
        p = args_cli.ppo_policy if os.path.isabs(args_cli.ppo_policy) \
            else os.path.join(REPO_ROOT, args_cli.ppo_policy)
        jit = torch.jit.load(p, map_location=dev).eval()
        label = args_cli.label or f"PPO {os.path.basename(p)}"

        def act(obs, reset=None):
            return jit(obs).clamp(-act_scale, act_scale)
    else:
        ck = args_cli.checkpoint
        if os.path.isdir(ck):
            ck = os.path.join(ck, "model_best.pt")
        agent = TDMPC2(agent_cfg, env.num_obs, env.num_actions, device)
        agent.load(ck)
        label = args_cli.label or os.path.basename(ck)

        def act(obs, reset=None):
            a, _, _ = agent.plan_batch(obs, eval_mode=True, reset=reset) if args_cli.plan \
                else (agent.act_pi(obs, eval_mode=True), None, None)
            return a * act_scale

    print(f"[falls] H={agent_cfg.horizon} warm_start={bool(args_cli.warm_start)}")
    print(f"[falls] {label} | plant={args_cli.plant} | "
          f"{'mppi' if args_cli.plan else ('ppo' if args_cli.ppo_policy else 'prior')} | "
          f"fall>{args_cli.fall_deg:.0f}deg severe>{args_cli.severe_deg:.0f}deg")

    # --- per-env fall state machine -------------------------------------------------------------
    was_down = torch.zeros(N, dtype=torch.bool, device=dev)   # currently past fall_deg
    ep_fell = torch.zeros(N, dtype=torch.bool, device=dev)    # this episode has had a fall
    fall_events = torch.zeros((), device=dev)
    severe_events = torch.zeros((), device=dev)
    recoveries = torch.zeros((), device=dev)
    episodes = torch.zeros((), device=dev)
    episodes_with_fall = torch.zeros((), device=dev)
    hard_collapse = torch.zeros((), device=dev)
    timeouts = torch.zeros((), device=dev)
    steps_down = torch.zeros((), device=dev)
    tilt_sum = torch.zeros((), device=dev)
    fwd_sum = torch.zeros((), device=dev)
    n = 0

    obs_p, _ = env.reset()
    prev_done = None
    with torch.no_grad():
        for i in range(args_cli.warmup + args_cli.steps):
            obs_p, _, _, term, tout, _ = env.step(act(obs_p, reset=prev_done))
            prev_done = term | tout
            gb = robot.data.projected_gravity_b.torch
            tilt = torch.acos((-gb[:, 2]).clamp(-1, 1)) * RAD2DEG
            if i >= args_cli.warmup:
                down = tilt > args_cli.fall_deg
                fall_events += (down & ~was_down).sum()                  # rising edge
                severe_events += ((tilt > args_cli.severe_deg) & ~was_down).sum()
                recoveries += ((tilt < args_cli.recover_deg) & was_down).sum()
                was_down = torch.where(tilt < args_cli.recover_deg, torch.zeros_like(down), down | was_down)
                ep_fell |= down
                steps_down += down.sum()
                tilt_sum += tilt.sum()
                fwd_sum += robot.data.root_lin_vel_b.torch[:, 0].sum()
                done = term | tout
                if done.any():
                    episodes += done.sum()
                    episodes_with_fall += (ep_fell & done).sum()
                    ep_fell = torch.where(done, torch.zeros_like(ep_fell), ep_fell)
                    was_down = torch.where(done, torch.zeros_like(was_down), was_down)
                hard_collapse += (term & ~tout).sum()
                timeouts += tout.sum()
                n += 1
            else:
                was_down = tilt > args_cli.fall_deg

    env_sec = n * N * dt
    eps = float(episodes.item())
    out = {
        "label": label, "plant": args_cli.plant,
        "policy": "ppo" if args_cli.ppo_policy else ("mppi" if args_cli.plan else "prior"),
        "num_envs": N, "measured_steps": n, "env_minutes": round(env_sec / 60.0, 1),
        "cmd_vx": args_cli.cmd_vx,
        "forward_speed_mean": float(fwd_sum.item() / (n * N)),
        "mean_tilt_deg": float(tilt_sum.item() / (n * N)),
        # --- the honest fall numbers ---
        "fall_events": float(fall_events.item()),
        "severe_fall_events": float(severe_events.item()),
        "recoveries": float(recoveries.item()),
        "real_falls_per_min": float(fall_events.item() / env_sec * 60.0),
        "episodes": eps,
        "episodes_with_a_fall": float(episodes_with_fall.item()),
        "pct_episodes_with_fall": round(100.0 * episodes_with_fall.item() / eps, 1) if eps else None,
        "pct_time_past_fall_angle": round(100.0 * steps_down.item() / (n * N), 2),
        # --- what the OLD metric saw ---
        "legacy_hard_collapse_falls": float(hard_collapse.item()),
        "legacy_falls_per_min": float(hard_collapse.item() / env_sec * 60.0),
        "timeouts": float(timeouts.item()),
    }
    print("[falls] RESULT " + json.dumps(out))
    print(f"[falls] >>> {out['pct_episodes_with_fall']}% of {int(eps)} episodes contained a fall "
          f"(>{args_cli.fall_deg:.0f}deg); {out['real_falls_per_min']:.2f} falls/min real vs "
          f"{out['legacy_falls_per_min']:.2f} by the old hard_collapse metric")
    if args_cli.out:
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
        with open(args_cli.out, "w") as f:
            json.dump(out, f, indent=2)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
