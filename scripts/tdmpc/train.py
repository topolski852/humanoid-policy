"""TD-MPC2 trainer entrypoint (Phase 0 skeleton).

Boots the biped walk env at low vectorization (32 envs) on the modeled actuator plant and, in
this P0 skeleton, drives a random-action loop through the raw-tuple adapter to prove the boot
path, config registration, and the 45/48 obs interface. Learning (buffer + world model + loop)
is wired in later phases.

Mirrors scripts/rsl_rl/train.py's AppLauncher-first ordering: nothing from isaaclab.envs may be
imported before AppLauncher() runs.
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# Reuse the variant->task-id registry from the rsl_rl scripts (framework-agnostic).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rsl_rl"))
import variants  # noqa: E402  isort: skip

parser = argparse.ArgumentParser(description="Train a TD-MPC2 world-model policy.")
parser.add_argument("--num_envs", type=int, default=32, help="Parallel envs (keep LOW for TD-MPC2).")
parser.add_argument("--task", type=str, default=None, help="Gym task id (usually via --variant).")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max_env_steps", type=int, default=200, help="P0: number of random env-steps to run.")
variants.add_variant_arg(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.task is None and getattr(args_cli, "variant", None) is None:
    args_cli.variant = "walk-biped"
variants.resolve_variant(args_cli)

# Train on the bench-modeled actuator plant by default so the world model learns real dynamics.
os.environ.setdefault("HUMANOID_ACTUATOR_MODEL", "1")

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- post-launch imports ---------------------------------------------------------------------
import torch  # noqa: E402
import gymnasium as gym  # noqa: E402

from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402
import humanoid_policy.tasks  # noqa: F401,E402  (triggers gym.register)

from env_adapter import TdmpcVecEnv  # noqa: E402  (sibling module, script dir on sys.path)


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed

    agent_cfg = load_cfg_from_registry(args_cli.task, "tdmpc_cfg_entry_point")
    print(f"[tdmpc] task={args_cli.task}  plant=HUMANOID_ACTUATOR_MODEL={os.environ['HUMANOID_ACTUATOR_MODEL']}")
    print(f"[tdmpc] agent_cfg: latent_dim={agent_cfg.latent_dim} horizon={agent_cfg.horizon} "
          f"num_envs={agent_cfg.num_envs} use_privileged_critic={agent_cfg.use_privileged_critic} "
          f"use_tdmpc2_square={agent_cfg.use_tdmpc2_square}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = TdmpcVecEnv(env)
    print(f"[tdmpc] adapter: num_envs={env.num_envs} num_obs={env.num_obs} "
          f"num_priv_obs={env.num_priv_obs} num_actions={env.num_actions} step_dt={env.step_dt} dev={env.device}")

    obs_p, obs_c = env.reset()
    assert obs_p.shape == (env.num_envs, 45), obs_p.shape
    assert obs_c.shape == (env.num_envs, 48), obs_c.shape
    print(f"[tdmpc] reset obs shapes: policy={tuple(obs_p.shape)} critic={tuple(obs_c.shape)}")

    # P0: random-action loop through the raw 5-tuple interface.
    rew_sum = torch.zeros(env.num_envs, device=env.device)
    n_term = 0
    n_tout = 0
    for i in range(args_cli.max_env_steps):
        action = (2.0 * torch.rand(env.num_envs, env.num_actions, device=env.device) - 1.0)
        obs_p, obs_c, reward, terminated, time_out, _ = env.step(action)
        rew_sum += reward
        n_term += int(terminated.sum().item())
        n_tout += int(time_out.sum().item())

    print(f"[tdmpc] ran {args_cli.max_env_steps} random steps | mean_step_reward="
          f"{(rew_sum.mean() / args_cli.max_env_steps).item():.4f} | terminated={n_term} time_out={n_tout}")
    print("[tdmpc] P0 boot OK")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
