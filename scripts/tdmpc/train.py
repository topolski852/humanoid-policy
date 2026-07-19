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
parser.add_argument("--max_env_steps", type=int, default=None, help="Total env-steps budget (override cfg).")
parser.add_argument("--seed_steps", type=int, default=None, help="Random-action warmup env-steps (override cfg).")
parser.add_argument("--plan_collection", action="store_true", help="Collect with the MPPI planner (proper TD-MPC2).")
parser.add_argument("--tdmpc2_square", action="store_true",
                    help="Enable TD-M(PC)² policy regularization (needs --plan_collection).")
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
from datetime import datetime  # noqa: E402

from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402
import humanoid_policy.tasks  # noqa: F401,E402  (triggers gym.register)

from humanoid_policy.tdmpc.agent import TDMPC2  # noqa: E402
from humanoid_policy.tdmpc.buffer import SequenceReplayBuffer  # noqa: E402
from env_adapter import TdmpcVecEnv  # noqa: E402  (sibling module, script dir on sys.path)
from trainer import TdmpcTrainer  # noqa: E402


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed

    agent_cfg = load_cfg_from_registry(args_cli.task, "tdmpc_cfg_entry_point")
    agent_cfg.num_envs = args_cli.num_envs
    agent_cfg.seed = args_cli.seed
    if args_cli.max_env_steps is not None:
        agent_cfg.max_env_steps = args_cli.max_env_steps
    if args_cli.seed_steps is not None:
        agent_cfg.seed_steps = args_cli.seed_steps
    if args_cli.plan_collection:
        agent_cfg.plan_collection = True
    if args_cli.tdmpc2_square:
        agent_cfg.use_tdmpc2_square = True
        agent_cfg.plan_collection = True   # TD-M(PC)² needs planner mu/std

    torch.manual_seed(args_cli.seed)
    device = args_cli.device or env_cfg.sim.device
    print(f"[tdmpc] task={args_cli.task}  plant=HUMANOID_ACTUATOR_MODEL={os.environ['HUMANOID_ACTUATOR_MODEL']}")
    print(f"[tdmpc] cfg: latent={agent_cfg.latent_dim} horizon={agent_cfg.horizon} num_envs={agent_cfg.num_envs} "
          f"max_env_steps={agent_cfg.max_env_steps} seed_steps={agent_cfg.seed_steps} "
          f"plan_collection={agent_cfg.plan_collection} priv={agent_cfg.use_privileged_critic} "
          f"tdmpc2sq={agent_cfg.use_tdmpc2_square}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = TdmpcVecEnv(env)
    print(f"[tdmpc] adapter: num_envs={env.num_envs} obs={env.num_obs} priv={env.num_priv_obs} "
          f"act={env.num_actions} step_dt={env.step_dt} dev={env.device}")

    agent = TDMPC2(agent_cfg, env.num_obs, env.num_actions, device)
    print(f"[tdmpc] world-model params: {agent.model.total_params:,}")
    buffer = SequenceReplayBuffer(agent_cfg, env.num_envs, env.num_obs, env.num_priv_obs,
                                  env.num_actions, device)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.abspath(os.path.join("logs", "tdmpc", agent_cfg.experiment_name, ts))
    print(f"[tdmpc] logging to {log_dir}")

    trainer = TdmpcTrainer(agent_cfg, env, agent, buffer, log_dir)
    trainer.train()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
