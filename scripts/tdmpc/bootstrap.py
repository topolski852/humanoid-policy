"""Bootstrap a TD-MPC2 world-model policy from the PPO walk (MPC-Injection style).

TD-MPC2 gets stuck in a standing basin because every run started from a stand/scratch. The PPO
policy already walks. This script SEEDS the TD-MPC2 replay buffer with PPO-walk transitions so
TD-MPC2 starts in the walking basin and REFINES the gait, instead of trying to discover it.

Why this is clean (verified in code): the TD-MPC2 walk env INHERITS the PPO env's observations
(45-dim) and actions (12-dim, scale 0.25, clip +/-4) UNCHANGED, and neither side normalizes obs.
A PPO raw action == (TD-MPC2 agent_action * act_env_scale=4.0) -- both are the pre-JointPositionAction
value. So we roll the PPO policy out IN the TD-MPC2 env (num_envs wide -> naturally buffer-shaped)
and store each step: obs (45), action = clamp(ppo_action, +/-4)/4 in [-1,1], reward (the TD-MPC2
reward), terminated, time_out.

Phases: 0) validate the PPO policy actually walks in this env (measure forward speed); 1) seed the
buffer; 2+3) hand to TdmpcTrainer with seed_steps=0 + a big seed-burst = the pretrain, then online
refine. BC of the policy prior is free via TD-M(PC)^2: seed transitions store plan_mean=demo_action
with a SHARPENED plan_std (default 0.3, vs the near-inert 2.0), so the prior loss pulls the policy
toward the demo actions. Run with --tdmpc2_square for that to be active.

Baseline plant by default (the deployed PPO walk was trained on uniform gains, NOT the modeled
actuator plant) -- keep seed + online dynamics consistent. Modeled plant is a follow-up.

Example:
  OMNI_KIT_ACCEPT_EULA=YES .venv/bin/python scripts/tdmpc/bootstrap.py \
    --variant walk-biped-tdmpc --num_envs 32 --tdmpc2_square --compile \
    --plant baseline --cmd_vx 0.3 --seed_transitions 300000 --pretrain_updates 20000 \
    --bc_plan_std 0.3 --max_env_steps 5000000 --headless
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rsl_rl"))
import variants  # noqa: E402  isort: skip

parser = argparse.ArgumentParser(description="Bootstrap TD-MPC2 from the PPO walk (buffer seeding).")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max_env_steps", type=int, default=None, help="Online budget after pretrain.")
parser.add_argument("--tdmpc2_square", action="store_true", help="TD-M(PC)^2 (enables BC-via-prior + plan collection).")
parser.add_argument("--updates_per_step", type=int, default=None)
parser.add_argument("--compile", action="store_true")
parser.add_argument("--overrides", type=str, default=None, help="JSON of dotted-path env_cfg patches.")
# --- bootstrap-specific ---
parser.add_argument("--ppo_policy", type=str, default="deploy/walk/policy.pt",
                    help="TorchScript PPO walk policy (obs45 -> action12). Source of truth: deploy/walk/.")
parser.add_argument("--seed_transitions", type=int, default=300_000,
                    help="Total PPO-walk transitions to inject (spread over num_envs).")
parser.add_argument("--pretrain_updates", type=int, default=20_000,
                    help="Gradient updates on the seeded buffer before online (the pretrain burst).")
parser.add_argument("--bc_plan_std", type=float, default=0.3,
                    help="plan_std stored on seed transitions -> BC strength via the TD-M(PC)^2 prior "
                         "(smaller = stronger; 2.0 = near-inert = buffer-injection only). Needs --tdmpc2_square.")
parser.add_argument("--cmd_vx", type=float, default=0.3, help="Fixed forward command (m/s) for seed + online.")
parser.add_argument("--plant", choices=["baseline", "modeled"], default="baseline")
parser.add_argument("--validate_only", action="store_true",
                    help="Phase 0 only: roll PPO out, report forward speed, exit (no seeding/training).")
variants.add_variant_arg(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.task is None and getattr(args_cli, "variant", None) is None:
    args_cli.variant = "walk-biped-tdmpc"
variants.resolve_variant(args_cli)

# plant: bootstrap defaults to BASELINE (matches the deployed PPO walk's uniform gains).
os.environ["HUMANOID_ACTUATOR_MODEL"] = "1" if args_cli.plant == "modeled" else "0"

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import gymnasium as gym  # noqa: E402
from datetime import datetime  # noqa: E402
import json  # noqa: E402

from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402
import humanoid_policy.tasks  # noqa: F401,E402

from humanoid_policy.tdmpc.agent import TDMPC2  # noqa: E402
from humanoid_policy.tdmpc.buffer import SequenceReplayBuffer  # noqa: E402
from env_adapter import TdmpcVecEnv  # noqa: E402
from trainer import TdmpcTrainer  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _apply_overrides(env_cfg, path):
    with open(path) as f:
        patches = json.load(f)
    for dotted, value in patches.items():
        if isinstance(value, list):
            value = tuple(value)
        segs = dotted.split("."); cur = env_cfg
        try:
            for s in segs[:-1]:
                cur = cur[s] if isinstance(cur, dict) else getattr(cur, s)
            if isinstance(cur, dict):
                cur[segs[-1]] = value
            else:
                setattr(cur, segs[-1], value)
            print(f"[bootstrap] override: {dotted} = {value}")
        except Exception as e:
            print(f"[bootstrap] WARNING override SKIPPED {dotted}: {e}")


def _load_ppo(path, device):
    """Load the exported TorchScript PPO walk policy. Returns a callable obs(N,45)->action(N,12)."""
    p = path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)
    print(f"[bootstrap] loading PPO policy: {p}")
    policy = torch.jit.load(p, map_location=device)
    policy.eval()

    def act(obs):
        with torch.no_grad():
            try:
                a = policy(obs)
            except Exception:
                a = policy({"policy": obs})   # fallback if it expects an obs dict
        return a if isinstance(a, torch.Tensor) else a["actions"]
    return act


def main():
    device = args_cli.device or "cuda:0"
    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    if args_cli.overrides:
        _apply_overrides(env_cfg, args_cli.overrides)
    # fix the command to a steady forward walk (seed + online + eval all use cmd_vx)
    r = env_cfg.commands.base_velocity.ranges
    r.lin_vel_x = (args_cli.cmd_vx, args_cli.cmd_vx)
    r.lin_vel_y = (0.0, 0.0)
    r.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.commands.base_velocity.heading_command = False

    agent_cfg = load_cfg_from_registry(args_cli.task, "tdmpc_cfg_entry_point")
    agent_cfg.num_envs = args_cli.num_envs
    agent_cfg.seed = args_cli.seed
    if args_cli.max_env_steps is not None:
        agent_cfg.max_env_steps = args_cli.max_env_steps
    if args_cli.updates_per_step is not None:
        agent_cfg.updates_per_step = args_cli.updates_per_step
    if args_cli.compile:
        agent_cfg.compile = True
    if args_cli.tdmpc2_square:
        agent_cfg.use_tdmpc2_square = True
        agent_cfg.plan_collection = True
    # bootstrap: NO random seed phase (the buffer is seeded from PPO); the "seed burst" is our pretrain.
    agent_cfg.seed_steps = 0
    agent_cfg.seed_burst_updates = int(args_cli.pretrain_updates)

    torch.manual_seed(args_cli.seed)
    print(f"[bootstrap] task={args_cli.task} plant={args_cli.plant} "
          f"(HUMANOID_ACTUATOR_MODEL={os.environ['HUMANOID_ACTUATOR_MODEL']}) cmd_vx={args_cli.cmd_vx}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = TdmpcVecEnv(env)
    N = env.num_envs
    print(f"[bootstrap] env: num_envs={N} obs={env.num_obs} act={env.num_actions} step_dt={env.step_dt}")

    ppo = _load_ppo(args_cli.ppo_policy, env.device)
    robot = env.uenv.scene["robot"]

    # ---------------- Phase 0/1: validate + seed the buffer with PPO-walk transitions ----------------
    agent = TDMPC2(agent_cfg, env.num_obs, env.num_actions, device)
    buffer = SequenceReplayBuffer(agent_cfg, N, env.num_obs, env.num_priv_obs, env.num_actions, device)
    act_scale = float(agent_cfg.act_env_scale)
    use_sq = bool(agent_cfg.use_tdmpc2_square)

    n_iters = max(1, args_cli.seed_transitions // N)
    print(f"[bootstrap] SEED phase: rolling PPO for {n_iters} steps x {N} envs "
          f"= {n_iters*N} transitions (bc_plan_std={args_cli.bc_plan_std if use_sq else 'n/a (no sq)'})")
    obs_p, obs_c = env.reset()
    fwd_sum = torch.zeros((), device=env.device)
    falls = torch.zeros((), device=env.device)
    warmup = 20
    counted = 0
    for i in range(n_iters if not args_cli.validate_only else min(n_iters, 400)):
        ppo_action = ppo(obs_p)                        # (N,12) raw, PPO deterministic mean
        env_action = ppo_action                        # same space the TD-MPC2 env consumes
        nobs_p, nobs_c, reward, terminated, time_out, _ = env.step(env_action)
        agent_action = (ppo_action.clamp(-act_scale, act_scale)) / act_scale   # -> [-1,1]
        if not args_cli.validate_only:
            if use_sq:
                plan_mean = agent_action
                plan_std = torch.full_like(agent_action, args_cli.bc_plan_std)
            else:
                plan_mean = plan_std = None
            buffer.add(obs_p, agent_action, reward, terminated, time_out,
                       plan_mean=plan_mean, plan_std=plan_std)
        if i >= warmup:
            fwd_sum += robot.data.root_lin_vel_b.torch[:, 0].sum()
            falls += (terminated & ~time_out).sum()
            counted += 1
        obs_p, obs_c = nobs_p, nobs_c
        if i % 500 == 0 and i > 0:
            fs = float(fwd_sum / max(counted * N, 1))
            print(f"[bootstrap]   seed step {i}/{n_iters}  ppo_fwd_speed={fs:.3f} m/s (cmd {args_cli.cmd_vx})")

    ppo_fwd = float(fwd_sum / max(counted * N, 1))
    print(f"[bootstrap] PHASE-0 RESULT: PPO forward_speed={ppo_fwd:.3f} m/s (cmd {args_cli.cmd_vx}), "
          f"falls={int(falls.item())} over {counted} measured steps.")
    if ppo_fwd < 0.15:
        print(f"[bootstrap] *** WARNING: PPO is NOT walking in this env (fwd {ppo_fwd:.3f} < 0.15). "
              f"Seed data will be poor -- check plant/command/policy before trusting a long run. ***")
    if args_cli.validate_only:
        print("[bootstrap] --validate_only: done."); env.close(); return
    print(f"[bootstrap] buffer seeded: {len(buffer)} transitions ({buffer.size} rows x {N}).")

    # ---------------- Phase 2+3: pretrain burst (on the seed) + online refine, via TdmpcTrainer -------
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.abspath(os.path.join("logs", "tdmpc", agent_cfg.experiment_name, ts))
    os.makedirs(log_dir, exist_ok=True)
    print(f"[bootstrap] logging to {log_dir}")
    # durable record
    try:
        import subprocess
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                                         stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        commit = "unknown"
    with open(os.path.join(log_dir, "run_config.json"), "w") as f:
        json.dump({"git_commit": commit, "kind": "bootstrap-from-ppo", "task": args_cli.task,
                   "plant": args_cli.plant, "cmd_vx": args_cli.cmd_vx, "seed": args_cli.seed,
                   "num_envs": N, "ppo_policy": args_cli.ppo_policy,
                   "seed_transitions": n_iters * N, "pretrain_updates": args_cli.pretrain_updates,
                   "bc_plan_std": args_cli.bc_plan_std if use_sq else None,
                   "max_env_steps": agent_cfg.max_env_steps, "tdmpc2_square": use_sq,
                   "updates_per_step": agent_cfg.updates_per_step, "overrides": args_cli.overrides}, f, indent=2)

    print(f"[bootstrap] PRETRAIN+ONLINE: seed_burst(pretrain)={agent_cfg.seed_burst_updates} updates, "
          f"then online to {agent_cfg.max_env_steps} env-steps.")
    trainer = TdmpcTrainer(agent_cfg, env, agent, buffer, log_dir)
    trainer.train()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
