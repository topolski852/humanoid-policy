"""Training scale profiles — a dual "full" / "fast" system.

Selected with ``--profile``. Explicit ``--num_envs`` / ``--max_iterations`` always win over the
profile. Independent of ``--variant`` (any variant can run at either scale).

- ``full``: the tuned RTX-5080 setup (16384 envs). High-fidelity run; uses each task's own
  ``max_iterations`` default. ~3.5 h wall time for the biped (~30 min IsaacSim startup + training).
- ``fast``: the original 4096-env scale for quick dev iterations, targeted to finish **sub-3 h**.
  With ~30 min fixed startup that leaves ~2.5 h of training; FAST_MAX_ITERATIONS is sized for that
  but is hardware/task dependent — watch the reported sec/iter on the first run and tune it here.
"""

FAST_MAX_ITERATIONS = 2000

PROFILES = {
    "full": {
        "num_envs": 16384,
        "num_steps_per_env": 48,
        "num_mini_batches": 8,
        "max_iterations": None,  # use the task's agent-cfg default
    },
    "fast": {
        "num_envs": 4096,
        "num_steps_per_env": 24,
        "num_mini_batches": 4,
        "max_iterations": FAST_MAX_ITERATIONS,
    },
}


def add_profile_arg(parser):
    """Add the ``--profile`` argument to an argparse parser."""
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        choices=sorted(PROFILES.keys()),
        help="Training scale preset. 'full' = tuned 16k-env run; 'fast' = 4k-env dev run (sub-3h). "
        "Explicit --num_envs / --max_iterations override the preset.",
    )


def apply_profile(agent_cfg, env_cfg, args_cli):
    """Apply the selected ``--profile`` to the loaded env/agent configs.

    Call this AFTER the existing --num_envs / --max_iterations overrides so those explicit flags win.
    """
    profile = getattr(args_cli, "profile", None)
    if profile is None:
        return
    p = PROFILES[profile]

    # scale: number of parallel envs (explicit --num_envs already applied by caller wins)
    if getattr(args_cli, "num_envs", None) is None:
        env_cfg.scene.num_envs = p["num_envs"]

    # rollout / batching
    agent_cfg.num_steps_per_env = p["num_steps_per_env"]
    agent_cfg.algorithm.num_mini_batches = p["num_mini_batches"]

    # iteration budget (explicit --max_iterations wins; else profile default if it sets one)
    if p["max_iterations"] is not None and getattr(args_cli, "max_iterations", None) is None:
        agent_cfg.max_iterations = p["max_iterations"]

    print(
        f"[INFO] profile='{profile}': num_envs={env_cfg.scene.num_envs}, "
        f"num_steps_per_env={agent_cfg.num_steps_per_env}, "
        f"num_mini_batches={agent_cfg.algorithm.num_mini_batches}, "
        f"max_iterations={agent_cfg.max_iterations}"
    )
