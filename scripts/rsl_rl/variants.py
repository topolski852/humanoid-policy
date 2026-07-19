"""Training-variant registry.

Maps the user-facing ``--variant`` name to the registered gym task id, so a single training call
can select between policy types (stand-up vs walk) and robot scopes (legs-only biped vs full
humanoid) without the user memorizing gym ids.
"""

# variant name -> gym task id
VARIANTS = {
    "walk-biped": "Walk-Humanoid-Policy-Biped-v0",
    "walk-biped-tdmpc": "Walk-Humanoid-Policy-Biped-Tdmpc-v0",  # TD-MPC2 gated-reward task (phase 2: walk)
    "walk-biped-tdmpc-stand": "Walk-Humanoid-Policy-Biped-Tdmpc-Stand-v0",  # phase 1: learn to stand
    "walk-humanoid": "Walk-Humanoid-Policy-v0",
    "standup-biped": "Standup-Humanoid-Policy-Biped-v0",
    "standup-humanoid": "Standup-Humanoid-Policy-v0",
    "squat-biped": "Squat-Humanoid-Policy-Biped-v0",
}


def add_variant_arg(parser):
    """Add the ``--variant`` argument to an argparse parser."""
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        choices=sorted(VARIANTS.keys()),
        help="Training variant; resolves to a gym task id. Overrides --task. "
        f"One of: {', '.join(sorted(VARIANTS.keys()))}.",
    )


def resolve_variant(args_cli):
    """If ``--variant`` is set, populate ``args_cli.task`` from it (variant wins over --task).

    Call this immediately after parsing args, before anything reads ``args_cli.task``.
    """
    variant = getattr(args_cli, "variant", None)
    if variant is None:
        return args_cli
    task = VARIANTS[variant]
    if getattr(args_cli, "task", None) not in (None, task):
        print(f"[WARN] --variant {variant} overrides --task {args_cli.task} -> {task}")
    args_cli.task = task
    return args_cli
