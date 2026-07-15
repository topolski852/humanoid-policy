"""Search configuration: the tunable terms, the fixed ground-truth fitness, TB tags.

Everything the loop needs to know about the walk-biped task lives here so the rest
of the package is task-agnostic. To retarget another task, add its term table +
experiment name and (if its fitness differs) its fitness inputs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

# --- The genome: walk-biped RewardsCfg terms -------------------------------------
# Verified against source/.../velocity/config/biped/env_cfg.py AND a real run's TB
# tags. name -> (default_weight, sign). sign +1 = bonus (kept >= 0), -1 = penalty
# (kept <= 0). The search evolves MAGNITUDES; the sign is fixed so a proposal can
# never invert a penalty into a reward (a fitness-hacking guard). 0 prunes a term.
WALK_TERMS: dict[str, tuple[float, int]] = {
    "track_lin_vel_xy_exp":       (2.0,     +1),
    "track_ang_vel_z_exp":        (1.0,     +1),
    "feet_air_time":              (1.15,    +1),
    "termination_penalty":        (-10.0,   -1),
    "lin_vel_z_l2":               (-0.1,    -1),
    "ang_vel_xy_l2":              (-0.05,   -1),
    "flat_orientation_l2":        (-2.0,    -1),
    "action_rate_l2":             (-0.05,   -1),
    "action_l2":                  (-0.01,   -1),
    "dof_vel_l2":                 (-2.5e-4, -1),
    "dof_torques_l2":             (-2e-3,   -1),
    "dof_acc_l2":                 (-1e-6,   -1),
    "dof_pos_limits":             (-1.0,    -1),
    "feet_slide":                 (-0.1,    -1),
    "undesired_contacts":         (-1.0,    -1),
    "joint_deviation_hip":        (-0.2,    -1),
    "joint_deviation_ankle_roll": (-0.2,    -1),
}
TERM_NAMES = list(WALK_TERMS)
DEFAULT_WEIGHTS = {k: v[0] for k, v in WALK_TERMS.items()}
SIGNS = {k: v[1] for k, v in WALK_TERMS.items()}
# generous per-term magnitude cap for the search (8x the default magnitude)
MAX_MAG = {k: max(abs(v[0]) * 8.0, 1e-5) for k, v in WALK_TERMS.items()}

# --- Ground-truth fitness (weight-INVARIANT; read from TB Metrics/* + Termination/*)
# Every input below is logged by the trainer independent of the reward weights, so
# the search cannot inflate a weight to raise its own fitness. Components map to
# [0,1] (higher = better); the fixed weights below combine them.
TRACK_SCALE = 0.35   # maps a velocity-tracking error to goodness via exp(-err/scale)
FITNESS_WEIGHTS = {
    "success":   0.35,   # Metrics/success_rate (velocity-command tracking success)
    "upright":   0.20,   # 1 - fall_rate  (Episode_Termination/base_orientation)
    "survive":   0.15,   # mean_episode_length / max_ep_steps
    "track_xy":  0.15,   # exp(-error_vel_xy / TRACK_SCALE)
    "track_yaw": 0.10,   # exp(-error_vel_yaw / TRACK_SCALE)
    "standing":  0.05,   # 1 - collapse_rate (Episode_Termination/base_height)
}

# TB scalar tags (confirmed present in a real biped run)
TAG_MEAN_REWARD = "Train/mean_reward"          # inner plateau signal (100-ep rolling mean)
TAG_MEAN_EP_LEN = "Train/mean_episode_length"
TAG_SUCCESS = "Metrics/success_rate"
TAG_ERR_XY = "Metrics/base_velocity/error_vel_xy"
TAG_ERR_YAW = "Metrics/base_velocity/error_vel_yaw"
TAG_TERM_ORIENT = "Episode_Termination/base_orientation"
TAG_TERM_HEIGHT = "Episode_Termination/base_height"

# variant -> rsl_rl experiment_name (the logs/rsl_rl/<experiment_name>/ dir)
VARIANT_EXPERIMENT = {
    "walk-biped": "biped", "walk-humanoid": "humanoid",
    "standup-biped": "standup_biped", "standup-humanoid": "standup_humanoid",
    "squat-biped": "squat_biped",
}
DEFAULT_MAX_EP_STEPS = 500   # walk-biped: 20.0s / (0.005*8); read per-run when possible


@dataclass
class SearchConfig:
    variant: str = "walk-biped"
    profile: str = "fast"
    max_iterations: int = 250      # inner proxy cap per candidate
    iterations: int = 8            # max outer generations
    candidates: int = 6            # candidates per generation
    inner_patience: int = 40       # iters of no mean_reward improvement -> stop the run
    inner_min_delta: float = 0.02  # mean_reward improvement that counts (its scale is ~10s)
    patience: int = 3              # generations of no best-fitness improvement -> stop search
    min_delta: float = 1e-3        # fitness improvement that counts
    backend: str = "subprocess"    # subprocess | persistent
    seed: int = 0
    inner_min_iters: int = 1500    # never plateau-stop before this many iters (let walking emerge)
    poll_secs: float = 20.0        # how often to poll the run's TB for plateau
    dry_run: bool = False          # score an existing run dir + exit (no training/API)
    log: str | None = None
    best_out: str = "eureka_best.json"
    verbose: bool = True
    run_name: str = "eureka"       # set per-candidate by the loop


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Eureka-style reward-weight tuner (API-free local search) for humanoid-policy")
    d = SearchConfig()
    ap.add_argument("--variant", default=d.variant, choices=list(VARIANT_EXPERIMENT))
    ap.add_argument("--profile", default=d.profile, choices=["fast", "full"])
    ap.add_argument("--max_iterations", type=int, default=d.max_iterations,
                    help="inner training cap per candidate (proxy run)")
    ap.add_argument("--iterations", type=int, default=d.iterations,
                    help="MAX outer generations (may stop earlier via --patience)")
    ap.add_argument("--candidates", type=int, default=d.candidates)
    ap.add_argument("--inner-patience", type=int, default=d.inner_patience,
                    help="stop a training run after N iters without mean_reward improvement")
    ap.add_argument("--inner-min-delta", type=float, default=d.inner_min_delta)
    ap.add_argument("--inner-min-iters", type=int, default=d.inner_min_iters,
                    help="never plateau-stop a run before this many iters (walking emerges ~3000)")
    ap.add_argument("--patience", type=int, default=d.patience,
                    help="stop the search after N generations without fitness improvement")
    ap.add_argument("--min-delta", type=float, default=d.min_delta)
    ap.add_argument("--backend", default=d.backend, choices=["subprocess", "persistent"])
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--poll-secs", type=float, default=d.poll_secs)
    ap.add_argument("--dry-run", action="store_true",
                    help="score the newest existing run dir for --variant and exit "
                         "(verifies TB->fitness with no training, no API)")
    ap.add_argument("--log", default=d.log, help="JSONL path to append every candidate")
    ap.add_argument("--best-out", default=d.best_out)
    ap.add_argument("--quiet", action="store_true")
    return ap


def build_config() -> SearchConfig:
    a = build_argparser().parse_args()
    return SearchConfig(
        variant=a.variant, profile=a.profile, max_iterations=a.max_iterations,
        iterations=a.iterations, candidates=a.candidates,
        inner_patience=a.inner_patience, inner_min_delta=a.inner_min_delta,
        inner_min_iters=a.inner_min_iters,
        patience=a.patience, min_delta=a.min_delta, backend=a.backend, seed=a.seed,
        poll_secs=a.poll_secs, dry_run=a.dry_run, log=a.log, best_out=a.best_out,
        verbose=not a.quiet,
    )


def hydra_override_str(weights: dict[str, float]) -> str:
    """The ready-to-paste Hydra CLI override that reproduces a weight set."""
    return " ".join(f"env.rewards.{t}.weight={weights[t]:.6g}" for t in TERM_NAMES)
