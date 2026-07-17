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
# DEFAULTS ARE BERKELEY-EXACT so generation-0 candidate "default" reproduces the
# Berkeley-Humanoid-Lite reward that is proven to walk (our known-good baseline). The
# three hardware-safety penalties we later added (base_accel_xy_l2, action_l2, dof_vel_l2)
# default to 0 here — Berkeley has none — but stay SEARCHABLE via MAX_MAG_OVERRIDE below,
# so Eureka can re-introduce the sim->real damping incrementally once it walks. action_rate
# is back to Berkeley's -0.01 (we had over-cranked it to -0.05, which helped kill walking).
WALK_TERMS: dict[str, tuple[float, int]] = {
    "track_lin_vel_xy_exp":       (2.0,     +1),
    "track_ang_vel_z_exp":        (1.0,     +1),
    "feet_air_time":              (1.0,     +1),   # Berkeley 1.0 (we had bumped to 1.15)
    "termination_penalty":        (-10.0,   -1),
    "lin_vel_z_l2":               (-0.1,    -1),
    "ang_vel_xy_l2":              (-0.05,   -1),
    "base_accel_xy_l2":           (0.0,     -1),   # OFF by default (Berkeley has none); searchable
    "flat_orientation_l2":        (-2.0,    -1),
    "action_rate_l2":             (-0.01,   -1),   # Berkeley -0.01 (we had over-cranked to -0.05)
    "action_l2":                  (0.0,     -1),   # OFF by default (Berkeley has none); searchable
    "dof_vel_l2":                 (0.0,     -1),   # OFF by default (Berkeley has none); searchable
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
# Terms whose Berkeley default is 0 but that we still want the search to EXPLORE (adding
# hardware-safety damping back). Without these, MAX_MAG would be ~0 and freeze them at 0.
MAX_MAG_OVERRIDE = {
    "base_accel_xy_l2": 0.05,
    "action_l2":        0.05,
    "dof_vel_l2":       1.0e-3,
}
# generous per-term magnitude cap for the search (8x the default magnitude, or the override)
MAX_MAG = {
    k: MAX_MAG_OVERRIDE.get(k, max(abs(v[0]) * 8.0, 1e-5)) for k, v in WALK_TERMS.items()
}

# --- Ground-truth fitness (weight-INVARIANT; read from TB Metrics/* + Termination/*)
# Every input below is logged by the trainer independent of the reward weights, so
# the search cannot inflate a weight to raise its own fitness. All inputs map to
# [0,1] (higher = better).
#
# GATED design: fitness = walk_gate * quality.
#   walk_gate  QUALIFIES a candidate as actually locomoting. It is a SATURATING gate on
#              tracked_ratio = tracked_speed / commanded_speed (fraction of the commanded
#              velocity actually achieved, in the commanded direction -- omnidirectional,
#              works for fwd/back/strafe alike). gate=0 for a statue, ramps up fast, and
#              hits 1.0 once locomotion is genuine (>= GATE_RATIO of the command). Because
#              it saturates, every real walker has gate~=1, so among walkers QUALITY is the
#              sole differentiator -- quality gets the edge -- while a non-walker still gets
#              gate->0 and can never win. (A statue can't be rescued by stability.)
#   quality    "how good is the walk" once walking: tracking accuracy + uprightness +
#              survival, TILTED toward stability (low rocking) and smoothness (low base
#              accel). Tracking is kept in the mix so a super-stable non-tracker can't beat
#              a crisp walker ("you can only get so far on stability alone").
TRACK_SCALE = 0.35   # maps a velocity-tracking error to goodness via exp(-err/scale)

# saturating walk_gate: gate = clamp(tracked_ratio / GATE_RATIO, 0, 1).
GATE_RATIO = 0.40    # achieving >=40% of commanded speed (in-direction) => fully "walking"

# scales for the "how smooth / how stable" quality components (exp(-x/scale)).
ACCEL_RMS_SCALE = 6.0   # RMS horizontal base lin-accel (m/s^2); ~smoothness
ROCK_RMS_SCALE = 1.0    # RMS roll/pitch base ang-vel (rad/s); ~rocking

# stability-tilted (stability + smoothness + upright = 0.55) but tracking kept (0.30).
QUALITY_WEIGHTS = {
    "track_xy":   0.22,  # exp(-error_vel_xy / TRACK_SCALE)  -- follows the linear command
    "track_yaw":  0.08,  # exp(-error_vel_yaw / TRACK_SCALE) -- follows the turn command
    "upright":    0.22,  # 1 - fall_rate (Episode_Termination/base_orientation)
    "stability":  0.20,  # exp(-rocking_rms / ROCK_RMS_SCALE)      -- low roll/pitch rock
    "smoothness": 0.13,  # exp(-base_accel_rms / ACCEL_RMS_SCALE)  -- smooth IMU X/Y
    "survive":    0.15,  # mean_episode_length / max_ep_steps
}

# TB scalar tags (confirmed present in a real biped run)
TAG_MEAN_REWARD = "Train/mean_reward"          # inner plateau signal (100-ep rolling mean)
TAG_MEAN_EP_LEN = "Train/mean_episode_length"
TAG_SUCCESS = "Metrics/success_rate"
TAG_ERR_XY = "Metrics/base_velocity/error_vel_xy"
TAG_ERR_YAW = "Metrics/base_velocity/error_vel_yaw"
TAG_TERM_ORIENT = "Episode_Termination/base_orientation"
TAG_TERM_HEIGHT = "Episode_Termination/base_height"
# readback metrics from WalkMetricsVelocityCommand (mdp/commands.py)
TAG_TRACKED_SPEED = "Metrics/base_velocity/tracked_speed"     # speed in commanded direction
TAG_CMD_SPEED = "Metrics/base_velocity/commanded_speed"       # mean commanded speed (denom)
TAG_FWD_SPEED = "Metrics/base_velocity/forward_speed"    # plain fwd readback
TAG_ACCEL_RMS = "Metrics/base_velocity/base_accel_rms"   # smoothness
TAG_ROCK_RMS = "Metrics/base_velocity/rocking_rms"       # rocking / stability

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
    seed_best: bool = False        # seed gen 0 from best_out (skip retrain), evolve from it
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
    ap.add_argument("--seed-best", action="store_true",
                    help="seed gen 0 from --best-out (re-grade its run dir, no retrain) and "
                         "evolve from that known-good walker")
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
        poll_secs=a.poll_secs, dry_run=a.dry_run, seed_best=a.seed_best,
        log=a.log, best_out=a.best_out, verbose=not a.quiet,
    )


def hydra_override_str(weights: dict[str, float]) -> str:
    """The ready-to-paste Hydra CLI override that reproduces a weight set."""
    return " ".join(f"env.rewards.{t}.weight={weights[t]:.6g}" for t in TERM_NAMES)
