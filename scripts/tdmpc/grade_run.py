"""Honest, reward-hack-proof fitness for a TD-MPC2 walk checkpoint.

The training return is gameable (a fallen robot can farm shaped terms — see EXPERIMENTS.md
runs 4 & 6). This grader instead reads ONLY out-of-band, weight-INVARIANT signals from a
headless rollout (`eval_smoothness.py`): body-frame forward speed (nets to ~0 for a robot
rocking in place), fall rate, and episode length. So "the math points to a walk when it just
falls over" cannot pass — walk_gate collapses to ~0 for anything that isn't genuinely
travelling forward while upright.

    fitness = walk_gate * quality        (mirrors eureka/evaluate.py::score, [0,1])
      walk_gate = saturating(forward_speed_mean / cmd_vx / GATE_RATIO)   # anti-hack gate
      quality   = w_up*upright + w_sv*survive + w_sm*smooth              # once it walks

Two entry points:
  - score(metrics, cmd_vx) -> (fitness, components): pure, no Isaac. Unit-testable.
  - CLI: runs eval_smoothness.py in a fresh Isaac process, then scores + writes grade.json.

The supervisor calls the CLI as a subprocess and reads grade.json.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys

# --- fitness constants (tunable; kept here so both the grader and the advisor read one source) ---
GATE_RATIO = 0.40          # forward-speed fraction of command that counts as "genuinely walking"
QW_UPRIGHT = 0.50          # quality weights (sum ~1)
QW_SURVIVE = 0.30
QW_SMOOTH = 0.20
FALL_SCALE = 3.0           # exp(-fall_rate_per_min / FALL_SCALE): 1/min->0.72, 3->0.37, 6->0.14
SURVIVE_TARGET_S = 20.0    # mean_episode_len_s that saturates the survive term
ACCEL_RMS_SCALE = 6.0      # exp(-base_accel_rms / ACCEL_RMS_SCALE): smoothness (secondary)

# --- success bar: a run must clear ALL of these to count as a genuine "win" ---
WIN_FWD_SPEED = 0.25       # m/s, body-frame forward
WIN_FALL_RATE = 3.0        # per minute, max
WIN_EP_LEN_S = 10.0        # s, min mean episode length
WIN_FALL_PCT_EP = 10.0     # %, max share of EPISODES containing a real (tilt>45 deg) fall. The old
                           # `fall_rate_per_min` counts only hard_collapse (base 0.30 m below
                           # standing), which the TD-MPC2 walk env almost never trips: measured
                           # 2026-08-13, the 07-24 "0 falls, is_win=True" winner actually falls in
                           # 30.2% of episodes and spends 24.8% of its time past 45 deg. The PPO demo
                           # it was seeded from manages 6.2%. Runs evaluated before the metric
                           # existed have no `pct_episodes_with_fall` and are not judged on it.
WIN_LEAN_DEG = 15.0        # deg, max mean torso lean MAGNITUDE (see abs() below). The 07-24 winners
                           # cleared every other bar while walking bent well forward: the gated
                           # reward's upright term is flat to ~50 deg (upright_min .8 / margin .8),
                           # so nothing in the objective saw it. Runs graded before the posture
                           # metric existed have no `torso_lean_fwd_deg` and are not judged on it.

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
VENV_PY = os.path.join(REPO_ROOT, ".venv", "bin", "python")
EVAL = os.path.join("scripts", "tdmpc", "eval_smoothness.py")
DEFAULT_TASK = "Walk-Humanoid-Policy-Biped-Tdmpc-v0"


# ============================ WALK OBJECTIVE v2 (2026-08-13) ============================
# The v1 fitness below is DEAD as a ranking signal. Measured on the five walkers this project
# has produced, re-graded at H=6: it spans 0.9256-0.9351 -- a range of 0.0095 -- across
# policies whose real fall rates differ by 2x (14.8-28.9%) and whose torso tilt differs by
# 2.7x (9.1-24.7 deg). Three structural reasons:
#   * `upright` reads `fall_rate_per_min`, the hard_collapse-only metric, which is 0.00 for
#     every policy including ones that fall in 30% of episodes.
#   * `survive` is ep_len_s/20 and 20 s IS the max episode length -> pinned at 1.0 always.
#   * `walk_gate` saturates at 0.4*cmd = 0.12 m/s and has no upper bound, so a policy running
#     1.5x the commanded speed scores identically to one tracking it exactly.
# v2 keeps the repo's established shape -- multiplicative for must-haves so they cannot be
# out-earned, additive for nice-to-haves so no single one zeroes the score -- but on signals
# that can actually see the failures we care about:
#     fitness = track * safety * (QW2_POSTURE*posture + QW2_SMOOTH*smooth)
# `track` uses tracking ERROR, not raw speed, so overshoot is penalized (H=6 pushed lean-B to
# 0.452 against a 0.300 command and v1 called that a perfect walk_gate of 1.0).
# `safety` is the real, tilt-based episode fall rate and is deliberately the steepest term.
TRACK_ERR_SCALE = 0.10     # m/s: exp(-|fwd-cmd|/.) -> 1.0 exact, 0.22 at 1.5x a 0.3 command
FALL_PCT_SCALE = 10.0      # %:   exp(-pct_eps_fall/.) -> PPO's 6.2% -> 0.54, 18.8% -> 0.15, 30% -> 0.05
TILT_SCALE = 15.0          # deg: exp(-mean_tilt/.) -> 9 deg -> 0.55, 22 deg -> 0.23
QW2_POSTURE = 0.6
QW2_SMOOTH = 0.4
MOVING_FLOOR = 0.05        # m/s: below this it is not locomoting at all -> fitness 0, no partial credit


def _score_v2(metrics: dict, cmd_vx: float) -> tuple[float, dict]:
    """Walk objective v2 — needs the real (tilt-based) fall metric. See the block comment above."""
    fwd = float(metrics.get("forward_speed_mean", 0.0))
    fall_pct = float(metrics.get("pct_episodes_with_fall", 100.0))
    tilt = float(metrics.get("torso_tilt_deg", 90.0))
    accel_rms = float(metrics.get("base_accel_rms", 1e6))
    lean = metrics.get("torso_lean_fwd_deg")
    lean = float(lean) if lean is not None else None

    track = math.exp(-abs(fwd - cmd_vx) / TRACK_ERR_SCALE)
    safety = math.exp(-max(0.0, fall_pct) / FALL_PCT_SCALE)
    posture = math.exp(-max(0.0, tilt) / TILT_SCALE)
    smooth = math.exp(-max(0.0, accel_rms) / ACCEL_RMS_SCALE)
    quality = QW2_POSTURE * posture + QW2_SMOOTH * smooth
    fitness = 0.0 if fwd < MOVING_FLOOR else track * safety * quality

    ep_len_s = float(metrics.get("mean_episode_len_s", 0.0))
    if not math.isfinite(ep_len_s):
        ep_len_s = SURVIVE_TARGET_S
    is_win = (fwd >= WIN_FWD_SPEED and ep_len_s >= WIN_EP_LEN_S
              and fall_pct <= WIN_FALL_PCT_EP
              and (lean is None or abs(lean) <= WIN_LEAN_DEG))
    comp = {
        "objective_version": "walk_v2",
        "fitness": round(fitness, 4),
        "track": round(track, 4), "safety": round(safety, 4),
        "posture": round(posture, 4), "smooth": round(smooth, 4), "quality": round(quality, 4),
        "tracked_ratio": round(fwd / cmd_vx, 4) if cmd_vx > 1e-3 else 0.0,
        "is_win": bool(is_win),
        "_raw": {
            "forward_speed_mean": round(fwd, 4),
            "pct_episodes_with_fall": round(fall_pct, 2),
            "real_falls_per_min": (round(float(metrics["real_falls_per_min"]), 3)
                                   if "real_falls_per_min" in metrics else None),
            "torso_lean_fwd_deg": round(lean, 2) if lean is not None else None,
            "torso_tilt_deg": round(tilt, 2),
            "base_accel_rms": round(accel_rms, 4) if accel_rms < 1e6 else "inf",
            "mean_episode_len_s": round(ep_len_s, 2),
            "legacy_fall_rate_per_min": metrics.get("fall_rate_per_min"),
            "plan_horizon": metrics.get("plan_horizon"),
            "cmd_vx": cmd_vx,
        },
    }
    return float(fitness), comp


def score(metrics: dict, cmd_vx: float) -> tuple[float, dict]:
    """Dispatch: v2 when the real fall metric is present, else the frozen v1 below.

    Evals produced before 2026-08-13 have no `pct_episodes_with_fall`, so they keep scoring
    under v1 and every historical grade in the journal stays reproducible. v1 and v2 numbers
    are NOT comparable — check `objective_version`.
    """
    if metrics.get("pct_episodes_with_fall") is not None:
        return _score_v2(metrics, cmd_vx)
    return _score_legacy(metrics, cmd_vx)


def _score_legacy(metrics: dict, cmd_vx: float) -> tuple[float, dict]:
    """(fitness, components) from an eval_smoothness.py metrics dict. Pure — no Isaac, no I/O."""
    fwd = float(metrics.get("forward_speed_mean", 0.0))
    fall_rate = float(metrics.get("fall_rate_per_min", 1e6))
    ep_len_s = float(metrics.get("mean_episode_len_s", 0.0))
    accel_rms = float(metrics.get("base_accel_rms", 1e6))
    if not math.isfinite(ep_len_s):          # inf = never fell during eval -> full survival credit
        ep_len_s = SURVIVE_TARGET_S
    if not math.isfinite(fall_rate):
        fall_rate = 1e6

    # walk_gate: 0 for a statue/rocker (fwd~0) or backward motion; saturates to 1 once fwd speed
    # reaches GATE_RATIO * command in the commanded direction. This is the un-fakeable gate.
    tracked_ratio = (fwd / cmd_vx) if cmd_vx > 1e-3 else 0.0
    walk_gate = max(0.0, min(1.0, tracked_ratio / GATE_RATIO))

    comp = {
        "upright": math.exp(-max(0.0, fall_rate) / FALL_SCALE),
        "survive": max(0.0, min(1.0, ep_len_s / SURVIVE_TARGET_S)),
        "smooth": math.exp(-max(0.0, accel_rms) / ACCEL_RMS_SCALE),
    }
    quality = QW_UPRIGHT * comp["upright"] + QW_SURVIVE * comp["survive"] + QW_SMOOTH * comp["smooth"]
    fitness = walk_gate * quality

    # posture: absent on pre-instrumentation eval JSONs -> not judged (None), never silently passed
    lean = metrics.get("torso_lean_fwd_deg")
    lean = float(lean) if lean is not None else None
    # abs(): the bar is on tilt MAGNITUDE, not forward lean. lean-A-firm walked at -15 deg (leaning
    # BACKWARD) and would have passed a one-sided `lean <= 15` check. The upright gate constrains
    # cos(tilt), which is direction-agnostic, so the policy is free to trade forward lean for
    # backward lean at identical reward -- and it did, within 200k steps.
    fall_pct = metrics.get("pct_episodes_with_fall")
    fall_pct = float(fall_pct) if fall_pct is not None else None
    is_win = (fwd >= WIN_FWD_SPEED and fall_rate <= WIN_FALL_RATE and ep_len_s >= WIN_EP_LEN_S
              and (lean is None or abs(lean) <= WIN_LEAN_DEG)
              and (fall_pct is None or fall_pct <= WIN_FALL_PCT_EP))
    comp.update({
        "objective_version": "walk_v1_legacy",
        "fitness": round(fitness, 4),
        "quality": round(quality, 4),
        "walk_gate": round(walk_gate, 4),
        "tracked_ratio": round(tracked_ratio, 4),
        "is_win": bool(is_win),
        "_raw": {
            "forward_speed_mean": round(fwd, 4),
            "fall_rate_per_min": round(fall_rate, 4) if fall_rate < 1e6 else "inf",
            "mean_episode_len_s": round(ep_len_s, 2),
            "base_accel_rms": round(accel_rms, 4) if accel_rms < 1e6 else "inf",
            "pct_episodes_with_fall": fall_pct,
            "real_falls_per_min": (round(float(metrics["real_falls_per_min"]), 3)
                                   if "real_falls_per_min" in metrics else None),
            "torso_lean_fwd_deg": round(lean, 2) if lean is not None else None,
            "torso_tilt_deg": round(float(metrics["torso_tilt_deg"]), 2) if "torso_tilt_deg" in metrics else None,
            "cmd_vx": cmd_vx,
        },
    })
    return float(fitness), comp


# --- STABILITY objective: track the command vector + keep the always-off-command IMU quiet ---
TRK_LIN_SCALE = 0.15    # exp(-track_err_lin / .): matching commanded (vx,vy) within ~0.15 m/s
TRK_ANG_SCALE = 0.30    # exp(-track_err_ang / .): matching commanded yaw-rate within ~0.3 rad/s
VERT_SCALE = 0.05       # exp(-|v_z| / .): vertical bob
ROCK_SCALE = 0.30       # exp(-rocking_rms / .): torso roll/pitch gyro
SACCEL_SCALE = 6.0      # exp(-base_accel_rms / .): IMU x/y jerk
# stability win bar
SWIN_TRK_LIN = 0.12     # m/s command-tracking error, max
SWIN_ROCK = 0.25        # rocking_rms, max
SWIN_VERT = 0.04        # |v_z|, max
SWIN_FALL = 3.0         # falls/min, max


def score_stability(metrics: dict) -> tuple[float, dict]:
    """(fitness, components) for the OMNIDIRECTIONAL stability objective. fitness =
    tracking_score * quietness_score, both in [0,1]. tracking_score gates 'follows the command'
    (a stand-still robot scores ~0 when commanded to move); quietness rewards a clean IMU (no
    vertical bob / rocking / jerk). Read from an eval run with --varied_commands."""
    trk_lin = float(metrics.get("track_err_lin_mean", 1e6))
    trk_ang = float(metrics.get("track_err_ang_mean", 1e6))
    vert = float(metrics.get("vertical_speed_mean", 1e6))
    rock = float(metrics.get("rocking_rms", 1e6))
    accel = float(metrics.get("base_accel_rms", 1e6))
    fall_rate = float(metrics.get("fall_rate_per_min", 1e6))
    if not math.isfinite(fall_rate):
        fall_rate = 1e6

    tracking = math.exp(-trk_lin / TRK_LIN_SCALE) * math.exp(-trk_ang / TRK_ANG_SCALE)
    quietness = (math.exp(-vert / VERT_SCALE) * math.exp(-max(0.0, rock) / ROCK_SCALE)
                 * math.exp(-max(0.0, accel) / SACCEL_SCALE))
    fitness = tracking * quietness
    is_win = (trk_lin <= SWIN_TRK_LIN and rock <= SWIN_ROCK and vert <= SWIN_VERT
              and fall_rate <= SWIN_FALL)
    comp = {
        "fitness": round(fitness, 4), "tracking": round(tracking, 4), "quietness": round(quietness, 4),
        "is_win": bool(is_win),
        "_raw": {"track_err_lin": round(trk_lin, 4), "track_err_ang": round(trk_ang, 4),
                 "vertical_speed": round(vert, 4), "rocking_rms": round(rock, 4),
                 "base_accel_rms": round(accel, 4),
                 "fall_rate_per_min": round(fall_rate, 4) if fall_rate < 1e6 else "inf",
                 "forward_speed": round(float(metrics.get("forward_speed_mean", 0.0)), 4),
                 "commanded_speed": round(float(metrics.get("commanded_speed_mean", 0.0)), 4)},
    }
    return float(fitness), comp


def run_eval(checkpoint: str, task: str, cmd_vx: float, num_envs: int, steps: int,
             plant: str, plan: bool, metrics_out: str, seed: int = 0, varied: bool = False,
             horizon: int | None = None, warm_start: bool = False) -> dict:
    """Launch eval_smoothness.py in a fresh Isaac process; return the parsed metrics dict."""
    cmd = [VENV_PY, EVAL, "--checkpoint", checkpoint, "--task", task,
           "--cmd_vx", str(cmd_vx), "--num_envs", str(num_envs), "--steps", str(steps),
           "--plant", plant, "--seed", str(seed), "--out", metrics_out, "--headless"]
    if varied:
        cmd.append("--varied_commands")
    if horizon is not None:
        cmd += ["--horizon", str(horizon)]
    if warm_start:
        cmd.append("--warm_start")
    if plan:
        cmd.append("--plan")
    env = dict(os.environ, OMNI_KIT_ACCEPT_EULA="YES")
    print(f"[grade] eval: {' '.join(cmd)}")
    subprocess.run(cmd, env=env, cwd=REPO_ROOT, check=True)
    with open(metrics_out) as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser(description="Grade a TD-MPC2 checkpoint with the honest fitness.")
    p.add_argument("--checkpoint", required=True, help="a .pt file, or a run dir (uses model_best.pt).")
    p.add_argument("--task", default=DEFAULT_TASK, help="eval env task id (default: the walk env).")
    p.add_argument("--cmd_vx", type=float, default=0.3)
    p.add_argument("--num_envs", type=int, default=64)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--plant", choices=["baseline", "modeled"], default="modeled")
    p.add_argument("--no-plan", dest="plan", action="store_false", help="score the bare policy prior "
                   "instead of the MPPI planner (default: planner, as deployed).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--metrics", default=None, help="score this existing eval JSON instead of running "
                   "eval (no Isaac needed — for testing / re-grading).")
    p.add_argument("--metrics-out", default=None, help="where run_eval writes the raw eval JSON "
                   "(default: <ckpt-dir>/eval_metrics.json).")
    p.add_argument("--out", default=None, help="where to write the grade JSON "
                   "(default: <ckpt-dir>/grade.json).")
    p.add_argument("--horizon", type=int, default=None,
                   help="MPPI planning horizon for the eval (see eval_smoothness --horizon).")
    p.add_argument("--warm_start", action="store_true", help="MPPI warm start for the eval.")
    p.add_argument("--objective", choices=["walk", "stability"], default="walk",
                   help="'walk' = honest forward-walk fitness (default); 'stability' = omnidirectional "
                        "command-tracking * IMU-quietness (uses --varied_commands eval).")
    args = p.parse_args()

    ckpt_dir = args.checkpoint if os.path.isdir(args.checkpoint) else os.path.dirname(args.checkpoint)
    metrics_out = args.metrics_out or os.path.join(ckpt_dir or ".", "eval_metrics.json")
    grade_out = args.out or os.path.join(ckpt_dir or ".", "grade.json")
    varied = (args.objective == "stability")

    if args.metrics:
        with open(args.metrics) as f:
            metrics = json.load(f)
    else:
        metrics = run_eval(args.checkpoint, args.task, args.cmd_vx, args.num_envs, args.steps,
                           args.plant, args.plan, metrics_out, args.seed, varied=varied,
                           horizon=args.horizon, warm_start=args.warm_start)

    fitness, comp = score_stability(metrics) if args.objective == "stability" else score(metrics, args.cmd_vx)
    grade = {"checkpoint": args.checkpoint, "task": args.task, "objective": args.objective,
             **comp, "metrics": metrics}
    os.makedirs(os.path.dirname(os.path.abspath(grade_out)), exist_ok=True)
    with open(grade_out, "w") as f:
        json.dump(grade, f, indent=2)
    if args.objective == "stability":
        r = comp["_raw"]
        print(f"[grade] STABILITY fitness={fitness:.4f} tracking={comp['tracking']} "
              f"quietness={comp['quietness']} is_win={comp['is_win']} "
              f"(trk_lin={r['track_err_lin']} rock={r['rocking_rms']} vert={r['vertical_speed']} "
              f"fall/min={r['fall_rate_per_min']})")
    else:
        r = comp["_raw"]
        if comp.get("objective_version") == "walk_v2":
            print(f"[grade] fitness={fitness:.4f} ({comp['objective_version']}) is_win={comp['is_win']} "
                  f"| track={comp['track']} safety={comp['safety']} posture={comp['posture']} "
                  f"smooth={comp['smooth']}")
            print(f"[grade]   fwd={r['forward_speed_mean']} ({comp['tracked_ratio']}x cmd) "
                  f"falls={r['pct_episodes_with_fall']}%eps ({r['real_falls_per_min']}/min) "
                  f"lean={r['torso_lean_fwd_deg']}deg tilt={r['torso_tilt_deg']}deg H={r['plan_horizon']}")
        else:
            print(f"[grade] fitness={fitness:.4f} (legacy) walk_gate={comp['walk_gate']} "
                  f"is_win={comp['is_win']} (fwd={r['forward_speed_mean']} "
                  f"fall/min={r['fall_rate_per_min']} ep_len_s={r['mean_episode_len_s']})")
    print(f"[grade] wrote {grade_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
