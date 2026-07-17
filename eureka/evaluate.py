"""Ground-truth fitness for a walk run, read from its TensorBoard.

Every input is weight-INVARIANT (logged by the trainer independent of the reward
weights), so the search cannot game its own fitness by inflating a weight. Returns
a scalar in ~[0,1] plus a component breakdown that the proposer reflects on.
"""

from __future__ import annotations

import math
import os

from . import config as C
from .tb_utils import last_scalar, read_scalars


def _max_ep_steps(run_dir: str) -> int:
    """episode_length_s / (sim.dt * decimation) from the run's dumped env.yaml."""
    path = os.path.join(run_dir, "params", "env.yaml")
    try:
        import yaml
        with open(path) as f:
            cfg = yaml.unsafe_load(f) if hasattr(yaml, "unsafe_load") else yaml.safe_load(f)
        els = float(cfg["episode_length_s"])
        dec = float(cfg["decimation"])
        dt = float(cfg["sim"]["dt"])
        return max(1, round(els / (dt * dec)))
    except Exception:
        return C.DEFAULT_MAX_EP_STEPS


def score(run_dir: str) -> tuple[float, dict]:
    """Return (fitness, components). fitness = -inf if the run produced no TB data."""
    sc = read_scalars(run_dir)
    if not sc:
        return float("-inf"), {"error": "no TB data in run dir"}

    success = float(last_scalar(sc, C.TAG_SUCCESS, 0.0))
    err_xy = float(last_scalar(sc, C.TAG_ERR_XY, 10.0))
    err_yaw = float(last_scalar(sc, C.TAG_ERR_YAW, 10.0))
    fall = float(last_scalar(sc, C.TAG_TERM_ORIENT, 1.0))
    collapse = float(last_scalar(sc, C.TAG_TERM_HEIGHT, 1.0))
    ep_len = float(last_scalar(sc, C.TAG_MEAN_EP_LEN, 0.0))
    # readback metrics (default to "not walking" so pre-metrics runs gate to ~0)
    tracked = float(last_scalar(sc, C.TAG_TRACKED_SPEED, 0.0))
    cmd_speed = float(last_scalar(sc, C.TAG_CMD_SPEED, 0.0))
    fwd_speed = float(last_scalar(sc, C.TAG_FWD_SPEED, 0.0))
    accel_rms = float(last_scalar(sc, C.TAG_ACCEL_RMS, 1e3))
    rock_rms = float(last_scalar(sc, C.TAG_ROCK_RMS, 1e3))
    max_steps = _max_ep_steps(run_dir)

    # quality: how good the walk is *once it is walking* (weight-invariant [0,1] parts)
    comp = {
        "track_xy":   math.exp(-err_xy / C.TRACK_SCALE),
        "track_yaw":  math.exp(-err_yaw / C.TRACK_SCALE),
        "upright":    max(0.0, 1.0 - fall),
        "stability":  math.exp(-max(0.0, rock_rms) / C.ROCK_RMS_SCALE),
        "smoothness": math.exp(-max(0.0, accel_rms) / C.ACCEL_RMS_SCALE),
        "survive":    max(0.0, min(1.0, ep_len / max_steps)),
    }
    quality = sum(C.QUALITY_WEIGHTS[k] * comp[k] for k in C.QUALITY_WEIGHTS)

    # walk_gate: SATURATING gate on tracked_ratio (fraction of commanded speed achieved in
    # the commanded direction). Omnidirectional; ~0 for a statue, 1.0 once it genuinely
    # locomotes (>= GATE_RATIO). Saturating => among walkers gate~=1 so quality decides.
    tracked_ratio = tracked / cmd_speed if cmd_speed > 1e-3 else 0.0
    walk_gate = max(0.0, min(1.0, tracked_ratio / C.GATE_RATIO))
    fitness = walk_gate * quality

    comp["quality"] = round(quality, 4)
    comp["walk_gate"] = round(walk_gate, 4)
    comp["tracked_ratio"] = round(tracked_ratio, 4)
    comp["_raw"] = {
        "success_rate": round(success, 4), "err_vel_xy": round(err_xy, 4),
        "err_vel_yaw": round(err_yaw, 4), "fall_rate": round(fall, 4),
        "collapse_rate": round(collapse, 4), "mean_ep_len": round(ep_len, 1),
        "max_ep_steps": max_steps, "tracked_speed": round(tracked, 4),
        "commanded_speed": round(cmd_speed, 4), "forward_speed": round(fwd_speed, 4),
        "accel_rms": round(accel_rms, 4), "rocking_rms": round(rock_rms, 4),
    }
    return float(fitness), comp
