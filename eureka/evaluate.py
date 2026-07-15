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
    max_steps = _max_ep_steps(run_dir)

    comp = {
        "success":   max(0.0, min(1.0, success)),
        "upright":   max(0.0, 1.0 - fall),
        "survive":   max(0.0, min(1.0, ep_len / max_steps)),
        "track_xy":  math.exp(-err_xy / C.TRACK_SCALE),
        "track_yaw": math.exp(-err_yaw / C.TRACK_SCALE),
        "standing":  max(0.0, 1.0 - collapse),
    }
    fitness = sum(C.FITNESS_WEIGHTS[k] * comp[k] for k in C.FITNESS_WEIGHTS)
    comp["_raw"] = {
        "success_rate": round(success, 4), "err_vel_xy": round(err_xy, 4),
        "err_vel_yaw": round(err_yaw, 4), "fall_rate": round(fall, 4),
        "collapse_rate": round(collapse, 4), "mean_ep_len": round(ep_len, 1),
        "max_ep_steps": max_steps,
    }
    return float(fitness), comp
