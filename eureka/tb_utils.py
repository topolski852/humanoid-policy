"""TensorBoard event reading + the shared plateau detector.

Used at both levels: the inner loop polls `Train/mean_reward` to early-stop a
training run; the outer loop uses the same `plateaued()` on per-generation best
fitness to stop the search.
"""

from __future__ import annotations

import glob
import os


def latest_event_file(run_dir: str) -> str | None:
    ev = sorted(glob.glob(os.path.join(run_dir, "events.out.tfevents.*")))
    return ev[-1] if ev else None


def read_scalars(run_dir: str) -> dict[str, list[tuple[int, float]]]:
    """{tag: [(step, value), ...]} for every scalar in the run's newest event file.

    Lazily imports tensorboard so nothing else in the package depends on it.
    Re-reads on each call (cheap) so it works for live polling of a running job.
    """
    ef = latest_event_file(run_dir)
    if ef is None:
        return {}
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(ef, size_guidance={"scalars": 0})
    ea.Reload()
    out = {}
    for tag in ea.Tags().get("scalars", []):
        out[tag] = [(s.step, s.value) for s in ea.Scalars(tag)]
    return out


def scalar_series(scalars: dict, tag: str) -> list[float]:
    return [v for _, v in scalars.get(tag, [])]


def last_scalar(scalars: dict, tag: str, default=None):
    s = scalars.get(tag)
    return s[-1][1] if s else default


def iters_since_best(curve: list[float], min_delta: float) -> tuple[int, float]:
    """(#points since the last value that beat the running best by min_delta, best)."""
    best, best_i = float("-inf"), -1
    for i, v in enumerate(curve):
        if v > best + min_delta:
            best, best_i = v, i
    return (len(curve) - 1 - best_i if best_i >= 0 else 0), best


def plateaued(curve: list[float], patience: int, min_delta: float) -> bool:
    """True once the best hasn't improved by min_delta in the last `patience` points."""
    if patience <= 0 or len(curve) < 2:
        return False
    stale, _ = iters_since_best(curve, min_delta)
    return stale >= patience
