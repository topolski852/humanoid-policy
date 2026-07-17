"""Plateau watcher for rsl_rl on-policy training.

Guards against the classic PPO failure mode where the smoothed reward peaks and then
*degrades* if training keeps running (policy collapse / over-fitting to stale data). Two
protections, independent of each other:

- **best-checkpoint tracking** — whenever the smoothed metric hits a new high we write
  ``model_best.pt`` alongside the numbered checkpoints. Even a fixed-length run that later
  degrades still has its peak policy on disk (see play.py, which prefers it at export time).
- **early stop** — after ``patience`` iterations with no meaningful improvement we stop the
  run, so a high ``--max_iterations`` ceiling costs nothing once the policy has converged.

The rsl_rl ``OnPolicyRunner.learn`` loop is monolithic (no per-iteration callback), but it
calls ``self.logger.log(it=..., ...)`` exactly once per iteration. ``PlateauRunner`` wraps
that method instead of duplicating the ~50-line loop, so it stays correct if the upstream
loop changes.
"""

from __future__ import annotations

import os
import statistics

from rsl_rl.runners import OnPolicyRunner


# smoothed rsl_rl logger buffers a watched metric can read (both are deque(maxlen=100))
_METRIC_BUFFERS = {
    "mean_reward": "rewbuffer",
    "mean_episode_length": "lenbuffer",
}


class _PlateauStop(Exception):
    """Sentinel used to break out of the monolithic ``learn`` loop from within the log hook."""


class PlateauStopper:
    """Tracks the best smoothed metric and decides when a run has plateaued.

    ``min_delta`` is a *relative* improvement fraction (0.005 = 0.5%), so the same config works
    across tasks with different reward scales. ``best`` is tracked from iteration 0 so
    ``model_best.pt`` is always available; the ``warmup`` gate only suppresses *early stopping*
    (early reward is noisy and would trip the patience counter spuriously).
    """

    def __init__(self, patience: int, min_delta: float, warmup: int, metric: str = "mean_reward"):
        self.patience = patience
        self.min_delta = min_delta
        self.warmup = warmup
        self.metric = metric
        self.best = float("-inf")
        self.best_it = -1
        self.since_improved = 0
        self.stopped = False

    def update(self, it: int, value: float) -> tuple[bool, bool]:
        """Feed one iteration's smoothed metric. Returns ``(new_best, should_stop)``."""
        # relative threshold; abs() so it behaves sanely when best is negative (common for reward)
        threshold = self.best + abs(self.best) * self.min_delta if self.best != float("-inf") else float("-inf")
        new_best = value > threshold
        if new_best:
            self.best = value
            self.best_it = it
            self.since_improved = 0
        elif it >= self.warmup:
            # only count stagnation once past warmup — early training is too noisy to judge
            self.since_improved += 1

        should_stop = it >= self.warmup and self.since_improved >= self.patience
        self.stopped = should_stop
        return new_best, should_stop


class PlateauRunner(OnPolicyRunner):
    """``OnPolicyRunner`` that saves ``model_best.pt`` and early-stops on a reward plateau."""

    def configure_plateau(self, stopper: PlateauStopper) -> None:
        self._stopper = stopper

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        stopper = getattr(self, "_stopper", None)
        if stopper is None or self.logger.log_dir is None:
            # no watcher configured (or no logging dir) -> behave exactly like the base runner
            return super().learn(num_learning_iterations, init_at_random_ep_len)

        buffer_name = _METRIC_BUFFERS[stopper.metric]
        best_path = os.path.join(self.logger.log_dir, "model_best.pt")
        original_log = self.logger.log

        def logging_hook(*args, **kwargs):
            original_log(*args, **kwargs)
            it = kwargs.get("it", args[0] if args else self.current_learning_iteration)
            buffer = getattr(self.logger, buffer_name, None)
            if not buffer:  # empty until the first episodes terminate
                return
            value = statistics.mean(buffer)
            new_best, should_stop = stopper.update(it, value)
            if new_best:
                self.save(best_path)
            if should_stop:
                raise _PlateauStop()

        self.logger.log = logging_hook
        try:
            super().learn(num_learning_iterations, init_at_random_ep_len)
        except _PlateauStop:
            # the base loop's post-loop save + writer close are skipped when we break via exception,
            # so replicate them here for the final checkpoint.
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))
            self.logger.stop_logging_writer()
            print(
                f"\n[PLATEAU] Stopped early at iter {self.current_learning_iteration}: "
                f"{stopper.metric} plateaued for {stopper.patience} iters "
                f"(best={stopper.best:.3f} @ iter {stopper.best_it}). "
                f"Best policy saved to model_best.pt."
            )
        else:
            best = stopper.best if stopper.best != float("-inf") else float("nan")
            print(
                f"\n[PLATEAU] Reached the iteration ceiling without a plateau. "
                f"Best {stopper.metric}={best:.3f} @ iter {stopper.best_it} (saved to model_best.pt)."
            )
        finally:
            self.logger.log = original_log  # restore so a subsequent learn() isn't double-wrapped


def add_plateau_args(parser) -> None:
    """Add the ``--plateau`` early-stop / best-checkpoint arguments to an argparse parser."""
    group = parser.add_argument_group("plateau", description="Reward-plateau watcher (best-ckpt + early stop).")
    group.add_argument(
        "--plateau",
        action="store_true",
        default=False,
        help="Enable the plateau watcher: save model_best.pt on new highs and stop when reward plateaus. "
        "Off by default (existing runs are unchanged). Recommended for 'full' runs with a high --max_iterations.",
    )
    group.add_argument(
        "--plateau-patience",
        type=int,
        default=800,
        help="Iterations with no meaningful improvement before stopping. Default 800 (conservative).",
    )
    group.add_argument(
        "--plateau-min-delta",
        type=float,
        default=0.005,
        help="Relative improvement (fraction) that counts as progress. Default 0.005 (0.5%%).",
    )
    group.add_argument(
        "--plateau-warmup",
        type=int,
        default=500,
        help="Iterations before early-stopping is allowed (best-ckpt tracking still runs). Default 500.",
    )
    group.add_argument(
        "--plateau-metric",
        type=str,
        default="mean_reward",
        choices=sorted(_METRIC_BUFFERS.keys()),
        help="Which smoothed metric to watch. Default mean_reward.",
    )


def build_stopper(args_cli) -> PlateauStopper | None:
    """Construct a :class:`PlateauStopper` from parsed CLI args, or ``None`` if ``--plateau`` is off."""
    if not getattr(args_cli, "plateau", False):
        return None
    return PlateauStopper(
        patience=args_cli.plateau_patience,
        min_delta=args_cli.plateau_min_delta,
        warmup=args_cli.plateau_warmup,
        metric=args_cli.plateau_metric,
    )
