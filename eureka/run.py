"""Run ONE candidate = a bounded training run with injected reward weights, with
plateau-based early-stop. Two backends behind `run_candidate(weights, cfg)`:

  - subprocess (default, ships first): one `train.py` per candidate, weights via
    Hydra CLI overrides, monitored + killed by an external poll loop. Zero trainer
    changes. Pays ~30 min Isaac startup PER candidate.
  - persistent (run_persistent.py): one long-lived Isaac process; startup paid once.
    Added second, gated on a cross-check against subprocess.
"""

from __future__ import annotations

import glob
import os
import signal
import subprocess
import time

from . import config as C
from .tb_utils import plateaued, read_scalars, scalar_series

VENV_PY = ".venv/bin/python"          # the repo venv that boots Isaac
TRAIN = "scripts/rsl_rl/train.py"


def _weight_overrides(weights: dict[str, float]) -> list[str]:
    return [f"env.rewards.{t}.weight={weights[t]:.6g}" for t in C.TERM_NAMES]


def _experiment_dir(variant: str) -> str:
    return os.path.join("logs", "rsl_rl", C.VARIANT_EXPERIMENT[variant])


def _find_run_dir(variant: str, run_name: str, since: float) -> str | None:
    """The run dir train.py just created: `<ts>_<run_name>/` newer than launch."""
    pat = os.path.join(_experiment_dir(variant), f"*_{run_name}")
    cands = [d for d in glob.glob(pat) if os.path.isdir(d) and os.path.getmtime(d) >= since - 5]
    return sorted(cands, key=os.path.getmtime)[-1] if cands else None


def run_candidate_subprocess(weights: dict[str, float], cfg: C.SearchConfig) -> tuple[str | None, str]:
    cmd = [VENV_PY, TRAIN, "--variant", cfg.variant, "--profile", cfg.profile,
           "--headless", "--seed", str(cfg.seed), "--run_name", cfg.run_name,
           "--max_iterations", str(cfg.max_iterations)] + _weight_overrides(weights)
    env = dict(os.environ, OMNI_KIT_ACCEPT_EULA="YES")
    t0 = time.time()
    proc = subprocess.Popen(cmd, env=env, start_new_session=True)
    run_dir, reason = None, "process exited"
    try:
        while True:
            if proc.poll() is not None:
                reason = "training finished"
                break
            if run_dir is None:
                run_dir = _find_run_dir(cfg.variant, cfg.run_name, t0)
            else:
                curve = scalar_series(read_scalars(run_dir), C.TAG_MEAN_REWARD)
                if curve and plateaued(curve, cfg.inner_patience, cfg.inner_min_delta):
                    reason = f"plateau at iter {len(curve)}"
                    break
                if len(curve) >= cfg.max_iterations:
                    reason = "reached max_iterations"
                    break
            time.sleep(cfg.poll_secs)
    finally:
        _terminate(proc)
    if run_dir is None:                       # last-chance discovery after exit
        run_dir = _find_run_dir(cfg.variant, cfg.run_name, t0)
    return run_dir, reason


def _terminate(proc: subprocess.Popen) -> None:
    """SIGINT (lets simulation_app.close() run) -> SIGTERM -> SIGKILL on the group."""
    if proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
        for sig, wait in ((signal.SIGINT, 60), (signal.SIGTERM, 30), (signal.SIGKILL, 5)):
            os.killpg(pgid, sig)
            for _ in range(wait):
                if proc.poll() is not None:
                    return
                time.sleep(1)
    except ProcessLookupError:
        pass


def run_candidate(weights: dict[str, float], cfg: C.SearchConfig) -> tuple[str | None, str]:
    if cfg.backend == "persistent":
        from .run_persistent import run_candidate_persistent
        return run_candidate_persistent(weights, cfg)
    return run_candidate_subprocess(weights, cfg)
