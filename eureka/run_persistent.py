"""Persistent-process backend — the cost win (pays Isaac startup ONCE per search).

STATUS: scaffold. The subprocess backend (run.py) is the correctness baseline that
ships first; this is the staged second step, to be built + cross-checked against it.

Design (verified feasible during planning):
  * Build AppLauncher + the walk env + an rsl_rl OnPolicyRunner ONCE, in a long-lived
    process (mirror scripts/rsl_rl/train.py's bootstrap; reuse the tuner's
    IsaacSingleJoint "keep one app alive" pattern).
  * Per candidate, mutate the live reward-manager weights:
        rm = env.unwrapped.reward_manager
        for name, w in weights.items():
            cfg_ = rm.get_term_cfg(name); cfg_.weight = float(w); rm.set_term_cfg(name, cfg_)
    (set_term_cfg syncs the on-device weight tensor, so this takes effect immediately.)
  * Rebuild a FRESH OnPolicyRunner on the same env and reseed so each candidate trains
    from scratch (fitness must be attributable to the weights, not a warm start);
    reset the env / episode-sum buffers.
  * Train to plateau/cap (reuse the tb_utils.plateaued logic on the runner's reward
    log), then hand the run dir to evaluate.score() exactly like the subprocess path.

GATE before trusting this overnight: run 1-2 candidates through BOTH backends with the
same seed+weights; fitness must match within noise.
"""

from __future__ import annotations

from . import config as C


def run_candidate_persistent(weights: dict[str, float], cfg: C.SearchConfig):
    raise NotImplementedError(
        "persistent backend not built yet — use --backend subprocess (the correctness "
        "baseline). See this module's docstring for the staged plan.")
