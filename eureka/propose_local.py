"""API-free evolutionary proposer over the reward-term WEIGHT vector.

Ported from humanoid-tuner/policy/reward_search/propose_local.py, adapted from a
variable-length code genome to the fixed-length, sign-constrained weight vector of
the walk RewardsCfg.

Design (walk-biped): the walk optimum is FRAGILE — random reweighting collapses the
policy back to standing, and the gated fitness scores every non-walker exactly 0, so
the landscape outside the walk basin is flat and gives the search no signal. So:

  * Evolve ONLY from candidates that actually walk (fitness >= WALKER_FITNESS_FLOOR).
    Mutating a statue is wasted compute.
  * PROTECT the load-bearing walk terms (track_* + termination_penalty): never pruned,
    jitter little, clamped near the parent so candidates stay in the walk basin. Even
    "random exploration" candidates keep these at their walking values, so they still
    walk and produce informative fitness instead of yet another statue.
  * ANNEAL the step: moderate exploration early, focusing later (sigma decays per
    generation), like an RL exploration schedule. Signs are fixed (config.SIGNS).

The result: exploration happens where it's informative — the stability / smoothness /
safety terms — while the walk itself is preserved, and selection focuses on the best
walkers over generations.
"""

from __future__ import annotations

import numpy as np

from .config import DEFAULT_WEIGHTS, MAX_MAG, SIGNS, TERM_NAMES
from .propose import Candidate

# Terms that PRODUCE walking. Perturbing these knocks the policy out of the walk basin,
# so they are protected: never pruned, small jitter, clamped near the parent value.
PROTECTED_TERMS = ("track_lin_vel_xy_exp", "track_ang_vel_z_exp", "termination_penalty")

# Only evolve from candidates whose fitness clears this floor (i.e. they actually walk;
# a statue gates to ~0). Keeps the whole population descended from real walkers.
WALKER_FITNESS_FLOOR = 0.10

# Annealed mutation: sigma_gen = SIGMA0 * DECAY**(gen-1), floored. gen 1 = widest.
SIGMA0 = 0.20
SIGMA_DECAY = 0.72
SIGMA_FLOOR = 0.06
PROTECTED_JITTER_FRAC = 0.35   # protected terms jitter at this fraction of sigma
PROTECTED_CLAMP = 0.30         # ...and stay within +/-30% of the parent value


def _clamp(term: str, magnitude: float) -> float:
    return float(min(max(magnitude, 0.0), MAX_MAG[term]))


def _base_mag(term: str) -> float:
    return abs(DEFAULT_WEIGHTS[term]) or MAX_MAG[term] * 0.1


def _sigma(gen: int) -> float:
    return max(SIGMA_FLOOR, SIGMA0 * (SIGMA_DECAY ** max(0, gen - 1)))


def _random_weights(rng: np.random.Generator) -> dict[str, float]:
    """Broad exploration of the NON-protected terms; protected terms stay at their
    walking (default) values so the candidate still walks and yields real signal."""
    w = {}
    for t in TERM_NAMES:
        if t in PROTECTED_TERMS:
            w[t] = DEFAULT_WEIGHTS[t]
            continue
        if rng.random() < 0.10:                       # 10% chance to prune a term
            w[t] = 0.0
            continue
        mag = _base_mag(t) * float(10 ** rng.uniform(-1.0, 1.0))   # 0.1x .. 10x default
        w[t] = SIGNS[t] * _clamp(t, mag)
    return w


def _mutate(weights: dict[str, float], rng: np.random.Generator, gen: int) -> dict[str, float]:
    sigma = _sigma(gen)
    prune_p = max(0.0, 0.05 * (0.6 ** max(0, gen - 1)))   # small, decaying
    w = {t: float(weights.get(t, 0.0)) for t in TERM_NAMES}
    for t in TERM_NAMES:
        protected = t in PROTECTED_TERMS
        cur = abs(w[t])
        if cur == 0.0:
            # revive an explore term occasionally (never auto-add a protected term)
            if not protected and rng.random() < 0.10:
                w[t] = SIGNS[t] * _clamp(t, _base_mag(t) * float(np.exp(rng.normal(0, sigma))))
            continue
        if not protected and rng.random() < prune_p:      # prune (explore terms only)
            w[t] = 0.0
            continue
        s = PROTECTED_JITTER_FRAC * sigma if protected else sigma
        cand = cur * float(np.exp(rng.normal(0, s)))       # log-normal jitter
        if protected:                                      # keep the walk basin
            cand = min(max(cand, (1.0 - PROTECTED_CLAMP) * cur), (1.0 + PROTECTED_CLAMP) * cur)
        w[t] = SIGNS[t] * _clamp(t, cand)
    return w


def _crossover(a: dict, b: dict, rng: np.random.Generator) -> dict[str, float]:
    return {t: (a.get(t, 0.0) if rng.random() < 0.5 else b.get(t, 0.0)) for t in TERM_NAMES}


def propose_weights_local(
    history: list[Candidate], n: int, rng: np.random.Generator, gen: int = 0
) -> list[Candidate]:
    """Return n fresh Candidates with `.weights` set (fitness filled by the caller).

    Evolve from the best WALKERS (fitness >= floor). If none walk yet, fall back to the
    single best candidate so the search still moves. `gen` drives the anneal schedule.
    """
    walkers = [c for c in history if c.fitness >= WALKER_FITNESS_FLOOR and c.weights]
    if not walkers and history:
        best = max(history, key=lambda c: c.fitness)
        walkers = [best] if best.weights else []
    parents = sorted(walkers, key=lambda c: c.fitness, reverse=True)[:6]
    parent_weights = [p.weights for p in parents]

    # pure-random exploration injection: present early, gone once we're focusing
    rand_p = 0.15 if gen <= 1 else 0.0

    out: list[Candidate] = []
    for i in range(n):
        if not parent_weights:
            w, how = _random_weights(rng), "rand"
        else:
            roll = rng.random()
            if roll < rand_p:
                w, how = _random_weights(rng), "rand"
            elif len(parent_weights) >= 2 and roll < rand_p + 0.25:
                # crossover between two WALKERS (both in the basin)
                a = parent_weights[int(rng.integers(len(parent_weights)))]
                b = parent_weights[int(rng.integers(len(parent_weights)))]
                w, how = _crossover(a, b, rng), "xover"
            else:
                p = parent_weights[int(rng.integers(len(parent_weights)))]
                w, how = _mutate(p, rng, gen), "mut"
        out.append(Candidate(name=f"{how}{i}", weights=w))
    return out
