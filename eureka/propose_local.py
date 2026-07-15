"""API-free evolutionary proposer over the reward-term WEIGHT vector.

Ported from humanoid-tuner/policy/reward_search/propose_local.py, adapted from a
variable-length code genome to the fixed-length, sign-constrained weight vector of
the walk RewardsCfg. Same operators: mutate (log-normal magnitude jitter +
prune/revive), crossover (per-term parent pick), and random exploration; elitism
on the top-K by fitness. Signs are fixed (config.SIGNS) so a bonus can't become a
penalty or vice-versa.
"""

from __future__ import annotations

import numpy as np

from .config import DEFAULT_WEIGHTS, MAX_MAG, SIGNS, TERM_NAMES
from .propose import Candidate


def _clamp(term: str, magnitude: float) -> float:
    return float(min(max(magnitude, 0.0), MAX_MAG[term]))


def _base_mag(term: str) -> float:
    return abs(DEFAULT_WEIGHTS[term]) or MAX_MAG[term] * 0.1


def _random_weights(rng: np.random.Generator) -> dict[str, float]:
    w = {}
    for t in TERM_NAMES:
        if rng.random() < 0.10:                       # 10% chance to prune a term
            w[t] = 0.0
            continue
        mag = _base_mag(t) * float(10 ** rng.uniform(-1.0, 1.0))   # 0.1x .. 10x default
        w[t] = SIGNS[t] * _clamp(t, mag)
    return w


def _mutate(weights: dict[str, float], rng: np.random.Generator) -> dict[str, float]:
    w = {t: float(weights.get(t, 0.0)) for t in TERM_NAMES}
    for t in TERM_NAMES:
        cur = abs(w[t])
        if cur == 0.0:
            if rng.random() < 0.15:                   # revive a pruned term
                w[t] = SIGNS[t] * _clamp(t, _base_mag(t) * float(10 ** rng.uniform(-1, 0)))
            continue
        if rng.random() < 0.10:                       # prune
            w[t] = 0.0
            continue
        w[t] = SIGNS[t] * _clamp(t, cur * float(np.exp(rng.normal(0, 0.4))))  # log-normal jitter
    return w


def _crossover(a: dict, b: dict, rng: np.random.Generator) -> dict[str, float]:
    return {t: (a.get(t, 0.0) if rng.random() < 0.5 else b.get(t, 0.0)) for t in TERM_NAMES}


def propose_weights_local(
    history: list[Candidate], n: int, rng: np.random.Generator
) -> list[Candidate]:
    """Return n fresh Candidates with `.weights` set (fitness filled by the caller).

    First generation (empty history): random. Later: evolve the top-6 by fitness.
    """
    parents = sorted(history, key=lambda c: c.fitness, reverse=True)[:6]
    parent_weights = [p.weights for p in parents if p.weights]

    out: list[Candidate] = []
    for i in range(n):
        if not parent_weights:
            w, how = _random_weights(rng), "rand"
        else:
            roll = rng.random()
            if roll < 0.5 and len(parent_weights) >= 2:
                a = parent_weights[int(rng.integers(len(parent_weights)))]
                b = parent_weights[int(rng.integers(len(parent_weights)))]
                w, how = _crossover(a, b, rng), "xover"
            elif roll < 0.85:
                p = parent_weights[int(rng.integers(len(parent_weights)))]
                w, how = _mutate(p, rng), "mut"
            else:
                w, how = _random_weights(rng), "rand"
        out.append(Candidate(name=f"{how}{i}", weights=w))
    return out
