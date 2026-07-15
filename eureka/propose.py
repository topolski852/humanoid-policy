"""Candidate type + an inert Claude proposer stub (for when an API key exists).

The default search uses the API-free local evolutionary proposer
(`propose_local.py`). This mirrors humanoid-tuner's split so the Claude generator
can be dropped in later without touching the loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Candidate:
    name: str
    weights: dict            # {term_name: weight}
    fitness: float = float("-inf")
    components: dict = field(default_factory=dict)
    run_dir: str | None = None
    stopped_reason: str = ""


def propose_weights_claude(history, n, model="claude-opus-4-8"):
    """Not wired for humanoid-policy yet (no API key). Kept as the drop-in point
    for a Claude generator — mirror humanoid-tuner/policy/reward_search/propose.py
    (structured-output weight vectors) when credentials are available."""
    raise NotImplementedError(
        "Claude proposer not available (no API key). The default local evolutionary "
        "proposer needs no credentials.")
