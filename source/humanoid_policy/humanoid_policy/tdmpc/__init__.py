"""TD-MPC2 proactive world-model walk policy (branch proactive-world-model).

A model-based RL trainer (latent encoder + latent dynamics + reward predictor + value +
policy prior + MPPI planner) added alongside the reactive rsl_rl PPO path. See
docs/proactive_world_model/DESIGN.md and the approved plan. Vendored/adapted from the
official TD-MPC2 (MIT, github.com/nicklashansen/tdmpc2); our own GPU replay buffer avoids
the upstream torchrl/torch pin.
"""

from .config import TdmpcAgentCfg  # noqa: F401
