"""MDP terms for the stand-up task.

Reuses all locomotion/velocity mdp terms (which themselves re-export isaaclab.envs.mdp) and
adds stand-up-specific reward terms.
"""

from humanoid_policy.tasks.locomotion.velocity.mdp import *  # noqa: F401, F403

from .rewards import *  # noqa: F401, F403
