"""TD-MPC2-ONLY biped walk env: the same env as HumanoidBipedEnvCfg but with a STABILITY-GATED
reward instead of the additive g2c3 (PPO/Eureka) reward.

Why a separate cfg/task: the additive reward lets TD-MPC2's short-horizon planner trade uprightness
for command-tracking and topple. This cfg swaps in a multiplicative uprightness gate
(mdp.gated_locomotion, HumanoidBench-style) so reward -> 0 the instant the robot isn't upright.
Everything else (obs, actions, terminations, events, commands, the modeled actuator plant) is
inherited unchanged. The PPO reward (RewardsCfg / HumanoidBipedEnvCfg) is NOT touched.
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

import humanoid_policy.tasks.locomotion.velocity.mdp as mdp
from .env_cfg import HumanoidBipedEnvCfg, _STAND_BASE_HEIGHT

# Standing-height threshold for the gate: a touch below the nominal stand height so normal gait
# bob still counts as "standing". Fallback if the pose library was unavailable at import.
_STAND_H = (float(_STAND_BASE_HEIGHT) - 0.05) if _STAND_BASE_HEIGHT is not None else 0.50


@configclass
class GatedRewardsCfg:
    """Stability-first reward: one uprightness-gated locomotion term dominates, plus small
    ungated shaping. All reward is forfeited while not upright (see mdp.gated_locomotion)."""

    # the whole task, gated by standing×upright (per-step in [0,1])
    stand_walk = RewTerm(
        func=mdp.gated_locomotion,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "tracking_std": 0.25,
            "stand_height": _STAND_H,
            "upright_min": 0.8,
            "move_weight": 0.5,   # upright-and-still earns 0.5; tracking the command earns the rest
        },
    )
    # explicit one-time cost for actually falling (episode-ending). Small vs the dense gate.
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-1.0)
    # light ungated shaping for smoothness / joint safety (don't fight the gate)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-0.1)


@configclass
class HumanoidBipedTdmpcEnvCfg(HumanoidBipedEnvCfg):
    """Biped walk env with the stability-gated reward — for the TD-MPC2 trainer."""

    rewards: GatedRewardsCfg = GatedRewardsCfg()
