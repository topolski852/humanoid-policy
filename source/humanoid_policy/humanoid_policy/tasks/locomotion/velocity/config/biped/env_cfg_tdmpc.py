"""TD-MPC2-ONLY biped walk env: the same env as HumanoidBipedEnvCfg but with a STABILITY-GATED
reward instead of the additive g2c3 (PPO/Eureka) reward.

Why a separate cfg/task: the additive reward lets TD-MPC2's short-horizon planner trade uprightness
for command-tracking and topple. This cfg swaps in a multiplicative uprightness gate
(mdp.gated_locomotion, HumanoidBench-style) so reward -> 0 the instant the robot isn't upright.
Everything else (obs, actions, terminations, events, commands, the modeled actuator plant) is
inherited unchanged. The PPO reward (RewardsCfg / HumanoidBipedEnvCfg) is NOT touched.
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

import humanoid_policy.tasks.locomotion.velocity.mdp as mdp
from .env_cfg import HumanoidBipedEnvCfg, EventsCfg, _STAND_BASE_HEIGHT

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
            # 0.75: standing-still only earns the 0.25 baseline; tracking the command earns the rest,
            # so walking pays far more than parking (escape the stand-still local optimum).
            "move_weight": 0.75,
        },
    )
    # explicit one-time cost for actually falling (episode-ending). Small vs the dense gate.
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-1.0)
    # light ungated shaping for smoothness / joint safety (don't fight the gate)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-0.1)


@configclass
class GentleEventsCfg(EventsCfg):
    """Gentler domain randomization so TD-MPC2 can learn to balance without being constantly
    knocked over. Drops the mid-episode pushes and softens the reset external force/torque
    (the harder-command DR was tuned for PPO's 590M-sample brute force)."""

    push_robot = None  # no mid-episode ±0.8 m/s shoves
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (-1.0, 1.0),    # softened from ±3
            "torque_range": (-1.0, 1.0),
        },
        mode="reset",
    )


@configclass
class HumanoidBipedTdmpcEnvCfg(HumanoidBipedEnvCfg):
    """Biped walk env with the stability-gated reward + gentler DR — for the TD-MPC2 trainer.
    PHASE 2 of the curriculum (full commands): warm-start from a phase-1 stand checkpoint."""

    rewards: GatedRewardsCfg = GatedRewardsCfg()
    events: GentleEventsCfg = GentleEventsCfg()


# =========================================================================================
# CURRICULUM PHASE 1 — learn to STAND (near-zero command, calm spawn). With the gated reward a
# zero command means "track zero velocity" = stand still, which earns full reward when upright.
# Once this stands cleanly, warm-start the walk task (phase 2) from its checkpoint
# (train.py --init_checkpoint <phase1 model_best>).
# =========================================================================================


@configclass
class StandEventsCfg(GentleEventsCfg):
    """Even gentler than phase 2: spawn calm and near the stand pose so it can learn to HOLD a
    stand before facing perturbations."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-0.3, 0.3)},
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                               "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
        },
        mode="reset",
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.98, 1.02), "velocity_range": (0.0, 0.0)},
    )


@configclass
class HumanoidBipedTdmpcStandEnvCfg(HumanoidBipedTdmpcEnvCfg):
    """PHASE 1: same gated reward, but commanded to STAND (zero velocity) with a calm spawn."""

    events: StandEventsCfg = StandEventsCfg()

    def __post_init__(self):
        super().__post_init__()
        r = self.commands.base_velocity.ranges
        r.lin_vel_x = (0.0, 0.0)
        r.lin_vel_y = (0.0, 0.0)
        r.ang_vel_z = (0.0, 0.0)
        r.heading = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 1.0
        self.commands.base_velocity.heading_command = False
