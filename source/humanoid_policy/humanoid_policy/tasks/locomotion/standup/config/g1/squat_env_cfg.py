"""Unitree G1: stand -> squat (controlled descent) -- the safety-critical direction.

This is the motion the safety framework performs before de-energizing: a stop descends into the
statically stable squat rather than dropping (xr_teleoperate ``docs/HANDOFF.md`` S7, and
``teleop/safety/tests/test_scenarios.py`` asserts the ordering). Gravity assists a *collapse*, so
the hard part is making the descent controlled -- the smoothness and contact terms carry the load,
exactly as in the biped's squat task.

Differences from the standup direction, all of them consequences of the target pose being folded:

* the arm trajectory runs stand -> squat;
* ``flat_orientation_l2`` is dropped. The captured squat pitches the pelvis hard, so on the way
  *down* an upright-pelvis penalty directly opposes the goal (on the way up it is a useful shaping
  term, which is why the standup task keeps it at a reduced weight);
* the pose-match term is sharpened so the policy commits to the full fold instead of settling for a
  safe shallow crouch -- the same failure the biped descent showed.
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

import humanoid_policy.tasks.locomotion.standup.mdp as mdp
from humanoid_policy_assets.robots.g1 import G1_POLICY_JOINTS, G1_SQUAT_CFG

from .env_cfg import (
    CommandsCfg as _StandupCommandsCfg,
    G1StandupEnvCfg,
    RewardsCfg as _StandupRewardsCfg,
    _ARM_DELAY_S,
    _ARM_DURATION_S,
    _ARM_WAYPOINT,
    _SQUAT_ARMS,
    _SQUAT_BASE_HEIGHT,
    _SQUAT_JOINTS,
    _STAND,
    _STAND_ARMS,
    _STAND_JOINTS,
)


@configclass
class CommandsCfg(_StandupCommandsCfg):
    """Arms run the trajectory in reverse: from the stand configuration into the squat."""

    # Same waypoint: the half-sine bump is symmetric, so it routes the hands around the legs in
    # either direction of travel (the collision-free path was in fact validated stand -> squat).
    arm_trajectory = mdp.ArmTrajectoryCommandCfg(
        asset_name="robot",
        start_pose=_STAND_ARMS,
        end_pose=_SQUAT_ARMS,
        waypoint_offset=_ARM_WAYPOINT,
        delay_s=_ARM_DELAY_S,
        duration_s=_ARM_DURATION_S,
        debug_vis=False,
    )


@configclass
class RewardsCfg(_StandupRewardsCfg):
    """Match the SQUAT pose from a stand; stability + smoothness dominate."""

    track_stand_pose = RewTerm(
        func=mdp.track_joint_pose_exp,
        params={
            "target": {j: v for j, v in _SQUAT_JOINTS.items() if j in G1_POLICY_JOINTS},
            "reference": {j: v for j, v in _STAND_JOINTS.items() if j in G1_POLICY_JOINTS},
            "std": 0.35,
            "asset_cfg": SceneEntityCfg("robot"),
        },
        weight=2.5,
    )
    # depth driver: only a real hip/knee fold lowers the pelvis, so make squat pelvis height a
    # first-class, sharp objective (the biped needed exactly this to stop crouching shallow).
    base_height_bonus = RewTerm(
        func=mdp.base_height_exp,
        params={"target_height": _SQUAT_BASE_HEIGHT, "std": 0.10},
        weight=2.0,
    )
    # see module docstring: an upright-pelvis penalty opposes the descent target.
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)


@configclass
class G1SquatEnvCfg(G1StandupEnvCfg):
    """G1 stand -> squat: spawn standing, descend into the captured safety squat."""

    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()

    def __post_init__(self):
        super().__post_init__()  # sets decimation/episode + the squat spawn, overridden below

        robot = G1_SQUAT_CFG.replace(prim_path="{ENV_REGEX_NS}/robot")
        init = robot.init_state.replace(joint_pos=dict(_STAND_JOINTS), joint_vel={".*": 0.0})
        if _STAND is not None:
            init = init.replace(
                pos=tuple(float(x) for x in _STAND.base_pos),
                rot=tuple(float(x) for x in _STAND.base_quat),  # (w, x, y, z)
            )
        robot.init_state = init
        self.scene.robot = robot
