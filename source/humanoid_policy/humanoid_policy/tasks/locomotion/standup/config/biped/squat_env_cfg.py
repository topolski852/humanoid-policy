"""Stand -> squat (controlled descent) task -- the reverse of the standup task.

Spawns the robot **standing** (the authored `stand` pose) and rewards matching the authored `squat`
pose, so the end-to-end sequence can be stand -> walk -> squat (the squat is the easy-to-handle
resting posture). Reuses the standup task's rewards/observations/actions/events verbatim and only
swaps the two poses: pose target = squat, normalization reference = stand, base-height target =
squat pelvis height. Descending is expected to be harder than rising (gravity assists a *collapse*
rather than a controlled descent), so the stability + smoothness terms inherited from the standup
reward do the heavy lifting of keeping it controlled.
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

import humanoid_policy.tasks.locomotion.standup.mdp as mdp
from humanoid_policy_assets.robots.humanoid import HUMANOID_BIPED_SQUAT_CFG

from .env_cfg import (
    HumanoidBipedStandupEnvCfg,
    RewardsCfg as _StandupRewardsCfg,
    _SQUAT,
    _STAND,
    _SQUAT_JOINTS,
    _STAND_JOINTS,
)

# squat pelvis height [m] (target of the descent)
_SQUAT_BASE_HEIGHT = float(_SQUAT.base_pos[2]) if _SQUAT is not None else -0.23


@configclass
class SquatRewardsCfg(_StandupRewardsCfg):
    """Reverse of the standup reward: match the SQUAT pose (from a stand), stability-dominant.

    Inherits every stability/smoothness/contact term from the standup reward; only the pose-match
    target and the base-height target are flipped to the squat.
    """

    # match the squat pose; per-joint normalized by the stand->squat travel (reference = stand).
    # minor bumps vs the standup default (weight 2.0 -> 2.5, std 0.4 -> 0.35) to push the descent
    # into the FULL deep squat instead of settling for a safe shallow crouch.
    track_stand_pose = RewTerm(
        func=mdp.track_joint_pose_exp,
        params={
            "target": _SQUAT_JOINTS,
            "reference": _STAND_JOINTS,
            "std": 0.35,
            "asset_cfg": SceneEntityCfg("robot"),
        },
        weight=2.5,
    )
    # DEPTH driver: pelvis height is the pure squat-depth signal (spreading the legs doesn't lower
    # the base -- only a deep knee/hip bend does). The per-joint pose-match gives cheap credit for the
    # easy lateral spread, so it was going "squat-shaped" but not deep. Make base-height a strong,
    # sharp term (weight 0.5 -> 2.0, std 0.18 -> 0.10) so reaching the low squat pelvis is a first-
    # class objective that forces the actual descent.
    base_height_bonus = RewTerm(
        func=mdp.base_height_exp,
        params={"target_height": _SQUAT_BASE_HEIGHT, "std": 0.10},
        weight=2.0,
    )


@configclass
class HumanoidBipedSquatEnvCfg(HumanoidBipedStandupEnvCfg):
    """Stand -> squat: spawn standing (authored stand pose), descend to match the squat pose."""

    rewards: SquatRewardsCfg = SquatRewardsCfg()

    def __post_init__(self):
        super().__post_init__()  # decimation/episode + (standup) squat spawn, which we override below
        # start every environment STANDING (the authored, floor-verified stand pose)
        robot = HUMANOID_BIPED_SQUAT_CFG.replace(prim_path="{ENV_REGEX_NS}/robot")
        if _STAND is not None:
            robot.init_state = robot.init_state.replace(
                pos=tuple(float(x) for x in _STAND.base_pos),
                rot=tuple(float(x) for x in _STAND.base_quat),  # (w, x, y, z)
                joint_pos={k: float(v) for k, v in _STAND.joint_pos.items()},
            )
        self.scene.robot = robot
