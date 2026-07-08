from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

import humanoid_policy.tasks.locomotion.standup.mdp as mdp
from humanoid_policy.tasks.locomotion.velocity.velocity_env_cfg import LocomotionVelocityEnvCfg
# Reuse the walk task's observation + action groups verbatim so the stand-up policy keeps the
# exact 45-dim obs / 12-dim action layout of the sim<->real contract.
from humanoid_policy.tasks.locomotion.velocity.config.biped.env_cfg import (
    ObservationsCfg,
    ActionsCfg,
    EventsCfg as _WalkEventsCfg,
)
from humanoid_policy_assets.robots.berkeley_humanoid_lite import (
    HUMANOID_LITE_BIPED_SQUAT_CFG,
    HUMANOID_LITE_LEG_JOINTS,
)

# Nominal standing base height [m]. The `base` link is the PELVIS (low, near the feet), not the
# torso top: measured in sim the pelvis sits ~-0.03 m standing vs ~-0.24 m in the deep squat, so the
# stand-up is a ~0.2 m pelvis rise. Target 0.0 (a few cm above the settled stand to encourage a full
# rise). (An earlier 0.55 assumed base==torso-top and was unreachable.)
STANDING_BASE_HEIGHT = 0.0


@configclass
class CommandsCfg:
    """Zero velocity command — keeps the 3-dim command obs slot (control feeds zeros for stand-up)."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=False,
        asset_name="robot",
        heading_command=False,
        rel_standing_envs=1.0,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=(0.0, 0.0),
        ),
    )


@configclass
class RewardsCfg:
    """Reward terms for squat -> stand."""

    # === main task: rise to and hold standing height ===
    # coarse quadratic pull toward standing height (gradient all the way from the squat)
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        params={"target_height": STANDING_BASE_HEIGHT},
        weight=-10.0,
    )
    # sharp bonus for settling exactly at standing height
    base_height_bonus = RewTerm(
        func=mdp.base_height_exp,
        params={"target_height": STANDING_BASE_HEIGHT, "std": 0.15},
        weight=2.0,
    )
    # stay upright
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)
    # alive bonus (discourages toppling to end the episode early)
    is_alive = RewTerm(func=mdp.is_alive, weight=0.5)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-10.0)

    # === keep the feet planted (no kicking a leg out for momentum) ===
    # penalize horizontal foot velocity while loaded (a foot skating outward on the ground)
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll"),
        },
        weight=-1.0,
    )
    # penalize either foot leaving the ground (catches a kick that lifts the foot clear)
    feet_off_ground = RewTerm(
        func=mdp.feet_off_ground,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll")},
        weight=-0.5,
    )

    # === smoothness / effort ===
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    # penalize joint speed -> rise slowly and controlled (the "slow movement" lever;
    # base_height_l2 still wants it up, so it rises only as fast as this tolerates)
    dof_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=HUMANOID_LITE_LEG_JOINTS)},
        weight=-5.0e-3,
    )
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=HUMANOID_LITE_LEG_JOINTS)},
        weight=-2.0e-3,
    )
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=HUMANOID_LITE_LEG_JOINTS)},
        weight=-1.0e-6,
    )
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)

    # don't cheat by resting weight on base/hips/knees
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", ".*_hip_.*", ".*_knee_.*"]),
            "threshold": 1.0,
        },
        weight=-1.0,
    )
    # keep the non-essential joints near default for a clean symmetric stand
    # stronger than the walk task's -0.2: the leg kick is hip yaw/roll abduction, so tax it harder
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
        weight=-0.5,
    )
    joint_deviation_ankle_roll = RewTerm(
        func=mdp.joint_deviation_l1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_roll_joint"])},
        weight=-0.2,
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # generous tilt limit: a deep squat leans the torso, but a true topple still ends the episode
    base_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 1.2, "asset_cfg": SceneEntityCfg("robot", body_names="base")},
    )


@configclass
class EventsCfg(_WalkEventsCfg):
    """Reuse the walk startup randomization; reset every env to (a lightly noised) squat."""

    # tight scale around the squat default instead of the walk task's (0.5, 1.5)
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.98, 1.02), "velocity_range": (0.0, 0.0)},
    )
    # randomize yaw, and spawn up to 5 cm high with small base velocity noise so the policy must
    # settle an imperfect, moving start before committing to the rise (generalizes to a real,
    # not-perfectly-placed robot; the deployment-side "wait for IMU to settle" gate is separate).
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"yaw": (-3.14, 3.14), "z": (0.0, 0.05)},
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (-0.2, 0.0),
                "roll": (-0.2, 0.2),
                "pitch": (-0.2, 0.2),
                "yaw": (-0.2, 0.2),
            },
        },
    )


@configclass
class CurriculumsCfg:
    pass


@configclass
class BerkeleyHumanoidLiteBipedStandupEnvCfg(LocomotionVelocityEnvCfg):

    commands: CommandsCfg = CommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculums: CurriculumsCfg = CurriculumsCfg()

    def __post_init__(self):
        super().__post_init__()

        # 25 Hz policy
        self.decimation = 8
        # shorter episodes than walking — standing up is a quick transient
        self.episode_length_s = 10.0

        # start every environment in the deep squat (contract pose + per-joint firmware gains)
        self.scene.robot = HUMANOID_LITE_BIPED_SQUAT_CFG.replace(prim_path="{ENV_REGEX_NS}/robot")
