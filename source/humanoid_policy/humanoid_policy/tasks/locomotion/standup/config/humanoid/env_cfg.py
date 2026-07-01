from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils.configclass import configclass

import humanoid_policy.tasks.locomotion.standup.mdp as mdp
from humanoid_policy.tasks.locomotion.velocity.velocity_env_cfg import LocomotionVelocityEnvCfg
# full-body obs/action groups (22 DoF) from the walk task
from humanoid_policy.tasks.locomotion.velocity.config.humanoid.env_cfg import (
    ObservationsCfg,
    ActionsCfg,
    EventsCfg as _WalkHumanoidEventsCfg,
)
# stand-up command/reward/termination terms are robot-agnostic — reuse the biped ones
from humanoid_policy.tasks.locomotion.standup.config.biped.env_cfg import (
    CommandsCfg,
    RewardsCfg,
    TerminationsCfg,
    CurriculumsCfg,
)
from humanoid_policy_assets.robots.berkeley_humanoid_lite import (
    HUMANOID_LITE_CFG,
    HUMANOID_LITE_SQUAT_POSE,
    HUMANOID_LITE_ARM_JOINTS,
)

# full-body squat: arms parked at 0, legs in the contract squat pose
_HUMANOID_SQUAT_POSE = {joint: 0.0 for joint in HUMANOID_LITE_ARM_JOINTS}
_HUMANOID_SQUAT_POSE.update(HUMANOID_LITE_SQUAT_POSE)

# NOTE: the full humanoid keeps the generic actuator gains (the per-joint firmware gains in the
# contract are legs-only). Arm regularization for stand-up is minimal for now — this variant is
# future-facing until arms are physically connected.
HUMANOID_LITE_SQUAT_CFG = HUMANOID_LITE_CFG.replace(
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos=_HUMANOID_SQUAT_POSE,
        joint_vel={".*": 0.0},
    ),
)


@configclass
class EventsCfg(_WalkHumanoidEventsCfg):
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.98, 1.02), "velocity_range": (0.0, 0.0)},
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {"yaw": (-3.14, 3.14)}, "velocity_range": {}},
    )


@configclass
class BerkeleyHumanoidLiteStandupEnvCfg(LocomotionVelocityEnvCfg):

    commands: CommandsCfg = CommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculums: CurriculumsCfg = CurriculumsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.decimation = 8
        self.episode_length_s = 10.0

        self.scene.robot = HUMANOID_LITE_SQUAT_CFG.replace(prim_path="{ENV_REGEX_NS}/robot")
