import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab.utils.configclass import configclass

import humanoid_policy.tasks.locomotion.velocity.mdp as mdp
from humanoid_policy.tasks.locomotion.velocity.velocity_env_cfg import LocomotionVelocityEnvCfg
from humanoid_policy_assets.robots.humanoid import HUMANOID_BIPED_WALK_CFG, HUMANOID_LEG_JOINTS
from humanoid_policy import pose_lib

# Spawn the walk policy from the authored `stand` pose (plus the reset randomization below), so it is
# robust to exactly where the standup policy ends -> clean stand->walk handoff. Falls back to the cfg
# default standing pose if the pose library is unavailable.
_STAND = pose_lib.load_library(pose_lib.DEFAULT_LIBRARY_PATH).get("stand")

# --- Action bound (guardrail) ---------------------------------------------------------------
# On hardware the walk policy diverged: raw actions grew to ~10 -> 149-deg target offsets, joints
# slammed the position limits and thrashed at 12 rad/s (see docs/walk-policy-divergence-report.md).
# Clip the RAW action to +/-_ACTION_RAW_LIMIT so the fed-back `prev_action` term cannot explode.
# Isaac Lab's JointAction `clip` acts on the PROCESSED target (raw*scale + default_pose), so we
# build a per-joint clip centered on the stand pose: [stand_j - R*scale, stand_j + R*scale], which
# is exactly equivalent to clipping the raw action to +/-R. The deploy exporter (scripts/rsl_rl/
# play.py) recovers the same R from this clip, so train and deploy bound identically. Kept generous
# so it does NOT distort a normal gait (a swing knee needs ~0.7 rad offset ~= raw 2.8); it only
# stops the catastrophic runaway.
_ACTION_SCALE = 0.25
_ACTION_RAW_LIMIT = 4.0  # max |raw action| -> +/-1.0 rad (57 deg) target offset from the stand pose
_ACTION_CLIP = (
    {j: (v - _ACTION_RAW_LIMIT * _ACTION_SCALE, v + _ACTION_RAW_LIMIT * _ACTION_SCALE)
     for j, v in _STAND.joint_pos.items()}
    if _STAND is not None else None
)

# Nominal standing base height (flat-ground world z), same convention as the standup task. Used to
# terminate an episode when the base collapses well below standing (guards against the policy
# learning to thrash/squat instead of walk). None -> termination inert (height unknown).
_STAND_BASE_HEIGHT = float(_STAND.base_pos[2]) if _STAND is not None else None
_MIN_BASE_HEIGHT = (_STAND_BASE_HEIGHT - 0.15) if _STAND_BASE_HEIGHT is not None else -10.0


##
# MDP settings
##

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    # Berkeley-Humanoid-Lite command: OMNIDIRECTIONAL velocity tracking (walk any
    # direction) — the actual end goal, and the setup Berkeley proved walks. We keep the
    # `WalkMetricsVelocityCommandCfg` subclass ONLY to log readback metrics (tracked_speed /
    # commanded_speed / base_accel_rms / rocking_rms) for the gated Eureka fitness; every
    # command FIELD below matches Berkeley's. The Eureka gate is on tracked_speed (velocity
    # in the commanded direction), which is direction-agnostic — a statue -> ~0, a tracker
    # -> ~1 — so the lenient success metric is no longer relied on for grading.
    base_velocity = mdp.WalkMetricsVelocityCommandCfg(
        resampling_time_range=(10.0, 10.0),
        debug_vis=True,
        asset_name="robot",
        heading_command=True,
        heading_control_stiffness=0.5,
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 0.5),
            lin_vel_y=(-0.25, 0.25),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"}
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.3, n_max=0.3),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=HUMANOID_LEG_JOINTS, preserve_order=True)},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=HUMANOID_LEG_JOINTS, preserve_order=True)},
            noise=Unoise(n_min=-2.0, n_max=2.0),
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True

    @configclass
    class CriticCfg(PolicyCfg):
        """Observations for critic group."""
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)

        def __post_init__(self):
            self.enable_corruption = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=HUMANOID_LEG_JOINTS,
        scale=_ACTION_SCALE,
        preserve_order=True,
        use_default_offset=True,
        clip=_ACTION_CLIP,
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP.

    Weights are the **g2c3** reward from the Eureka search (fitness 0.6926) — the best
    stable-walk tuning found on top of the Berkeley-Humanoid-Lite base (which we reverted
    to after the earlier motion-suppression penalty stack collapsed the policy into
    standing; see docs/ + eureka/). Values are g2c3 rounded to ~4 sig figs (the extra
    digits were within seed noise). The three hardware-safety penalties Berkeley/g2c3
    leave at 0 (base_accel_xy_l2, action_l2, dof_vel_l2) are kept defined-but-off so they
    can be re-introduced for sim->real without re-adding the term.
    """

    # === Reward for task-space performance ===
    # command tracking performance
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        params={"command_name": "base_velocity", "std": 0.25},
        weight=1.787,
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        params={"command_name": "base_velocity", "std": 0.25},
        weight=1.042,
    )

    # === Reward for basic behaviors ===
    # termination penalty
    termination_penalty = RewTerm(
        func=mdp.is_terminated,
        weight=-9.588,
    )

    # base motion smoothness
    lin_vel_z_l2 = RewTerm(
        func=mdp.lin_vel_z_l2,
        weight=-0.1081,
    )
    ang_vel_xy_l2 = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-0.03924,
    )
    # smooth walk: penalize fast horizontal base linear acceleration ("fast IMU X/Y changes").
    # OFF in g2c3 (Berkeley has no such term); a small negative weight here is the natural
    # "small bump to stability/smoothness" knob for a full run.
    base_accel_xy_l2 = RewTerm(
        func=mdp.base_lin_accel_xy_l2,
        weight=0.0,
    )
    # ensure the robot is standing upright
    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-2.183,
    )

    # joint motion smoothness
    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.014,
    )
    # action_l2 / dof_vel_l2: hardware-safety penalties (docs/walk-policy-divergence-report.md
    # §4B) that suppress high-frequency, large-amplitude actions. Berkeley/g2c3 leave them OFF
    # (they had over-damped the gait into standing); re-introduce with small weights for sim->real.
    action_l2 = RewTerm(
        func=mdp.action_l2,
        weight=0.0,
    )
    dof_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=HUMANOID_LEG_JOINTS)},
        weight=0.0,
    )
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=HUMANOID_LEG_JOINTS)},
        weight=-0.001783,
    )
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=HUMANOID_LEG_JOINTS)},
        weight=-1.027e-6,
    )
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-0.8446,
    )

    # === Reward for encouraging behaviors ===
    # encourage robot to take steps
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll"),
            "threshold": 0.4,
        },
        weight=1.199,
    )
    # penalize feet sliding on the ground to exploit physics sim inaccuracies
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll"),
        },
        weight=-0.07207,
    )

    # penalize undesired contacts (falls, and -- with self-collision enabled -- leg-vs-leg contact)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", ".*_hip_.*", ".*_knee_.*"]),
            "threshold": 1.0,
        },
        weight=-1.298,
    )

    # penalize deviation from default of the joints that are not essential for locomotion
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
        weight=-0.1607,
    )
    joint_deviation_ankle_roll = RewTerm(
        func=mdp.joint_deviation_l1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_roll_joint"])},
        weight=-0.1707,
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(
        func=mdp.time_out,
        time_out=True,
    )
    base_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 0.78, "asset_cfg": SceneEntityCfg("robot", body_names="base")},
    )
    # End the episode if the base collapses ~15 cm below standing, so the policy is penalized for
    # sinking/thrashing instead of walking (the runaway on hardware pinned joints at their limits).
    base_height = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": _MIN_BASE_HEIGHT, "asset_cfg": SceneEntityCfg("robot", body_names="base")},
    )


@configclass
class EventsCfg:
    """Configuration for events."""

    # === Startup behaviors ===
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 1.2),
            "dynamic_friction_range": (0.4, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
        mode="startup",
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 2.0),
            "operation": "add",
        },
        mode="startup",
    )
    add_all_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.05, 0.05),
            "operation": "add",
        },
        mode="startup",
    )
    scale_all_actuator_torque_constant = EventTerm(
        func=mdp.randomize_actuator_gains,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
        mode="startup",
    )
    # === Actuator-model domain randomization (bench-validated motor models) ===============
    # We have ONE bench unit of each motor type, so randomize AROUND the fitted nominals to
    # cover per-joint 3D-printed variation. Latency DR (0.5-1.5x) is built into the actuator
    # itself (per-reset random delay in [min_delay, max_delay]); here we add inertia + friction.
    # Both are no-ops / harmless under the implicit baseline (HUMANOID_ACTUATOR_MODEL=0).
    randomize_leg_armature = EventTerm(
        func=mdp.randomize_joint_parameters,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=HUMANOID_LEG_JOINTS),
            "armature_distribution_params": (0.8, 1.2),  # reflected motor+gearbox inertia +-20%
            "operation": "scale",
        },
        mode="startup",
    )
    randomize_joint_friction = EventTerm(
        func=mdp.randomize_stickslip_friction,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=HUMANOID_LEG_JOINTS),
            "friction_distribution_params": (0.7, 1.3),  # stick-slip coulomb/breakaway/viscous +-30%
        },
        mode="startup",
    )

    # === Reset behaviors ===
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.0),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
        mode="reset",
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (-2.0, 2.0),
            "torque_range": (-2.0, 2.0),
            # "force_range": (-3.0, 3.0),
            # "torque_range": (-3.0, 3.0),
        },
        mode="reset",
    )

    # === Interval behaviors ===
    # push_robot = EventTerm(
    #     func=mdp.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(10.0, 15.0),
    #     params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
    # )


@configclass
class CurriculumsCfg:
    """Curriculum terms for the MDP."""

    # No curriculum — Berkeley trains omnidirectional walking with no command/terrain
    # curriculum and it walks. (A forward-command curriculum lived here previously; removed
    # with the switch back to Berkeley's symmetric command.)
    pass


@configclass
class HumanoidBipedEnvCfg(LocomotionVelocityEnvCfg):

    # Policy commands
    commands: CommandsCfg = CommandsCfg()

    # Policy observations
    observations: ObservationsCfg = ObservationsCfg()

    # Policy actions
    actions: ActionsCfg = ActionsCfg()

    # Policy rewards
    rewards: RewardsCfg = RewardsCfg()

    # Termination conditions
    terminations: TerminationsCfg = TerminationsCfg()

    # Randomization events
    events: EventsCfg = EventsCfg()

    # Curriculums
    curriculums: CurriculumsCfg = CurriculumsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Physics settings
        # 25 Hz override
        self.decimation = 8

        # Scene: spawn STANDING at the authored stand pose (+ reset randomization) for a clean
        # stand->walk handoff, instead of the cfg's default standing pose.
        robot = HUMANOID_BIPED_WALK_CFG.replace(prim_path="{ENV_REGEX_NS}/robot")
        if _STAND is not None:
            robot.init_state = robot.init_state.replace(
                pos=tuple(float(x) for x in _STAND.base_pos),
                rot=tuple(float(x) for x in _STAND.base_quat),  # (w, x, y, z)
                joint_pos={k: float(v) for k, v in _STAND.joint_pos.items()},
            )
        self.scene.robot = robot
