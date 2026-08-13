"""Unitree G1: squat -> stand.

Scope comes from xr_teleoperate ``docs/HANDOFF.md`` S6 and the measurements behind it:

* **13-DoF action space** -- 12 leg joints + ``waist_pitch``. The arms carry no load in the
  captured safety squat, so they follow a deterministic trajectory (``ArmTrajectoryCommand``)
  outside the action space; their phase is observed so the policy can anticipate the inertial
  disturbance rather than only react to it.
* **Terminal pose, not commanded velocity.** The reward is ``track_joint_pose_exp`` toward the
  authored stand pose, normalized by the squat->stand travel, plus a pelvis-height term.
* **Contact schedule.** The squat rests on feet *and* shins; the shins lift off during the rise.
  ``scheduled_contact`` charges shin contact only once the pelvis is above ``_SHIN_RELEASE_HEIGHT``,
  so the start pose is free but dragging up is not.
* **Self-collision stays offline.** Only hand-vs-knee contact matters and it is a fixed-geometry
  question about the arm path, answered by ``xr_teleoperate/tools/check_self_collision.py``
  (validate the arm trajectory with ``--from``/``--to`` before training). There is deliberately no
  collision reward term, and the articulation keeps ``enabled_self_collisions=False``: calf-on-thigh
  and arm-on-chest contact are wanted in this pose.

Poses live in ``configs/poses.yaml`` (keys ``g1_squat`` / ``g1_stand``); import the robot capture
with ``scripts/rsl_rl/import_g1_pose.py`` and floor-snap it in ``pose_editor.py`` before training.
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab.utils.configclass import configclass

import humanoid_policy.tasks.locomotion.standup.mdp as mdp
from humanoid_policy.tasks.locomotion.velocity.velocity_env_cfg import LocomotionVelocityEnvCfg
from humanoid_policy_assets.robots.g1 import (
    G1_ARM_JOINTS,
    G1_FOOT_BODIES,
    G1_POLICY_JOINTS,
    G1_SAFETY_SQUAT_POSE,
    G1_SHIN_BODIES,
    G1_SQUAT_CFG,
)
from humanoid_policy import pose_lib

_POSES = pose_lib.load_library(pose_lib.DEFAULT_LIBRARY_PATH)
_SQUAT = _POSES.get("g1_squat")
_STAND = _POSES.get("g1_stand")

# Fallbacks keep the module importable (and the env constructible) before the poses are authored,
# the same degrade-gracefully convention the biped task uses.
_SQUAT_JOINTS = dict(_SQUAT.joint_pos) if _SQUAT is not None else dict(G1_SAFETY_SQUAT_POSE)
# Nominal G1 stand = the stock init pose (slight hip/knee/ankle bend, everything else zero).
_STAND_FALLBACK = {j: 0.0 for j in _SQUAT_JOINTS}
_STAND_FALLBACK.update(
    {
        "left_hip_pitch_joint": -0.10, "right_hip_pitch_joint": -0.10,
        "left_knee_joint": 0.30, "right_knee_joint": 0.30,
        "left_ankle_pitch_joint": -0.20, "right_ankle_pitch_joint": -0.20,
    }
)
_STAND_JOINTS = dict(_STAND.joint_pos) if _STAND is not None else _STAND_FALLBACK

# Pelvis heights [m], world z on flat ground. Fallbacks are the G1's nominal standing pelvis height
# and a rough deep-squat pelvis; both are placeholders until the poses are floor-snapped.
_STAND_BASE_HEIGHT = float(_STAND.base_pos[2]) if _STAND is not None else 0.75
_SQUAT_BASE_HEIGHT = float(_SQUAT.base_pos[2]) if _SQUAT is not None else 0.30
# Shin contact is free below this and penalized above it -- set a third of the way up the rise.
_SHIN_RELEASE_HEIGHT = _SQUAT_BASE_HEIGHT + 0.33 * (_STAND_BASE_HEIGHT - _SQUAT_BASE_HEIGHT)

# Arm trajectory endpoints: the arms hold the captured squat configuration and move to a
# relaxed-at-side stand. Only arm joints; the legs/waist belong to the policy.
_SQUAT_ARMS = {j: _SQUAT_JOINTS.get(j, 0.0) for j in G1_ARM_JOINTS}
_STAND_ARMS = {j: _STAND_JOINTS.get(j, 0.0) for j in G1_ARM_JOINTS}

# Arm-path waypoint. A straight joint-space line between the two arm configurations puts both
# hands through the thighs and knees (50 mm penetration at the worst sample) even though both
# endpoints are clean. Swinging the shoulders out and back clears it: with this bump the tightest
# point on the whole path is the *endpoint* itself, at 4.2 mm.
#
# Verified with ``check_self_collision.py`` over 240 samples against the placeholder stand pose.
# **Re-validate once the real g1_stand is captured** -- these numbers are tuned to an endpoint
# that is still a placeholder.
_ARM_WAYPOINT = {
    "left_shoulder_pitch_joint": -1.047,   # -60 deg
    "right_shoulder_pitch_joint": -1.047,
    "left_shoulder_roll_joint": 0.349,     # +20 deg, abducting each arm away from the thigh
    "right_shoulder_roll_joint": -0.349,
}
# Hold the arms until the legs are well into the fold, then move them. Validated as "arms start at
# ~50% of the leg motion"; expressed here in seconds against episode time, since the legs are
# policy-driven rather than scheduled. Re-check if episode_length_s or the rise duration changes.
_ARM_DELAY_S = 2.0
_ARM_DURATION_S = 3.0

_ACTION_SCALE = 0.25


@configclass
class CommandsCfg:
    """The scripted arm motion, observed by the policy as ``[phase, phase_rate]``."""

    arm_trajectory = mdp.ArmTrajectoryCommandCfg(
        asset_name="robot",
        start_pose=_SQUAT_ARMS,
        end_pose=_STAND_ARMS,
        waypoint_offset=_ARM_WAYPOINT,
        delay_s=_ARM_DELAY_S,
        duration_s=_ARM_DURATION_S,
        debug_vis=False,
    )


@configclass
class ObservationsCfg:
    """47-dim policy observation: 2 phase + 3 ang vel + 3 gravity + 13 q + 13 qd + 13 actions."""

    @configclass
    class PolicyCfg(ObsGroup):

        arm_phase = ObsTerm(func=mdp.generated_commands, params={"command_name": "arm_trajectory"})
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.3, n_max=0.3))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=G1_POLICY_JOINTS, preserve_order=True)},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=G1_POLICY_JOINTS, preserve_order=True)},
            noise=Unoise(n_min=-2.0, n_max=2.0),
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True

    @configclass
    class CriticCfg(PolicyCfg):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)

        def __post_init__(self):
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class ActionsCfg:
    """13-DoF position action, ordered to match rt/lowcmd indices 0-11 and 14."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=G1_POLICY_JOINTS,
        scale=_ACTION_SCALE,
        preserve_order=True,
        use_default_offset=True,
    )


@configclass
class RewardsCfg:
    """Stability-dominant, with the stand pose as the task signal (mirrors the biped standup)."""

    # === the task: reach the stand pose ===
    track_stand_pose = RewTerm(
        func=mdp.track_joint_pose_exp,
        params={
            "target": {j: v for j, v in _STAND_JOINTS.items() if j in G1_POLICY_JOINTS},
            "reference": {j: v for j, v in _SQUAT_JOINTS.items() if j in G1_POLICY_JOINTS},
            "std": 0.4,
            "asset_cfg": SceneEntityCfg("robot"),
        },
        weight=2.0,
    )
    base_height_bonus = RewTerm(
        func=mdp.base_height_exp,
        params={"target_height": _STAND_BASE_HEIGHT, "std": 0.25},
        weight=1.0,
    )

    # === stability ===
    # NOTE: weaker than the biped's -2.0. The G1 safety squat folds the hips to -145 deg, which
    # pitches the pelvis hard; a strong flat-orientation penalty fights the *start* pose rather
    # than only the failure mode. Tune this against the pose reward first if the rise stalls.
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    is_alive = RewTerm(func=mdp.is_alive, weight=1.0)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-10.0)

    # === contact schedule ===
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=G1_FOOT_BODIES),
            "asset_cfg": SceneEntityCfg("robot", body_names=G1_FOOT_BODIES),
        },
        weight=-1.0,
    )
    feet_off_ground = RewTerm(
        func=mdp.feet_off_ground,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=G1_FOOT_BODIES)},
        weight=-0.5,
    )
    # shins may rest on the ground in the squat; dragging them up is penalized
    shin_contact = RewTerm(
        func=mdp.scheduled_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=G1_SHIN_BODIES),
            "release_height": _SHIN_RELEASE_HEIGHT,
            "threshold": 1.0,
        },
        weight=-0.5,
    )
    # the pelvis/torso must never take load
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["pelvis", "torso_link"]),
            "threshold": 1.0,
        },
        weight=-1.0,
    )

    # === smoothness / effort (policy joints only) ===
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    dof_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=G1_POLICY_JOINTS)},
        weight=-5.0e-3,
    )
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=G1_POLICY_JOINTS)},
        weight=-2.0e-3,
    )
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=G1_POLICY_JOINTS)},
        weight=-1.0e-6,
    )
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)


@configclass
class TerminationsCfg:

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # Generous: the deep squat already pitches the pelvis well past a walking tilt limit. Only a
    # genuine topple should end the episode.
    base_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 1.4, "asset_cfg": SceneEntityCfg("robot", body_names="pelvis")},
    )


@configclass
class EventsCfg:
    """Startup domain randomization + reset into the (lightly noised) squat.

    Written fresh rather than inherited from the biped walk events: those reference the Berkeley
    robot's ``base`` body and its stick-slip actuator model, neither of which exists on the G1.
    """

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
    add_torso_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-1.0, 2.0),
            "operation": "add",
        },
        mode="startup",
    )
    scale_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
        mode="startup",
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.98, 1.02), "velocity_range": (0.0, 0.0)},
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"yaw": (-3.14, 3.14), "z": (0.0, 0.03)},
            "velocity_range": {
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                "z": (-0.1, 0.0),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.1, 0.1),
            },
        },
    )


@configclass
class CurriculumsCfg:
    pass


@configclass
class G1StandupEnvCfg(LocomotionVelocityEnvCfg):
    """G1 squat -> stand: spawn in the captured safety squat, rise to the stand pose."""

    commands: CommandsCfg = CommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculums: CurriculumsCfg = CurriculumsCfg()

    def __post_init__(self):
        super().__post_init__()

        # 50 Hz policy (sim dt 0.005). The G1 deployment loop in xr_teleoperate runs the low-level
        # command stream faster than the Berkeley biped's 25 Hz; keep the decimation adjustable
        # here and match whatever the deploy runner ends up using.
        self.decimation = 4
        self.episode_length_s = 8.0

        robot = G1_SQUAT_CFG.replace(prim_path="{ENV_REGEX_NS}/robot")
        init = robot.init_state.replace(joint_pos=dict(_SQUAT_JOINTS), joint_vel={".*": 0.0})
        if _SQUAT is not None:
            init = init.replace(
                pos=tuple(float(x) for x in _SQUAT.base_pos),
                rot=tuple(float(x) for x in _SQUAT.base_quat),  # (w, x, y, z)
            )
        robot.init_state = init
        self.scene.robot = robot
