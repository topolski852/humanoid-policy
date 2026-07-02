# Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

ISAACLAB_ASSET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))

HUMANOID_LITE_LEG_JOINTS = [
    "leg_left_hip_roll_joint",
    "leg_left_hip_yaw_joint",
    "leg_left_hip_pitch_joint",
    "leg_left_knee_pitch_joint",
    "leg_left_ankle_pitch_joint",
    "leg_left_ankle_roll_joint",
    "leg_right_hip_roll_joint",
    "leg_right_hip_yaw_joint",
    "leg_right_hip_pitch_joint",
    "leg_right_knee_pitch_joint",
    "leg_right_ankle_pitch_joint",
    "leg_right_ankle_roll_joint",
]

HUMANOID_LITE_ARM_JOINTS = [
    "arm_left_shoulder_pitch_joint",
    "arm_left_shoulder_roll_joint",
    "arm_left_shoulder_yaw_joint",
    "arm_left_elbow_pitch_joint",
    "arm_left_elbow_roll_joint",
    "arm_right_shoulder_pitch_joint",
    "arm_right_shoulder_roll_joint",
    "arm_right_shoulder_yaw_joint",
    "arm_right_elbow_pitch_joint",
    "arm_right_elbow_roll_joint",
]

HUMANOID_LITE_JOINTS = HUMANOID_LITE_ARM_JOINTS + HUMANOID_LITE_LEG_JOINTS

HUMANOID_LITE_BIPED_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSET_DIR}/robots/berkeley_humanoid/berkeley_humanoid_lite/usd/berkeley_humanoid_lite_biped.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            "leg_left_hip_roll_joint": 0.0,
            "leg_left_hip_yaw_joint": 0.0,
            "leg_left_hip_pitch_joint": -0.2,
            "leg_left_knee_pitch_joint": 0.4,
            "leg_left_ankle_pitch_joint": -0.3,
            "leg_left_ankle_roll_joint": 0.0,
            "leg_right_hip_roll_joint": 0.0,
            "leg_right_hip_yaw_joint": 0.0,
            "leg_right_hip_pitch_joint": -0.2,
            "leg_right_knee_pitch_joint": 0.4,
            "leg_right_ankle_pitch_joint": -0.3,
            "leg_right_ankle_roll_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "leg_.*_hip_yaw_joint",
                "leg_.*_hip_roll_joint",
                "leg_.*_hip_pitch_joint",
                "leg_.*_knee_pitch_joint",
            ],
            effort_limit=6,
            velocity_limit=10.0,
            stiffness=20,
            damping=2,
            armature=0.007,
        ),
        "ankles": ImplicitActuatorCfg(
            joint_names_expr=[
                "leg_.*_ankle_pitch_joint",
                "leg_.*_ankle_roll_joint",
            ],
            effort_limit=6,
            velocity_limit=10.0,
            stiffness=20,
            damping=2,
            armature=0.002,
        ),
    },
)
"""Configuration for the Berkeley Humanoid Lite robot in bipedal mode."""

HUMANOID_LITE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSET_DIR}/robots/berkeley_humanoid/berkeley_humanoid_lite/usd/berkeley_humanoid_lite.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            "arm_left_shoulder_pitch_joint": 0.0,
            "arm_left_shoulder_roll_joint": 0.0,
            "arm_left_shoulder_yaw_joint": 0.0,
            "arm_left_elbow_pitch_joint": 0.0,
            "arm_left_elbow_roll_joint": 0.0,
            "arm_right_shoulder_pitch_joint": 0.0,
            "arm_right_shoulder_roll_joint": 0.0,
            "arm_right_shoulder_yaw_joint": 0.0,
            "arm_right_elbow_pitch_joint": 0.0,
            "arm_right_elbow_roll_joint": 0.0,
            "leg_left_hip_roll_joint": 0.0,
            "leg_left_hip_yaw_joint": 0.0,
            "leg_left_hip_pitch_joint": -0.2,
            "leg_left_knee_pitch_joint": 0.4,
            "leg_left_ankle_pitch_joint": -0.3,
            "leg_left_ankle_roll_joint": 0.0,
            "leg_right_hip_roll_joint": 0.0,
            "leg_right_hip_yaw_joint": 0.0,
            "leg_right_hip_pitch_joint": -0.2,
            "leg_right_knee_pitch_joint": 0.4,
            "leg_right_ankle_pitch_joint": -0.3,
            "leg_right_ankle_roll_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                "arm_.*_shoulder_pitch_joint",
                "arm_.*_shoulder_roll_joint",
                "arm_.*_shoulder_yaw_joint",
                "arm_.*_elbow_pitch_joint",
                "arm_.*_elbow_roll_joint",
            ],
            effort_limit=4,
            velocity_limit=10.0,
            stiffness=10,
            damping=2,
            armature=0.002,
        ),
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "leg_.*_hip_yaw_joint",
                "leg_.*_hip_roll_joint",
                "leg_.*_hip_pitch_joint",
                "leg_.*_knee_pitch_joint",
            ],
            effort_limit=6,
            velocity_limit=10.0,
            stiffness=20,
            damping=2,
            armature=0.007,
        ),
        "ankles": ImplicitActuatorCfg(
            joint_names_expr=[
                "leg_.*_ankle_pitch_joint",
                "leg_.*_ankle_roll_joint",
            ],
            effort_limit=6,
            velocity_limit=10.0,
            stiffness=20,
            damping=2,
            armature=0.002,
        ),
    },
)
"""Configuration for the Berkeley Humanoid Lite robot."""


##
# Stand-up (squat -> stand) configuration.
#
# Init pose + per-joint PD gains/effort are the sim<->real contract from humanoid-control
# (configs/leg_policy_params.json / policy_starting_pose.json). Kept as a SEPARATE cfg so the
# walk configs above stay on the original generic gains until walking is deployed.
#
# NOTE: the hardware values are LEFT/RIGHT ASYMMETRIC (individually tuned 3D-printed joints,
# e.g. hip_yaw kp 10.5 L vs 20 R; ankle_pitch kd 2.0 L vs 0.5 R). This is device truth for
# THIS robot; if a more symmetric / generalizable policy is preferred, symmetrize these dicts.
##

# Deep-squat starting pose (radians), sim joint names. Derived from policy_starting_pose.json
# "starting_pose_final" (per-pair L/R averaged, clamped to URDF limits; ankle_roll mirrored).
#
# IMPORTANT: hip_pitch is INVERTED between this sim/USD and the hardware. The contract/encoder reads
# the squat hip at +0.982 (+56 deg, its upper limit), but in this USD that value drives the thigh the
# WRONG way (feet fold up). The physical squat is reproduced in sim at the OPPOSITE limit, hip_pitch
# ~= -1.898 (-108.75 deg). Verified in sim: with this pose + the base rot in the squat cfg it settles
# stable, feet flat, ~14 deg forward torso tilt, matching the real robot. This inversion also affects
# the sim<->real contract (walk included) and is an open reconciliation item.
#
# knee_pitch, ankle_pitch, and now hip_pitch are nudged ~0.01 rad (0.57 deg) OFF their exact
# URDF-limit values because Isaac Lab rejects default joint positions that are not STRICTLY inside
# the joint limits.
_LIMIT_EPS = 0.01
HUMANOID_LITE_SQUAT_POSE = {
    "leg_left_hip_roll_joint": 0.029593753814697265,
    "leg_left_hip_yaw_joint": 0.0038009449839591977,
    "leg_left_hip_pitch_joint": -1.8980527578749847 + _LIMIT_EPS,
    "leg_left_knee_pitch_joint": 2.443460952792061 - _LIMIT_EPS,
    "leg_left_ankle_pitch_joint": -0.7853981633974483 + _LIMIT_EPS,
    "leg_left_ankle_roll_joint": 0.013601303100585938,
    "leg_right_hip_roll_joint": 0.029593753814697265,
    "leg_right_hip_yaw_joint": 0.0038009449839591977,
    "leg_right_hip_pitch_joint": -1.8980527578749847 + _LIMIT_EPS,
    "leg_right_knee_pitch_joint": 2.443460952792061 - _LIMIT_EPS,
    "leg_right_ankle_pitch_joint": -0.7853981633974483 + _LIMIT_EPS,
    "leg_right_ankle_roll_joint": 0.013601303100585938,
}

# Per-joint firmware gains pulled from the ESCs (device truth). kp->position_kp, kd->velocity_kp.
_CONTRACT_KP = {
    "leg_left_hip_roll_joint": 20.0, "leg_left_hip_yaw_joint": 10.5, "leg_left_hip_pitch_joint": 68.4,
    "leg_left_knee_pitch_joint": 27.0, "leg_left_ankle_pitch_joint": 18.0, "leg_left_ankle_roll_joint": 23.3,
    "leg_right_hip_roll_joint": 20.0, "leg_right_hip_yaw_joint": 20.0, "leg_right_hip_pitch_joint": 68.4,
    "leg_right_knee_pitch_joint": 30.0, "leg_right_ankle_pitch_joint": 20.0, "leg_right_ankle_roll_joint": 20.0,
}
_CONTRACT_KD = {
    "leg_left_hip_roll_joint": 4.0, "leg_left_hip_yaw_joint": 0.5, "leg_left_hip_pitch_joint": 9.8,
    "leg_left_knee_pitch_joint": 2.45, "leg_left_ankle_pitch_joint": 2.0, "leg_left_ankle_roll_joint": 4.0,
    "leg_right_hip_roll_joint": 4.0, "leg_right_hip_yaw_joint": 1.0, "leg_right_hip_pitch_joint": 9.8,
    "leg_right_knee_pitch_joint": 1.22, "leg_right_ankle_pitch_joint": 0.5, "leg_right_ankle_roll_joint": 2.0,
}
_CONTRACT_EFFORT = {
    "leg_left_hip_roll_joint": 6.0, "leg_left_hip_yaw_joint": 12.0, "leg_left_hip_pitch_joint": 9.5,
    "leg_left_knee_pitch_joint": 6.0, "leg_left_ankle_pitch_joint": 6.0, "leg_left_ankle_roll_joint": 7.0,
    "leg_right_hip_roll_joint": 6.0, "leg_right_hip_yaw_joint": 6.0, "leg_right_hip_pitch_joint": 9.5,
    "leg_right_knee_pitch_joint": 6.0, "leg_right_ankle_pitch_joint": 6.0, "leg_right_ankle_roll_joint": 6.0,
}

_LEG_GROUP = ["leg_.*_hip_yaw_joint", "leg_.*_hip_roll_joint", "leg_.*_hip_pitch_joint", "leg_.*_knee_pitch_joint"]
_ANKLE_GROUP = ["leg_.*_ankle_pitch_joint", "leg_.*_ankle_roll_joint"]


def _subset(d, joint_exprs_leaf):
    """Pick the contract-dict entries whose joint name contains one of the given leaf tokens."""
    return {k: v for k, v in d.items() if any(tok in k for tok in joint_exprs_leaf)}


_LEG_LEAVES = ["hip_yaw", "hip_roll", "hip_pitch", "knee_pitch"]
_ANKLE_LEAVES = ["ankle_pitch", "ankle_roll"]

HUMANOID_LITE_BIPED_SQUAT_CFG = HUMANOID_LITE_BIPED_CFG.replace(
    init_state=ArticulationCfg.InitialStateCfg(
        # Squat spawn placed to match the real robot (verified in sim, stable under gravity holding
        # the pose): base(pelvis) origin low with a base pitch that settles to ~14 deg forward torso
        # tilt, feet flat on the floor. rot is (w,x,y,z) for a ~197 deg base pitch about Y.
        pos=(0.0, 0.0, -0.22),
        rot=(-0.147809, 0.0, 0.989016, 0.0),
        joint_pos=dict(HUMANOID_LITE_SQUAT_POSE),
        joint_vel={".*": 0.0},
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=_LEG_GROUP,
            velocity_limit=10.0,
            effort_limit=_subset(_CONTRACT_EFFORT, _LEG_LEAVES),
            stiffness=_subset(_CONTRACT_KP, _LEG_LEAVES),
            damping=_subset(_CONTRACT_KD, _LEG_LEAVES),
            armature=0.007,
        ),
        "ankles": ImplicitActuatorCfg(
            joint_names_expr=_ANKLE_GROUP,
            velocity_limit=10.0,
            effort_limit=_subset(_CONTRACT_EFFORT, _ANKLE_LEAVES),
            stiffness=_subset(_CONTRACT_KP, _ANKLE_LEAVES),
            damping=_subset(_CONTRACT_KD, _ANKLE_LEAVES),
            armature=0.002,
        ),
    },
)
"""Berkeley Humanoid Lite biped configured for squat->stand: deep-squat init pose + per-joint
firmware PD gains from the humanoid-control policy contract."""
