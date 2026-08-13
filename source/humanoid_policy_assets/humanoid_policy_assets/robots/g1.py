"""Unitree G1 (29-DoF) articulation config for the squat<->stand task.

Why this file exists instead of using ``isaaclab_assets.robots.unitree.G1_29DOF_CFG`` directly:

* **Contact sensors.** ``G1_29DOF_CFG`` spawns with ``activate_contact_sensors=False``; the
  standup/squat rewards need foot and shin contacts.
* **Deployment gains.** The stock config carries Isaac's tuning (leg Kp 100-200). The robot is
  driven over ``rt/lowcmd`` with the Unitree SDK reference gains (legs Kp 60/60/60/100/40/40,
  Kd 1/1/1/2/1/1; waist 60/40/40; arms Kp 40 Kd 1 -- see xr_teleoperate ``docs/HANDOFF.md`` S4).
  Training with the gains the robot actually runs is the cheapest sim2real win available.
* **Soft joint limits.** The captured safety squat rides the mechanical limits (see
  ``G1_SAFETY_SQUAT_POSE`` below), so ``soft_joint_pos_limit_factor=0.9`` would place the *goal
  pose* outside the soft limits and make ``dof_pos_limits`` fight the objective. Raised to 1.0.
* **Joint ordering.** ``G1_LEG_JOINTS`` is written in ``rt/lowcmd`` motor-index order so a
  ``preserve_order=True`` action term maps 1:1 onto the deployment command vector.

Joint names match the Unitree G1 29-DoF URDF/MJCF exactly (verified against
``xr_teleoperate/assets/g1/g1_body29_hand14.xml``).
"""

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab_assets.robots.unitree import G1_29DOF_CFG

# --- joint groups -------------------------------------------------------------------------
# rt/lowcmd motor index order: 0..11 legs, 12..14 waist (yaw, roll, pitch), 15..28 arms.
G1_LEG_JOINTS = [
    "left_hip_pitch_joint",     # 0
    "left_hip_roll_joint",      # 1
    "left_hip_yaw_joint",       # 2
    "left_knee_joint",          # 3
    "left_ankle_pitch_joint",   # 4
    "left_ankle_roll_joint",    # 5
    "right_hip_pitch_joint",    # 6
    "right_hip_roll_joint",     # 7
    "right_hip_yaw_joint",      # 8
    "right_knee_joint",         # 9
    "right_ankle_pitch_joint",  # 10
    "right_ankle_roll_joint",   # 11
]

G1_WAIST_JOINTS = [
    "waist_yaw_joint",    # 12
    "waist_roll_joint",   # 13
    "waist_pitch_joint",  # 14
]

G1_ARM_JOINTS = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# The policy's action space: 12 legs + waist_pitch (13 DoF). waist_yaw/roll stay at their
# defaults -- the squat<->stand transition is sagittal, and the hands carry no load, so the
# arms follow a scripted trajectory outside the action space (HANDOFF S5/S6).
G1_POLICY_JOINTS = G1_LEG_JOINTS + ["waist_pitch_joint"]

# Bodies that carry load in the deep squat. The "shin" is the link between knee and ankle,
# named ``*_knee_link`` in the G1 model -- it is in ground contact at the bottom of the squat
# and lifts off as the robot rises (the contact schedule in HANDOFF S6).
G1_FOOT_BODIES = [".*_ankle_roll_link"]
G1_SHIN_BODIES = [".*_knee_link"]

# --- deployment (rt/lowcmd) PD gains ------------------------------------------------------
_LEG_STIFFNESS = {
    ".*_hip_pitch_joint": 60.0,
    ".*_hip_roll_joint": 60.0,
    ".*_hip_yaw_joint": 60.0,
    ".*_knee_joint": 100.0,
}
_LEG_DAMPING = {
    ".*_hip_pitch_joint": 1.0,
    ".*_hip_roll_joint": 1.0,
    ".*_hip_yaw_joint": 1.0,
    ".*_knee_joint": 2.0,
}
_FOOT_STIFFNESS = {".*_ankle_pitch_joint": 40.0, ".*_ankle_roll_joint": 40.0}
_FOOT_DAMPING = {".*_ankle_pitch_joint": 1.0, ".*_ankle_roll_joint": 1.0}
_WAIST_STIFFNESS = {"waist_yaw_joint": 60.0, "waist_roll_joint": 40.0, "waist_pitch_joint": 40.0}
_WAIST_DAMPING = {"waist_yaw_joint": 1.0, "waist_roll_joint": 1.0, "waist_pitch_joint": 1.0}


def _with_deploy_gains(cfg: ArticulationCfg) -> ArticulationCfg:
    """Return a copy of ``cfg`` with the SDK reference gains the robot is driven with."""
    out = cfg.copy()
    out.actuators = {name: act.copy() for name, act in cfg.actuators.items()}
    out.actuators["legs"].stiffness = dict(_LEG_STIFFNESS)
    out.actuators["legs"].damping = dict(_LEG_DAMPING)
    out.actuators["feet"].stiffness = dict(_FOOT_STIFFNESS)
    out.actuators["feet"].damping = dict(_FOOT_DAMPING)
    out.actuators["waist"].stiffness = dict(_WAIST_STIFFNESS)
    out.actuators["waist"].damping = dict(_WAIST_DAMPING)
    # arms are position-tracked along a scripted trajectory, not learned: Kp 40 / Kd 1 as on
    # the robot (the stock 3000/10 is a manipulation-grade stiffness that hides arm dynamics
    # the policy has to reject).
    out.actuators["arms"] = ImplicitActuatorCfg(
        joint_names_expr=[
            ".*_shoulder_pitch_joint",
            ".*_shoulder_roll_joint",
            ".*_shoulder_yaw_joint",
            ".*_elbow_joint",
            ".*_wrist_.*_joint",
        ],
        effort_limit=300.0,
        velocity_limit=100.0,
        stiffness=40.0,
        damping=1.0,
        armature=0.001,
    )
    return out


G1_SQUAT_CFG = _with_deploy_gains(G1_29DOF_CFG)
G1_SQUAT_CFG.spawn = G1_SQUAT_CFG.spawn.copy()
G1_SQUAT_CFG.spawn.activate_contact_sensors = True
# The captured pose sits ON the mechanical limits (hip_pitch/ankle_pitch within 0.005 rad,
# knee 1-2 deg past the URDF limit). A 0.9 soft factor would place the goal outside the soft
# range on 7 joints and saturate the dof_pos_limits penalty at the target. Keep the hard
# limits and rely on the explicit clamp applied when the pose is imported.
G1_SQUAT_CFG.soft_joint_pos_limit_factor = 1.0
"""Unitree G1 29-DoF configured for the squat<->stand task (deploy gains, contact sensors)."""


# --- fallback pose ------------------------------------------------------------------------
# The authoritative pose lives in ``configs/poses.yaml`` under ``g1_squat`` / ``g1_stand``,
# imported from the robot capture by ``scripts/rsl_rl/import_g1_pose.py``. This dict is only a
# graceful fallback (same convention as HUMANOID_SQUAT_POSE) if the library is unavailable:
# the captured safety squat in radians, clamped into the URDF hard limits.
G1_SAFETY_SQUAT_POSE = {
    "left_hip_pitch_joint": -2.5260,
    "left_hip_roll_joint": 0.1023,
    "left_hip_yaw_joint": -0.0068,
    "left_knee_joint": 2.8700,   # clamped from 2.8988 (URDF hi = 2.8798)
    "left_ankle_pitch_joint": -0.8709,
    "left_ankle_roll_joint": 0.0106,
    "right_hip_pitch_joint": -2.5279,
    "right_hip_roll_joint": -0.0579,
    "right_hip_yaw_joint": -0.0047,
    "right_knee_joint": 2.8700,  # clamped from 2.9107
    "right_ankle_pitch_joint": -0.8690,
    "right_ankle_roll_joint": -0.0134,
    "waist_yaw_joint": 0.0208,
    "waist_roll_joint": -0.0346,
    "waist_pitch_joint": 0.4986,
}
