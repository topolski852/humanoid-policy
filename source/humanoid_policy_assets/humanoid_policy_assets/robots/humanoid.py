# Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from humanoid_policy_assets.actuators import StickSlipDelayedPDActuatorCfg, load_actuator_model

ISAACLAB_ASSET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))

HUMANOID_LEG_JOINTS = [
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

HUMANOID_ARM_JOINTS = [
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

HUMANOID_JOINTS = HUMANOID_ARM_JOINTS + HUMANOID_LEG_JOINTS

HUMANOID_BIPED_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSET_DIR}/robots/humanoid/usd/humanoid_biped.usd",
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
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
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
"""Configuration for the Humanoid robot in bipedal mode."""

HUMANOID_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSET_DIR}/robots/humanoid/usd/humanoid.usd",
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
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
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
"""Configuration for the Humanoid robot."""


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
HUMANOID_SQUAT_POSE = {
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
# --- TUNED GAINS (2026-08-17) ------------------------------------------------------------
# Replaces the per-joint asymmetric ESC gains that were commissioned by hand. Bench tuning in
# humanoid-tuner found kp=45 / kd=1.5 gave the best response for BOTH motor+gearbox types
# (M6C12 legs, MAD5010 ankles), so this is the new sim<->real contract, not a sim-only knob:
# the robot will be flashed with these and scripts/rsl_rl/play.py exports whatever is set here
# into deploy/walk/leg_policy_contract.json, keeping train and deploy identical.
#
# Kept as per-joint dicts (rather than a scalar) so the export path stays per-joint and any
# future re-asymmetrization is a value edit, not a structural one.
#
# SUPERSEDED per-joint values, for reference / revert:
#   kp  hip_roll 20.0 | hip_yaw 10.5 L, 20.0 R | hip_pitch 68.4 | knee 27.0 L, 30.0 R
#       ankle_pitch 18.0 L, 20.0 R | ankle_roll 23.3 L, 20.0 R
#   kd  hip_roll 4.0 | hip_yaw 0.5 L, 1.0 R | hip_pitch 9.8 | knee 2.45 L, 1.22 R
#       ankle_pitch 2.0 L, 0.5 R | ankle_roll 4.0 L, 2.0 R
# NOTE the magnitude of two of these: hip_pitch drops 68.4/9.8 -> 45/1.5, i.e. a 6.5x cut in
# damping on the strongest joint on the robot. That is what the bench says; it is also the joint
# the 2026-07-14 divergence report flagged for a possible sim<->hardware SIGN inversion, still
# unresolved. Watch hip_pitch behaviour in the first training run.
# _CONTRACT_EFFORT is derived per MOTOR TYPE below -- torque caps are firmware limits, not tuning.
# Gain preset, selectable with HUMANOID_GAIN_PRESET (default "tuned"). Only kp/kd change --
# the plant (actuator model, armature, friction, latency) and _CONTRACT_EFFORT stay fixed, so a
# run-to-run comparison isolates the GAINS and nothing else.
#   tuned    45.0 / 1.5  bench-calibrated on the real motors (humanoid-tuner, 2026-08); deployed.
#   berkeley 20.0 / 2.0  upstream Berkeley Humanoid Lite defaults, unchanged since the original
#                        scaffold (commit 31cfd92). Lower stiffness AND higher damping than tuned,
#                        i.e. markedly better damped -- the A/B for on-robot jitter.
#   tuned_kd3 45.0 / 3.0 same stiffness as tuned, damping doubled. NOT YET RUN -- staged for the
#                        isolated A/B that follows the measurement round. Motivation: the bench
#                        gain validation inside configs/actuators/*.json says BOTH motors do best
#                        at kd=3 (M6C12 7.12 mrad at kp40/kd3 vs 16.84 at kp40/kd1.5 -- the
#                        deployed setting is the WORST of the six points tested; MAD5010 6.39 vs
#                        10.67). Smooth B's 15.8 s non-decaying 4.11 Hz ring after a single push
#                        (humanoid-control REPORT_2026-09-23_smoothB.md sec 1) is an underdamped
#                        signature. Run this ALONE -- changing gains alongside the reward and
#                        observation changes makes attribution impossible.
_GAIN_PRESETS = {"tuned": (45.0, 1.5), "berkeley": (20.0, 2.0), "tuned_kd3": (45.0, 3.0)}
_GAIN_PRESET = os.environ.get("HUMANOID_GAIN_PRESET", "tuned").strip().lower()
if _GAIN_PRESET not in _GAIN_PRESETS:
    raise ValueError(
        f"HUMANOID_GAIN_PRESET={_GAIN_PRESET!r} is not one of {sorted(_GAIN_PRESETS)}"
    )
_TUNED_KP, _TUNED_KD = _GAIN_PRESETS[_GAIN_PRESET]
print(f"[INFO] humanoid gain preset '{_GAIN_PRESET}': kp={_TUNED_KP} kd={_TUNED_KD}")
_LEG_JOINT_NAMES = (
    "leg_left_hip_roll_joint", "leg_left_hip_yaw_joint", "leg_left_hip_pitch_joint",
    "leg_left_knee_pitch_joint", "leg_left_ankle_pitch_joint", "leg_left_ankle_roll_joint",
    "leg_right_hip_roll_joint", "leg_right_hip_yaw_joint", "leg_right_hip_pitch_joint",
    "leg_right_knee_pitch_joint", "leg_right_ankle_pitch_joint", "leg_right_ankle_roll_joint",
)
_CONTRACT_KP = {j: _TUNED_KP for j in _LEG_JOINT_NAMES}
_CONTRACT_KD = {j: _TUNED_KD for j in _LEG_JOINT_NAMES}
##
# Torque caps are a MOTOR property, not a joint property.
#
# The 12 leg joints carry exactly two motor types (confirmed three ways: `torque_constant` in
# humanoid-studio/configs/humanoid_lite.json, the _LEG_GROUP/_ANKLE_GROUP split below, and the
# mass audit in configs/actuators/PROVENANCE.md):
#
#   MAD M6C12 150KV  x8  hip_roll, hip_yaw, hip_pitch, knee_pitch  (both sides)
#   MAD 5010  200KV  x4  ankle_pitch, ankle_roll                   (both sides)
#
# Physical ceiling of a joint is  Kt * gear * current_limit  -- no firmware torque_limit above
# that is reachable, because current binds first. Kt and gear come from the bench MotorSpec
# (humanoid-tuner sim/isaac/motor.py), which is also the source of the friction/inertia/latency
# models vendored in configs/actuators/. current_limit is a per-motor ESC setting; both specs
# declare 20 A, so that is the value to flash on every joint of a type.
#
#   M6C12   Kt 0.08958 * 15 = 1.3437 Nm/A  -> ceiling 26.87 Nm @ 20 A
#   MAD5010 Kt 0.06588 * 15 = 0.9882 Nm/A  -> ceiling 19.76 Nm @ 20 A
#
# WHY NOT TRAIN AT THE CEILING. The cap has to leave headroom, for two measured reasons:
#   1. docs/measurements/REPORT_2026-09-23_smoothA.md sec 2 -- knees demand p95 28 Nm on
#      hardware, ABOVE the 26.87 ceiling. Training at the ceiling would let the policy keep
#      asking for torque the motor cannot deliver; the cap is what teaches it not to.
#   2. Raising the knee cap 6.0 -> 11.0 on 2026-08-24 removed what had been an accidental
#      low-pass filter, and the policy's 4-5 Hz command content reached the joint. Smooth B
#      then rang at 4.11 Hz for 15.8 s after a single push without decaying
#      (REPORT_2026-09-23_smoothB.md sec 1). Going to 26.87 removes that limiting entirely.
#
# CHOSEN VALUE: the highest cap already in service on that motor type. Nothing on the robot
# loses authority, nothing gains more than a joint of the same type already runs, and the
# per-joint spread (four different values across eight identical M6C12s, plus an L/R asymmetric
# hip_yaw) is gone. Both land at a third to a half of ceiling, which is the headroom.
##
_MOTOR_SPECS = {
    # name:        (Kt Nm/A, gear, current_limit A, effort cap Nm)
    "M6C12_150KV": (0.08958, 15.0, 20.0, 12.0),
    "MAD5010_200KV": (0.06588, 15.0, 20.0, 7.0),
}
_MOTOR_BY_LEAF = {
    "hip_roll": "M6C12_150KV", "hip_yaw": "M6C12_150KV",
    "hip_pitch": "M6C12_150KV", "knee_pitch": "M6C12_150KV",
    "ankle_pitch": "MAD5010_200KV", "ankle_roll": "MAD5010_200KV",
}


def _motor_of(joint_name: str) -> str:
    """Motor type driving a leg joint, by leaf token. Raises on an unmapped joint rather than
    silently defaulting -- an unmapped joint would get a wrong torque cap on real hardware."""
    for leaf, motor in _MOTOR_BY_LEAF.items():
        if leaf in joint_name:
            return motor
    raise KeyError(f"no motor mapping for joint {joint_name!r}")


def _motor_ceiling(motor: str) -> float:
    """Kt * gear * current_limit -- the most torque this motor can physically produce."""
    kt, gear, i_limit, _ = _MOTOR_SPECS[motor]
    return kt * gear * i_limit


# Every joint of a type gets its type's cap. Flash the SAME numbers to the ESCs (torque_limit,
# and current_limit 20.0 on all 12 -- two joints currently run below that and so cannot reach
# their configured cap: both ankle_roll at 6 A can only make 5.93 Nm against a 7.0 setting, and
# left_hip_yaw at 10 A only 13.44 Nm. See REPORT_2026-09-23_smoothB.md sec 2a).
_CONTRACT_EFFORT = {j: _MOTOR_SPECS[_motor_of(j)][3] for j in _LEG_JOINT_NAMES}

for _m, (_kt, _g, _i, _cap) in _MOTOR_SPECS.items():
    if _cap > _motor_ceiling(_m):
        raise ValueError(
            f"{_m}: effort cap {_cap} Nm exceeds physical ceiling "
            f"{_motor_ceiling(_m):.2f} Nm (Kt {_kt} * gear {_g} * {_i} A)"
        )
print("[INFO] motor torque caps: " + ", ".join(
    f"{_m.split('_')[0]} {_MOTOR_SPECS[_m][3]:.1f} Nm "
    f"({100 * _MOTOR_SPECS[_m][3] / _motor_ceiling(_m):.0f}% of {_motor_ceiling(_m):.1f} ceiling)"
    for _m in _MOTOR_SPECS
))

_LEG_GROUP = ["leg_.*_hip_yaw_joint", "leg_.*_hip_roll_joint", "leg_.*_hip_pitch_joint", "leg_.*_knee_pitch_joint"]
_ANKLE_GROUP = ["leg_.*_ankle_pitch_joint", "leg_.*_ankle_roll_joint"]


def _subset(d, joint_exprs_leaf):
    """Pick the contract-dict entries whose joint name contains one of the given leaf tokens."""
    return {k: v for k, v in d.items() if any(tok in k for tok in joint_exprs_leaf)}


_LEG_LEAVES = ["hip_yaw", "hip_roll", "hip_pitch", "knee_pitch"]
_ANKLE_LEAVES = ["ankle_pitch", "ankle_roll"]

HUMANOID_BIPED_SQUAT_CFG = HUMANOID_BIPED_CFG.replace(
    init_state=ArticulationCfg.InitialStateCfg(
        # Squat spawn placed to match the real robot (verified in sim, stable under gravity holding
        # the pose): base(pelvis) origin low with a base pitch that settles to ~14 deg forward torso
        # tilt, feet flat on the floor. rot is (w,x,y,z) for a ~197 deg base pitch about Y.
        pos=(0.0, 0.0, -0.22),
        rot=(-0.147809, 0.0, 0.989016, 0.0),
        joint_pos=dict(HUMANOID_SQUAT_POSE),
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
"""Humanoid biped configured for squat->stand: deep-squat init pose + per-joint
firmware PD gains from the humanoid-control policy contract."""


##
# WALK actuators — two builds behind a toggle so the modeled plant is reversible / A/B-able.
#
#  * IMPLICIT baseline: the original ImplicitActuatorCfg groups (PhysX PD, no friction, no
#    latency, light armature 0.007/0.002). What the currently-deployed policy trained on.
#  * MODELED plant: bench-validated actuator models (humanoid-tuner commit bd7c613). Adds the
#    measured reflected inertia (armature), stick-slip joint friction, and command latency —
#    closing the sim-to-real gap that made the deployed policy "freak out" on the robot.
#
# Select with _WALK_USE_ACTUATOR_MODEL (env var HUMANOID_ACTUATOR_MODEL=0 forces the baseline).
# BOTH keep the exact policy<->robot contract: per-joint contract PD gains (kp/kd) and the real
# per-joint effort limits (_CONTRACT_EFFORT). Only the sim plant differs.
##

# Physics step is 5 ms (sim.dt=0.005). Command latency is expressed in integer physics steps,
# randomized per-reset in [min_delay, max_delay] (this IS the 0.5-1.5x latency DR). The 5 ms grid
# can't hit the measured values exactly, so we bracket them: 7.2 ms (legs) -> 1-2 steps (5-10 ms);
# 12 ms (ankles) -> 2-3 steps (10-15 ms). Measured latencies are from humanoid-tuner sim/isaac/
# motor.py MotorSpec (NOT the fitted JSON's latency_s, which reads 0 — the gentle fit didn't excite it).
_LEG_DELAY = (1, 2)
_ANKLE_DELAY = (2, 3)

_M6C12 = load_actuator_model("m6c12_pitch")    # legs: hip roll/yaw/pitch + knee
_MAD5010 = load_actuator_model("mad5010_roll")  # ankles: ankle pitch/roll

# Ankle STATIC friction, measured on the robot 2026-09-28 (humanoid-control
# docs/measurements/REPORT_2026-09-28_M7.md, TRAINING_INPUT.json -> actuator.m7_static_stiffness).
# With both legs hanging and all 12 joints held, ankle_pitch needed ~0.40 N·m beyond gravity to
# hold on BOTH sides (left 0.41, right 0.43 median), present even near zero gravity load.
#
# The vendored bench fit has breakaway 0.249. With the +-30% friction DR that spans 0.174-0.324,
# so the measured value sat OUTSIDE the randomisation envelope -- training never met it.
#
# Only BREAKAWAY is overridden. M7 is a static test: it measures the force to hold, which is the
# stick regime. It says nothing about kinetic friction ("Friction while moving isn't [measured]"),
# so COULOMB keeps the bench value 0.222 from constant-velocity ramps -- a different quantity,
# measured by a different method. The pair gives a Stribeck hump from 0.40 relaxing to 0.222.
# The DR scales both, so breakaway now spans 0.28-0.52, straddling the measurement.
#
# Hip/knee (M6C12) friction is NOT changed: sim coulomb 0.429 against M7's +-0.5 N·m stiction
# scatter is consistent, and the 0.30-0.56 DR envelope already covers it.
_ANKLE_BREAKAWAY_MEASURED = 0.40


def _walk_actuators_implicit():
    """Original walk plant: PhysX implicit PD, no friction/latency, light armature."""
    return {
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
    }


def _walk_actuators_modeled():
    """Bench-validated walk plant: explicit delayed PD + stick-slip friction + measured inertia.

    Contract PD gains (stiffness/damping) and effort limits are unchanged from the baseline.
    ``armature`` = measured reflected motor+gearbox inertia (JSON ``inertia``); the URDF link
    masses provide the load inertia on top. Friction (coulomb/breakaway/viscous) is the fitted
    stick-slip model; effort_limit is the internal actuator clip (effort_limit_sim left at its
    explicit-actuator default 1e9 so PhysX doesn't double-clip — keeps _CONTRACT_EFFORT as the
    single torque clamp, matching the firmware / bench torque cap semantics).
    """
    lf, af = _M6C12["friction"], _MAD5010["friction"]
    return {
        "legs": StickSlipDelayedPDActuatorCfg(
            joint_names_expr=_LEG_GROUP,
            effort_limit=_subset(_CONTRACT_EFFORT, _LEG_LEAVES),
            velocity_limit=10.0,
            stiffness=_subset(_CONTRACT_KP, _LEG_LEAVES),
            damping=_subset(_CONTRACT_KD, _LEG_LEAVES),
            armature=float(_M6C12["inertia"]),
            min_delay=_LEG_DELAY[0], max_delay=_LEG_DELAY[1],
            coulomb=float(lf["coulomb"]), breakaway=float(lf["breakaway"]),
            viscous=float(lf["viscous"]), stribeck_vel=float(lf["stribeck_vel"]),
            stick_vel=float(lf["stick_vel"]),
        ),
        "ankles": StickSlipDelayedPDActuatorCfg(
            joint_names_expr=_ANKLE_GROUP,
            effort_limit=_subset(_CONTRACT_EFFORT, _ANKLE_LEAVES),
            velocity_limit=10.0,
            stiffness=_subset(_CONTRACT_KP, _ANKLE_LEAVES),
            damping=_subset(_CONTRACT_KD, _ANKLE_LEAVES),
            armature=float(_MAD5010["inertia"]),
            min_delay=_ANKLE_DELAY[0], max_delay=_ANKLE_DELAY[1],
            coulomb=float(af["coulomb"]), breakaway=_ANKLE_BREAKAWAY_MEASURED,
            viscous=float(af["viscous"]), stribeck_vel=float(af["stribeck_vel"]),
            stick_vel=float(af["stick_vel"]),
        ),
    }


# Toggle: modeled plant on by default; set HUMANOID_ACTUATOR_MODEL=0 to train the baseline.
_WALK_USE_ACTUATOR_MODEL = os.environ.get("HUMANOID_ACTUATOR_MODEL", "1") not in ("0", "false", "False")

HUMANOID_BIPED_WALK_CFG = HUMANOID_BIPED_CFG.replace(
    # Same real per-joint firmware PD gains as the squat cfg (the deployed ESC/contract gains,
    # asymmetric, kp up to 68.4), but WITHOUT the squat init pose — the walk/velocity env sets
    # its own standing init_state. Point the walk task at this so the policy trains on the real
    # deployed plant instead of the uniform kp=20/kd=2 of HUMANOID_BIPED_CFG. REQUIRES RETRAINING
    # + re-export of deploy/walk. Gains verified against humanoid-studio/configs/humanoid_lite.json.
    actuators=(_walk_actuators_modeled() if _WALK_USE_ACTUATOR_MODEL else _walk_actuators_implicit()),
)
"""Humanoid biped for the WALK / velocity task: real per-joint firmware PD gains from the
policy contract (matches the ESC gains in humanoid-control / humanoid_lite.json), keeping
HUMANOID_BIPED_CFG's default init pose (the walk env overrides init_state itself). Use this
instead of HUMANOID_BIPED_CFG so the walk policy trains on the deployed hardware plant.

The leg/ankle actuators are the bench-validated stick-slip + delayed-PD motor models by default
(set HUMANOID_ACTUATOR_MODEL=0 for the original friction-free implicit baseline)."""
