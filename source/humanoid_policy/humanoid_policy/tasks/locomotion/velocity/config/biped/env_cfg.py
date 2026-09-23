import math
import os

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

# --- walk smoothness sweep (docs/walk-smoothness-sweep.md) --------------------------------
# The deployed policy's gait runs at 5.1 Hz, which the real robot cannot execute (peak demanded
# torque 47.9 Nm vs a 26.9 Nm motor ceiling; knees sag IN PHASE, corr +0.91). Runtime fixes are
# exhausted -- an action low-pass filter smooths the gait only by deleting it. The lever that is
# left is training: these smoothness penalties exist but are effectively switched off.
#
# Select with HUMANOID_SMOOTH_PRESET. Only these three weights change; dof_acc_l2 is unchanged.
#   off  (default)  the shipped values -- reproduces the 5.1 Hz gait
#   A    light      ~13% of the tracking reward
#   B    moderate   ~27%   <-- run first, most likely to land in the useful band
#   C    strong     ~60%   <-- collapse-into-standing risk lives here; a plausible outcome
#
# Screen the result with scripts/screen_gait.py -- judge on gait frequency and whether it still
# steps, NEVER on reward: a policy collapsed into standing scores well on smoothness.
_SMOOTH_PRESETS = {
    "off": (-0.014, 0.0,    0.0),
    "a":   (-0.05,  -0.002, -1e-4),
    "b":   (-0.10,  -0.005, -3e-4),
    "c":   (-0.20,  -0.010, -1e-3),
    # D = midpoint of A and B. Sweep result 2026-08-25: A and B both land in-band but each
    # fails a different secondary target (A: S(da^2) 2.24 > 1.5 at speed; B: knee corr -0.29
    # and swing 0.092 rad at vx=0.40). C collapsed outright, so the cliff is past B.
    "d":   (-0.075, -0.0035, -2e-4),
}
_SMOOTH_PRESET = os.environ.get("HUMANOID_SMOOTH_PRESET", "off").strip().lower()
if _SMOOTH_PRESET not in _SMOOTH_PRESETS:
    raise ValueError(
        f"HUMANOID_SMOOTH_PRESET={_SMOOTH_PRESET!r} is not one of {sorted(_SMOOTH_PRESETS)}"
    )
_W_ACTION_RATE, _W_ACTION_L2, _W_DOF_VEL = _SMOOTH_PRESETS[_SMOOTH_PRESET]
print(f"[INFO] walk smoothness preset '{_SMOOTH_PRESET}': "
      f"action_rate_l2={_W_ACTION_RATE} action_l2={_W_ACTION_L2} dof_vel_l2={_W_DOF_VEL}")

# ---------------------------------------------------------------------------------------------
# Observation model, selectable with HUMANOID_OBS_MODEL (default "measured").
#
#   measured  joint_pos/joint_vel carry the MEASURED transport staleness (tau ~ U(0,10 ms)) and
#             encoder quantisation (1.023e-4 rad), with per-step noise cut to the measured floor.
#   legacy    the pre-2026-09-23 guessed values: Unoise(+-0.05) rad and Unoise(+-2.0) rad/s, no
#             staleness, no quantisation. Kept so the change is A/B-able against every bundle
#             trained before the hardware captures existed.
#
# The two are NOT a simple "more noise / less noise" pair. The legacy fixed +-0.05 rad was a
# stand-in for the PEAK of a velocity-dependent effect; "measured" replaces it with the effect
# itself. See mdp/observations.py for the derivation and the capture references.
#
# Noise scales under "measured":
#   joint_pos  +-4.3e-5 rad   measured floor, 4.18e-5 / 4.32e-5 on two independent 600 s stands
#   joint_vel  +-0.05 rad/s   DERIVED, NOT MEASURED -- see below
#
# joint_vel's floor was never characterised: the report notes the firmware low-passes velocity
# (velocity_filter_alpha = 0.7154) before it is visible. Differentiating a 4.2e-5 rad position
# signal at 100 Hz puts the floor near 0.006 rad/s, so +-0.05 is ~8x above the derived floor and
# ~40x tighter than legacy -- deliberately conservative pending measurement. It IS measurable
# from captures already on the robot PC (PDO4 carries velocity in payload bytes 4-7); that is
# item 1 of the hardware plan. Tighten it here once the number exists.
#
# base_ang_vel and projected_gravity keep their guessed scales in BOTH modes: the IMU auto-zeroes
# its gyro at rest, so B5 could not measure the floor. Do not invent one.
_OBS_MODELS = ("measured", "legacy")
_OBS_MODEL = os.environ.get("HUMANOID_OBS_MODEL", "measured").strip().lower()
if _OBS_MODEL not in _OBS_MODELS:
    raise ValueError(f"HUMANOID_OBS_MODEL={_OBS_MODEL!r} is not one of {sorted(_OBS_MODELS)}")
_OBS_MEASURED = _OBS_MODEL == "measured"
_N_JOINT_POS_NOISE = 4.3e-5 if _OBS_MEASURED else 0.05
_N_JOINT_VEL_NOISE = 0.05 if _OBS_MEASURED else 2.0
_OBS_MAX_LAG_S = mdp.MAX_LAG_S if _OBS_MEASURED else 0.0
_OBS_QUANTUM = mdp.ENCODER_QUANTUM_RAD if _OBS_MEASURED else 0.0
print(f"[INFO] walk observation model '{_OBS_MODEL}': "
      f"joint_pos_noise=±{_N_JOINT_POS_NOISE} joint_vel_noise=±{_N_JOINT_VEL_NOISE} "
      f"max_lag_s={_OBS_MAX_LAG_S} quantum={_OBS_QUANTUM}")

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
            # Harder command envelope (balanced) so the full-profile policy is robust across
            # the real operating range, not just a gentle 0.3 m/s forward walk. Widened from
            # Berkeley's ±0.5 / ±0.25 / ±1.0. Paired with interval pushes below.
            lin_vel_x=(-0.8, 0.8),
            lin_vel_y=(-0.4, 0.4),
            ang_vel_z=(-1.2, 1.2),
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
        # joint_pos/joint_vel carry the measured CAN transport staleness and encoder
        # quantisation (see mdp/observations.py and the HUMANOID_OBS_MODEL block above).
        # Under HUMANOID_OBS_MODEL=legacy these collapse to the old plain joint_pos_rel /
        # joint_vel_rel behaviour, because max_lag_s and quantum both become 0.
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel_stale,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=HUMANOID_LEG_JOINTS, preserve_order=True),
                "max_lag_s": _OBS_MAX_LAG_S,
                "quantum": _OBS_QUANTUM,
            },
            noise=Unoise(n_min=-_N_JOINT_POS_NOISE, n_max=_N_JOINT_POS_NOISE),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel_stale,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=HUMANOID_LEG_JOINTS, preserve_order=True),
                "max_lag_s": _OBS_MAX_LAG_S,
            },
            noise=Unoise(n_min=-_N_JOINT_VEL_NOISE, n_max=_N_JOINT_VEL_NOISE),
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True

    @configclass
    class CriticCfg(PolicyCfg):
        """Observations for critic group."""
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)

        # The critic is privileged and must NOT inherit the actor's sensing defects.
        # enable_corruption=False already suppresses the noise, but staleness and quantisation
        # live inside the observation function, so they have to be switched off explicitly.
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel_stale,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=HUMANOID_LEG_JOINTS, preserve_order=True),
                "max_lag_s": 0.0,
                "quantum": 0.0,
            },
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel_stale,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=HUMANOID_LEG_JOINTS, preserve_order=True),
                "max_lag_s": 0.0,
            },
        )

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
        weight=_W_ACTION_RATE,
    )
    # action_l2 / dof_vel_l2: hardware-safety penalties (docs/walk-policy-divergence-report.md
    # §4B) that suppress high-frequency, large-amplitude actions. Berkeley/g2c3 leave them OFF
    # (they had over-damped the gait into standing); re-introduce with small weights for sim->real.
    action_l2 = RewTerm(
        func=mdp.action_l2,
        weight=_W_ACTION_L2,
    )
    dof_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=HUMANOID_LEG_JOINTS)},
        weight=_W_DOF_VEL,
    )
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=HUMANOID_LEG_JOINTS)},
        weight=-0.001783,
    )
    # Torque demanded BEYOND the actuator cap, in N·m. Measured motivation: the knees sit at or
    # over their limit on 52.1% / 33.3% of policy ticks on hardware, demanding p95 28 N·m -- over
    # even the motor's physical ceiling (humanoid-control REPORT_2026-09-23_smoothA.md sec 2).
    # Raising the cap fixed the static droop and not the dynamic demand, so the remaining lever
    # is training. Distinct from dof_torques_l2, which prices torque MAGNITUDE everywhere and is
    # dominated by ordinary operation; this one is silent until the joint runs out of headroom.
    # Starting weight -0.02. Tune against torque_sat_frac from scripts/rsl_rl/eval_plant_compare.py
    # rather than by eye.
    #
    # READ THIS BEFORE EXPECTING IT TO FIX THE KNEES. Replaying the smoothA-full bundle on the
    # modeled plant (2026-09-23, --num_envs 128 --steps 600 --cmd_vx 0.3) shows the sim does NOT
    # reproduce the hardware knee demand at all:
    #
    #     joint              sim sat    hw sat    sim p95   hw p95
    #     left_knee_pitch      0.16%     52.1%      6.6      28.0
    #     right_knee_pitch     0.17%     33.3%      7.1      23.1
    #     left_ankle_pitch    27.58%       n/a     15.3       n/a
    #     right_ankle_pitch   21.57%       n/a     14.0       n/a
    #
    # So in sim this term acts on the ANKLES, not the knees -- there is essentially no knee
    # overshoot to price. The knee gap is a PLANT gap (the simulated knee tracks its target; the
    # real one sits 0.208 rad behind, and that error is what produces the 28 N·m reconstructed
    # demand), and no reward weight closes it. Ankle saturation at 22-28% is a genuine problem in
    # its own right and worth penalizing, but do not read a drop in it as progress on the knees.
    # The measurement that can explain the knee discrepancy is M7 (commanded vs reported torque
    # under static load) -- see humanoid-control docs/HARDWARE_PLAN_2026-09-23.md.
    dof_torque_saturation = RewTerm(
        func=mdp.joint_torque_saturation,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=HUMANOID_LEG_JOINTS)},
        weight=-0.02,
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
            # Stronger reset perturbation (balanced robustness pass): ±3 N/N·m (was ±2).
            "force_range": (-3.0, 3.0),
            "torque_range": (-3.0, 3.0),
        },
        mode="reset",
    )

    # === Interval behaviors ===
    # Mid-episode shove (balanced robustness pass): every 10-15 s, apply a ±0.8 m/s velocity
    # kick to the base so the policy learns to recover from disturbances (the on-robot failure
    # mode). Enabled for the harder-command training; drop the weight/range to soften.
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.8, 0.8), "y": (-0.8, 0.8)}},
    )


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
