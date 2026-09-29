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
#   measured  the measured MECHANISMS (transport staleness, encoder quantisation) on top of an
#             explicit uncertainty budget -- see the budget below.
#   legacy    the pre-2026-09-23 guessed values: Unoise(+-0.05) rad and Unoise(+-2.0) rad/s, no
#             staleness, no quantisation. What smoothA -- still the best bundle on hardware --
#             trained on. Kept so the change stays A/B-able.
#
# ### WHY THE NOISE WENT BACK UP (2026-09-25)
#
# measA-full was the first bundle trained on the measured floors, and it REGRESSED badly on
# hardware: tilt rate 20.6 deg/s against smoothA's 0.4 (50x), joint_vel p99 5.09 against 0.73,
# and an ESC encoder fault at 41 s that took a CAN bus down. Operator ranking is
# smoothA > smoothB > measA. See humanoid-control docs/measurements/REPORT_2026-09-25_measA.md.
#
# The mechanism is that staleness robustness is PROPORTIONAL TO JOINT VELOCITY (obs = pos - tau*vel),
# and balancing happens at almost zero velocity. Using smoothA's measured standing figures
# (joint vel p95 0.06 rad/s, tau mean 4.88 ms) the whole budget in the standing regime was:
#
#     staleness tau*v  2.9e-4      quantisation 1.0e-4      sensor floor 4.3e-5
#     total          ~4.4e-4 rad   vs smoothA's flat 5.0e-2 rad  ->  ~114x LESS
#
# And the calibration offset measured on 2026-09-25 is +-0.029 rad -- SIXTY-SIX TIMES larger than
# the uncertainty measA was trained to tolerate. The policy learned to trust its encoders to a
# precision the robot cannot deliver, so a ~1.4 deg stance error reads as a large state error and
# it reacts hard. That is exactly the reported behaviour: it lurches rather than drifts.
#
# The measured floors are a LOWER BOUND on what to randomise, not the value to use. The fix is to
# keep the measured mechanisms and restore a velocity-INDEPENDENT margin for the plant error that
# is real but unmodelled (calibration drift, the unresolved 4x knee load path, backlash, link
# flex, IMU mounting error, ground irregularity).
#
# ### THE BUDGET -- keep these separate so the next regression is attributable
#
#   joint_pos = sensor floor 4.3e-5  +  plant margin 0.05   (measured + deliberate)
#   joint_vel = sensor floor 0.049   +  plant margin 2.0    (measured + deliberate)
#
# The margins are set to smoothA's values on purpose: it is the known-good bundle, so this is the
# minimal delta from something that works. If the next run behaves, the noise was the cause; if it
# still fails, the cause is the torque caps or the saturation reward, which measA changed too.
# Shrink the margins only once M7 and the encoder-drift test have accounted for the plant error
# they stand in for.
#
# base_ang_vel and projected_gravity keep their guessed scales in BOTH modes: the IMU auto-zeroes
# its gyro at rest, so B5 could not measure the floor. Do not invent one.
_OBS_MODELS = ("measured", "legacy")
_OBS_MODEL = os.environ.get("HUMANOID_OBS_MODEL", "measured").strip().lower()
if _OBS_MODEL not in _OBS_MODELS:
    raise ValueError(f"HUMANOID_OBS_MODEL={_OBS_MODEL!r} is not one of {sorted(_OBS_MODELS)}")
_OBS_MEASURED = _OBS_MODEL == "measured"

# Measured sensor floors (humanoid-control docs/measurements/TRAINING_INPUT.json).
_SENSOR_FLOOR_POS = 4.3e-5   # rad; 4.18e-5 / 4.32e-5 on two independent 600 s stands
_SENSOR_FLOOR_VEL = 0.049    # rad/s; median-of-window std on the filtered signal, max across joints
# Deliberate margin for measured-but-unmodelled and unmeasured plant error. NOT a sensor claim.
_PLANT_MARGIN_POS = 0.05     # rad/s ... rad; smoothA's value
_PLANT_MARGIN_VEL = 2.0      # rad/s; smoothA's value

_N_JOINT_POS_NOISE = (_SENSOR_FLOOR_POS + _PLANT_MARGIN_POS) if _OBS_MEASURED else 0.05
_N_JOINT_VEL_NOISE = (_SENSOR_FLOOR_VEL + _PLANT_MARGIN_VEL) if _OBS_MEASURED else 2.0
_OBS_MAX_LAG_S = mdp.MAX_LAG_S if _OBS_MEASURED else 0.0
# Velocity carries the transport delay PLUS the firmware's velocity EMA
# (velocity_filter_alpha = 0.7154 at 100 Hz -> tau_f ~ 8 ms), which the sim does not otherwise
# model. Dominated by the margin above at present; it starts to matter once that margin shrinks.
_OBS_MAX_LAG_S_VEL = mdp.MAX_LAG_S_VEL if _OBS_MEASURED else 0.0
_OBS_QUANTUM = mdp.ENCODER_QUANTUM_RAD if _OBS_MEASURED else 0.0
print(f"[INFO] walk observation model '{_OBS_MODEL}': "
      f"joint_pos_noise=±{_N_JOINT_POS_NOISE:.5g} (floor {_SENSOR_FLOOR_POS:g} + margin {_PLANT_MARGIN_POS:g}) "
      f"joint_vel_noise=±{_N_JOINT_VEL_NOISE:.5g} (floor {_SENSOR_FLOOR_VEL:g} + margin {_PLANT_MARGIN_VEL:g}) "
      f"lag_pos={_OBS_MAX_LAG_S}s lag_vel={_OBS_MAX_LAG_S_VEL}s quantum={_OBS_QUANTUM:g}")

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
        # 0.02 -> 0.30 on 2026-09-25. Standing unattended is the nearer milestone than walking:
        # NO policy to date has walked without the operator's hand on the robot, so every gait
        # figure in every report describes a SUPPORTED walk, while smoothA does stand alone for
        # 10 minutes and measA cannot stand at all (humanoid-control
        # docs/measurements/REPORT_2026-09-25_measA.md section 4). At 2% the policy had almost no
        # practice at the thing it is actually judged on. 0.30 matches Asimov's setting.
        rel_standing_envs=0.30,
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
                "max_lag_s": _OBS_MAX_LAG_S_VEL,
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
    # HARDWARE SAFETY, not smoothness. The ESC loses encoder tracking near 13 rad/s and floods
    # the CAN bus until it drops -- measA-full did exactly that at 41 s.
    #
    # THRESHOLD 2.0 -> 8.0 (2026-09-25). The first version cited "smoothA never exceeded
    # 1.08 rad/s", which is smoothA's STANDING figure; the hardware plan's section 3b made the
    # same conflation. smoothA's own WALK capture
    # (humanoid-control walk_20260923T104636_smoothA-walk1_M2M3M6.json) says otherwise:
    #
    #     smoothA standing   max 1.08 rad/s
    #     smoothA WALKING    median per-joint max 6.90, peaks 7.1-8.8 rad/s
    #
    # So the best bundle on hardware routinely reaches 7-9 rad/s while walking, and a hinge at
    # 2.0 penalised its normal gait continuously. measB-fast, trained that way, came out a
    # shuffle: knee swing 0.391 rad against smoothA's 0.849, knee correlation -0.158 against
    # -0.655, forward speed 0.209 on a 0.3 command -- while its safety numbers were the best of
    # any bundle (joint_vel p99 0.77, max 9.13). The constraint worked and cost the gait.
    #
    # 8.0 sits above smoothA's normal walking range and 5 rad/s below the observed fault, so it
    # fires on genuinely dangerous excursions rather than on locomotion. The weight is raised to
    # match: above 8 rad/s is a hardware-fault trajectory, not an inefficiency.
    # (Note smoothA itself peaks at 14.98 rad/s in sim -- past the 13.02 fault. It is not safe
    # either; it has just not been unlucky yet. That is what this term exists to prevent.)
    #
    # Tune against joint_vel_p99_rad_s / joint_vel_over_10_frac from
    # scripts/rsl_rl/eval_plant_compare.py, not by eye.
    dof_vel_excess = RewTerm(
        func=mdp.joint_vel_excess,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=HUMANOID_LEG_JOINTS), "max_vel": 8.0},
        weight=-0.5,
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
    # A joint past 10 rad/s is a HARDWARE FAULT on this robot: the ESC loses encoder tracking
    # near 13 rad/s, raises ERROR_ENCODER_FAULT (0x2000) and floods EMCY until the CAN bus drops
    # and needs a power cycle. measA-full ended that way 41 s into its first run.
    #
    # OFF BY DEFAULT -- it destroys training. Measured 2026-09-25 at 1024 envs, 250 iterations:
    #
    #     iter    vel_fault    ep_len    reward
    #        0      0.642        10.1     -3.03
    #       30      0.999         1.7     -1.08
    #      150      1.000         1.1     -0.58
    #
    # 100% of episodes terminate at ~1.2 steps and the policy never gets a locomotion gradient.
    # The cause is NOT the reset transient -- stepping with zero actions peaks at 8.03 rad/s and
    # never crosses 10 (0.00% of joint-steps). It is that an UNTRAINED policy (action std 1.0,
    # scale 0.25, kp 45) genuinely commands >10 rad/s on almost every step, so the constraint
    # fires before there is any behaviour to shape. A hard termination on a condition a random
    # policy violates constantly is a learning dead end, whatever the threshold.
    #
    # The hinge penalty ``dof_vel_excess`` carries the constraint instead: it is ~2500x stronger
    # than the ``dof_vel_l2`` measA trained under, and it shapes rather than forbids. The gate
    # before deploying is the MEASURED one -- ``joint_vel_p99_rad_s`` and
    # ``joint_vel_over_10_frac`` from scripts/rsl_rl/eval_plant_compare.py (hardware target
    # p99 < 1.5 rad/s). If a bundle clears that in sim, this termination would never have fired.
    #
    # Set HUMANOID_VEL_TERMINATION=1 to enable, e.g. to fine-tune an already-competent policy
    # where the dead-end failure above does not apply.
    if os.environ.get("HUMANOID_VEL_TERMINATION", "0") not in ("0", "false", "False"):
        joint_vel_fault = DoneTerm(
            func=mdp.joint_vel_out_of_manual_limit,
            params={
                "max_velocity": 10.0,
                "asset_cfg": SceneEntityCfg("robot", joint_names=HUMANOID_LEG_JOINTS),
            },
        )


@configclass
class EventsCfg:
    """Configuration for events."""

    # === Startup behaviors ===
    # Floor friction floor 0.4 -> 0.25 (2026-09-29). The terrain is 1.0 with multiply combine, so
    # this range IS the foot-ground coefficient the policy meets.
    #
    # Evidence (plant identification, humanoid-control ROBOT_PC_BRIEF_2026-09-29): replaying
    # measC-full at the hardware walking command (vx 0.6) on perturbed plants, foot mu 0.3 is the
    # ONLY change that reproduces the hardware torso twist -- body yaw rate p95 2.18 rad/s against
    # 1.4-2.5 measured on the robot and 0.60 on the old plant. Sagittal damping x8, inertia x3 and a
    # supporting-hand force do not. And the policy is weak there: 0.18 falls/min at mu 0.3 against
    # 0.04 on the old range. It had simply never been trained on a floor that slippery.
    #
    # Also closes an observation gap: on hardware the policy sees |ang_vel_z| of 1.4-2.5 rad/s every
    # step, a range the old plant almost never produced.
    #
    # Widened, not shifted: 0.25-1.2 contains the old 0.4-1.2, so this stays safe if the direct floor
    # measurement (brief item D) comes back grippier than 0.3 -- the cost is robustness we didn't
    # need, not a wrong plant. Floor friction is genuinely unknown across floors, which is exactly
    # the kind of parameter domain randomisation is for.
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.25, 1.2),
            "dynamic_friction_range": (0.25, 1.2),
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
