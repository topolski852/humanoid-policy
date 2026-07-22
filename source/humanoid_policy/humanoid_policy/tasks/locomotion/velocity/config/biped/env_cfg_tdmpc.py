"""TD-MPC2-ONLY biped walk env: the same env as HumanoidBipedEnvCfg but with a STABILITY-GATED
reward instead of the additive g2c3 (PPO/Eureka) reward.

Why a separate cfg/task: the additive reward lets TD-MPC2's short-horizon planner trade uprightness
for command-tracking and topple. This cfg swaps in a multiplicative uprightness gate
(mdp.gated_locomotion, HumanoidBench-style) so reward -> 0 the instant the robot isn't upright.
Everything else (obs, actions, terminations, events, commands, the modeled actuator plant) is
inherited unchanged. The PPO reward (RewardsCfg / HumanoidBipedEnvCfg) is NOT touched.
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

import humanoid_policy.tasks.locomotion.velocity.mdp as mdp
from .env_cfg import HumanoidBipedEnvCfg, EventsCfg, _STAND_BASE_HEIGHT
from .env_cfg import TerminationsCfg as EpisodicTerminationsCfg  # 45° tilt / height-collapse resets

# Standing-height threshold for the gate: a touch below the nominal stand height so normal gait
# bob still counts as "standing". Fallback if the pose library was unavailable at import.
_STAND_H = (float(_STAND_BASE_HEIGHT) - 0.05) if _STAND_BASE_HEIGHT is not None else 0.50

# Collapse-reset height: DEEP (~30 cm below standing = torso essentially floored). Matching
# HumanoidBench (terminate only at qpos[2]<0.2, i.e. truly on the floor), a fall does NOT reset —
# the robot lies there earning ~0 for the REST of the episode. That makes falling COSTLY (you lose
# the rest of the horizon), which is what forces a sustained gait rather than lean-fall-reset. The
# earlier shallow -0.18 reset made falling cheap (quick respawn to a good stand) and removed that
# pressure -> the robot leaned and let itself reset. Only a true collapse resets now.
_HARD_COLLAPSE_H = (float(_STAND_BASE_HEIGHT) - 0.30) if _STAND_BASE_HEIGHT is not None else -10.0

# Tight-stand height: only ~12 cm of crouch allowed before the episode ends (vs 15 cm), forcing a
# TALLER stand. Used by the STAND phase's tightened terminations.
_STAND_TIGHT_H = (float(_STAND_BASE_HEIGHT) - 0.12) if _STAND_BASE_HEIGHT is not None else -10.0


@configclass
class NonEpisodicTerminationsCfg:
    """TD-MPC2/HumanoidBench-style episode structure: a fall does NOT end the episode. Only a
    true collapse (torso on the floor, which a biped can't self-right from) resets — this desyncs
    env resets and keeps the buffer from filling with dead floor-lying data, while removing the
    45-degree ``base_orientation`` termination that made "survive by standing still" the optimum.
    A mild tilt/stumble now costs ~0 reward (the gate) but lets the robot RECOVER instead of dying.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    hard_collapse = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": _HARD_COLLAPSE_H, "asset_cfg": SceneEntityCfg("robot", body_names="base")},
    )


@configclass
class HybridRewardsCfg:
    """HYBRID TD-MPC2 walk reward = the uprightness-GATED locomotion core (keeps the short-horizon
    planner from banking reward while tipping) + a small UNGATED upright bonus (cold-start gradient
    where the multiplicative gate is flat) + the proven PPO GAIT terms (feet_air_time / feet_slide,
    which teach the *stride* the velocity reward doesn't) + torso STABILITY / IMU-smoothness terms
    (low gyro+accel noise, smooth sim-to-real motion). Weights start from the PPO/Eureka walk (which
    walks on THIS robot) but scaled so the multiplicative gate stays the dominant positive. Stability
    weights are deliberately MODEST for the first run so they shape the gait without killing it —
    raise them once walking emerges (smoothness > raw progress once it's on its feet).

    Design rules: (a) the gated core is the main positive; (b) additive NEGATIVE penalties can only
    subtract, so they can't be farmed while falling; (c) additive POSITIVE shaping (upright bonus,
    feet_air_time) is kept small/moderate so it can't out-earn "actually walk while upright"."""

    # --- CORE: uprightness-gated linear-speed locomotion (dominant positive, per-step [0,1]) ---
    stand_walk = RewTerm(
        func=mdp.gated_locomotion,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "tracking_std": 0.25,          # unused by the linear-speed move term (kept for signature)
            "stand_height": _STAND_H,
            "stand_margin": 0.12,          # smooth height-gate gradient (positive margin)
            "upright_min": 0.8,
            "move_weight": 0.75,           # standing-still earns only the 0.25 baseline; moving earns the rest
        },
    )

    # --- COLD-START: small UNGATED upright bonus. Gives a smooth gradient toward vertical from any
    # tilt (where standing×upright is ~0 and flat), so a fallen/from-scratch policy can learn to
    # stand back up. Small so it can't become a "stand still" attractor. ---
    upright_bonus = RewTerm(func=mdp.upright_posture, weight=0.3)

    # --- GAIT (the PPO walk terms) but feet_air_time is UPRIGHTNESS-GATED so a fallen robot can't
    # farm airborne-foot reward by lying down and waving a leg (the run-6 hack). Only an upright
    # robot taking a real step is rewarded. This supplies the "how to step" signal the speed term
    # lacks; feet_slide punishes a planted foot dragging. ---
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_upright_gated,
        weight=1.5,   # credited only at foot-strike (sparse), so a bit stronger than the continuous version
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll"),
            "air_lo": 0.20, "air_hi": 0.40, "air_margin": 0.15,   # target step air-time band (s)
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.07,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll"),
        },
    )

    # --- SAFETY / light smoothness only. The heavy stability stack (ang_vel/flat_orientation/
    # base_accel/lin_vel_z/undesired_contacts) BARBERED the gait during learning (it suppresses the
    # torso sway + weight-shift a step needs), so it's dropped for the walk-LEARNING phase -- add
    # smoothness back once it walks, same as the stand plan. ---
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)        # light jerk penalty
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-0.1)       # joint safety


@configclass
class GentleEventsCfg(EventsCfg):
    """Gentler domain randomization so TD-MPC2 can learn to balance without being constantly
    knocked over. Drops the mid-episode pushes and softens the reset external force/torque
    (the harder-command DR was tuned for PPO's 590M-sample brute force)."""

    push_robot = None  # no mid-episode ±0.8 m/s shoves
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (-1.0, 1.0),    # softened from ±3
            "torque_range": (-1.0, 1.0),
        },
        mode="reset",
    )


@configclass
class HumanoidBipedTdmpcEnvCfg(HumanoidBipedEnvCfg):
    """Biped walk env with the stability-gated reward + gentler DR — for the TD-MPC2 trainer.
    PHASE 2 of the curriculum (full commands): warm-start from a phase-1 stand checkpoint."""

    rewards: HybridRewardsCfg = HybridRewardsCfg()
    events: GentleEventsCfg = GentleEventsCfg()
    terminations: NonEpisodicTerminationsCfg = NonEpisodicTerminationsCfg()


# =========================================================================================
# CURRICULUM PHASE 1 — learn to STAND (near-zero command, calm spawn). With the gated reward a
# zero command means "track zero velocity" = stand still, which earns full reward when upright.
# Once this stands cleanly, warm-start the walk task (phase 2) from its checkpoint
# (train.py --init_checkpoint <phase1 model_best>).
# =========================================================================================


@configclass
class StandEventsCfg(GentleEventsCfg):
    """Even gentler than phase 2: spawn calm and near the stand pose so it can learn to HOLD a
    stand before facing perturbations."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-0.3, 0.3)},
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                               "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
        },
        mode="reset",
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.98, 1.02), "velocity_range": (0.0, 0.0)},
    )


@configclass
class StandRewardsCfg:
    """PHASE-1 STAND reward (v2) = the recipe that PROVABLY stood before (stand phase-1: 0 falls).
    EPISODIC survival is the driver: a fall ends the episode + costs -1, so the only way to keep
    earning is to hold the upright spawn -> "survive by standing still" is exactly the goal here.
    MINIMAL penalties so the robot can freely make the corrective micro-movements a biped needs to
    balance (an inverted pendulum must move to stay up); the heavy stability stack (flat_orientation
    / ang_vel / base_accel / undesired_contacts) suppressed that and trapped v1 in a still slump.
    NO gait terms (nothing to farm by falling). Add smoothness back only AFTER it stands cleanly."""

    stand_hold = RewTerm(
        func=mdp.gated_locomotion,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "tracking_std": 0.25,
            "stand_height": _STAND_H,
            "stand_margin": 0.12,
            "upright_min": 0.8,
            "move_weight": 0.75,   # cmd=0 -> move term rewards stillness
        },
    )
    upright_bonus = RewTerm(func=mdp.upright_posture, weight=0.4)     # strong "be vertical" gradient
    # (stance_width reward removed: adding it (v3) destabilized the value and regressed the stand;
    #  forcing a wide base didn't help. Instead, TightStandTerminationsCfg forces a taller/more
    #  upright stand by making a lean/crouch END the episode.)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-1.0)  # episodic: falling costs
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-0.1)  # joint safety
    # --- STILLNESS / IMU smoothness (v5): the v4 stand survives (0 falls) but JITTERS at the edge
    # of the bounds (rocking/joint_vel/action_rate ~5x a calm stand). Now that it CAN stand, add
    # modest smoothness penalties (safe to add here that weren't during from-scratch learning, since
    # v5 WARM-STARTS a working stand) to calm the twitch into a still, smooth stand -> lower IMU
    # noise / better sim-to-real. Kept modest so the ~0.055/step stand reward stays positive. ---
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)       # torso roll/pitch rate (gyro)
    base_accel_xy_l2 = RewTerm(func=mdp.base_lin_accel_xy_l2, weight=-0.02)  # accel smoothness
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.3)  # torso level (also fights the edge-lean)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.03)     # jerk / micro-twitch (up from -0.01)


@configclass
class TightStandTerminationsCfg:
    """Tightened episodic terminations for the STAND phase: a LEAN past 30° (was 45°) or a crouch
    past ~12 cm ends the episode + resets to the upright spawn. This makes "surviving" require
    standing TALLER and more UPRIGHT — attacking the mediocre-lean ceiling where a robot passed the
    lenient 45° bound while parked in a poor posture. Stand-env only; PPO / walk env untouched."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 0.52, "asset_cfg": SceneEntityCfg("robot", body_names="base")},  # 30° (was 0.78=45°)
    )
    base_height = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": _STAND_TIGHT_H, "asset_cfg": SceneEntityCfg("robot", body_names="base")},
    )


@configclass
class HumanoidBipedTdmpcStandEnvCfg(HumanoidBipedTdmpcEnvCfg):
    """PHASE 1: learn to STAND — zero command, calm spawn, minimal stand reward. EPISODIC (unlike
    the walk env): a fall terminates and resets to the upright spawn, giving the survival pressure
    that makes standing the optimum (and lots of upright restarts). This mirrors the config that
    stood cleanly (stand phase-1). Warm-start the walk phase from this once it holds a solid stand."""

    rewards: StandRewardsCfg = StandRewardsCfg()
    events: StandEventsCfg = StandEventsCfg()
    terminations: TightStandTerminationsCfg = TightStandTerminationsCfg()  # lean/crouch -> reset

    def __post_init__(self):
        super().__post_init__()
        r = self.commands.base_velocity.ranges
        r.lin_vel_x = (0.0, 0.0)
        r.lin_vel_y = (0.0, 0.0)
        r.ang_vel_z = (0.0, 0.0)
        r.heading = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 1.0
        self.commands.base_velocity.heading_command = False
