"""Measured observation model for the leg joints — transport staleness + encoder quantisation.

Everything here is calibrated against the 2026-09-23 hardware captures in ``humanoid-control``
(``docs/measurements/REPORT_2026-09-23_smoothA.md`` / ``_smoothB.md``), which characterised the
real CAN signal path for the first time. Two independent 600 s stands agreed closely, so the
numbers below are reproduced rather than one-off.

**What the robot actually does.** The ESCs broadcast PDO4 at 100 Hz; the policy consumes them at
25 Hz. So each observation is a sample of uniformly random age in 0-10 ms:

    sample age   mean 4.78 / 4.84 ms, p95 9.5 ms       -> tau ~ U(0, 10 ms)
    spread ACROSS joints  0.17-0.84 ms (2-8% of the frame period)
    stale-hold   0.00% on every joint, 1.4M frames     -> no repeat/dropout term
    frame loss   0.10% at the transport layer, invisible at the observation layer

The cross-joint spread is small enough that a single ``tau`` shared by all 12 joints is the
faithful model. Asimov's per-CAN-slot staggering (isaac-sim/IsaacLab#7071) does NOT transfer:
they have a different bus layout, and this robot shows no poll-slot structure.

**Why staleness and not just noise.** A joint read ``tau`` late reports ``pos(t) - tau*vel(t)``.
At 10 rad/s and 4.78 ms that is ~0.048 rad -- roughly 1100x the measured 4.2e-5 rad encoder
floor. The observation error is dominated by staleness, not by sensor noise, and it scales with
speed. The legacy fixed ``Unoise(±0.05)`` was a reasonable stand-in for the PEAK of that, but it
has the wrong shape: constant when it should track velocity. Shrinking it to the encoder floor
WITHOUT modelling staleness would make the sim less faithful, not more -- the policy would learn
to trust its position readings exactly when they are most stale.

The first-order expansion is used rather than a ring buffer because 0-10 ms is a fraction of the
40 ms policy tick; whole-frame delay cannot represent a sub-tick lag at all.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# 2*pi / cpr / gear = 2*pi / 4096 / 15. Identical on all 12 joints (same encoder and gearing),
# predicted analytically and then confirmed on hardware: the dithering joints at rest move by
# exactly 1.00 count, 0.000102 rad peak-to-peak.
ENCODER_QUANTUM_RAD = 1.023e-4

# Uniform 0-10 ms, from the measured age distribution (mean 4.78/4.84 ms -> a U(0,10ms) mean of
# 5.0 ms; measured p95 9.5 ms -> U(0,10ms) p95 of 9.5 ms). The match is close enough that no
# shaping is warranted.
MAX_LAG_S = 0.010

# Velocity is older than position by the firmware's velocity EMA, which the sim has no other
# model for. ``velocity_filter_alpha = 0.7154`` at the 100 Hz feed
# (humanoid-esc-firmware Core/Src/encoder.c) is a time constant of
# -dt / ln(1 - alpha) = -0.01 / ln(0.2846) ~= 8 ms, on top of the 10 ms transport delay.
# Derived from a firmware constant, not measured on the robot.
MAX_LAG_S_VEL = 0.018


def _lag(env: ManagerBasedRLEnv, max_lag_s: float) -> torch.Tensor:
    """Per-env sample age for this step, shape ``(num_envs, 1)``.

    One draw per environment rather than per joint: measured cross-joint spread is 0.17-0.84 ms
    against a 10 ms frame period, i.e. the joints are effectively read together. Resampled every
    step because the policy and the PDO4 timers run on independent clocks, so consecutive ticks
    see uncorrelated ages.
    """
    return torch.rand(env.num_envs, 1, device=env.device) * max_lag_s


def joint_pos_rel_stale(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_lag_s: float = MAX_LAG_S,
    quantum: float = ENCODER_QUANTUM_RAD,
) -> torch.Tensor:
    """``joint_pos_rel`` as the robot actually reports it: ``tau`` stale, then quantised.

    Pass ``max_lag_s=0.0`` for a ground-truth read (the critic — it is privileged and should not
    be handed the actor's sensing defects). ``quantum=0.0`` disables quantisation.
    """
    asset = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids
    pos = asset.data.joint_pos.torch[:, ids] - asset.data.default_joint_pos.torch[:, ids]
    if max_lag_s > 0.0:
        pos = pos - _lag(env, max_lag_s) * asset.data.joint_vel.torch[:, ids]
    if quantum > 0.0:
        # The encoder reports whole counts; the fractional part is never transmitted.
        pos = torch.round(pos / quantum) * quantum
    return pos


def joint_vel_rel_stale(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_lag_s: float = MAX_LAG_S_VEL,
) -> torch.Tensor:
    """``joint_vel_rel`` carrying the same transport staleness: ``vel(t-tau) ~= vel - tau*acc``.

    Not quantised: velocity is derived inside the ESC firmware and low-pass filtered
    (``velocity_filter_alpha = 0.7154``) before we ever see it, so it does not arrive on the
    encoder's count grid.

    Pass ``max_lag_s=0.0`` for the critic.
    """
    asset = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids
    vel = asset.data.joint_vel.torch[:, ids] - asset.data.default_joint_vel.torch[:, ids]
    if max_lag_s > 0.0:
        vel = vel - _lag(env, max_lag_s) * asset.data.joint_acc.torch[:, ids]
    return vel
