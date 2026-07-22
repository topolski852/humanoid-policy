from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# =============================================================================================
# GATED (multiplicative uprightness) reward — for the TD-MPC2 walk task ONLY.
#
# The PPO reward (RewardsCfg in config/biped/env_cfg.py) is ADDITIVE: the short-horizon TD-MPC2
# planner can bank tracking reward while the torso tips, then "pay" the fall penalty beyond its
# horizon -> it marches toward the command and topples. HumanoidBench (where TD-MPC2 walks
# humanoids in ~2M steps) instead MULTIPLIES the task reward by an uprightness/standing gate, so
# reward -> 0 the instant the robot is not upright: staying up is a prerequisite, not a competing
# term. `gated_locomotion` reproduces that structure. These functions are NOT used by the PPO cfg.
# =============================================================================================


def _tolerance(x: torch.Tensor, lower: float, upper: float, margin: float,
               value_at_margin: float = 0.1) -> torch.Tensor:
    """dm_control-style tolerance: 1 inside [lower, upper], gaussian decay to `value_at_margin`
    at `margin` beyond a bound, ->0 further out. margin<=0 -> hard 0/1 indicator."""
    in_bounds = (x >= lower) & (x <= upper)
    if margin <= 0.0:
        return in_bounds.float()
    d = torch.where(x < lower, lower - x, x - upper) / margin
    scale = math.sqrt(-2.0 * math.log(value_at_margin))
    decay = torch.exp(-0.5 * (d * scale) ** 2)
    return torch.where(in_bounds, torch.ones_like(x), decay)


def gated_locomotion(
    env: ManagerBasedRLEnv,
    command_name: str,
    tracking_std: float,
    stand_height: float,
    upright_min: float = 0.8,
    move_weight: float = 0.5,
    stand_margin: float = 0.12,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Uprightness-GATED velocity-tracking reward (HumanoidBench-style), per env, in [0, 1].

        stand_gate  = standing(base_height) * upright(torso)              # both 0..1
        move        = command tracking (lin + yaw), 0..1
        reward      = stand_gate * small_control * ((1-move_weight) + move_weight * move)

    If the robot is not near standing height AND upright, ``stand_gate -> 0`` and the whole reward
    collapses regardless of tracking — so it cannot earn reward by toppling toward the command.
    Being upright-and-still already earns the ``(1-move_weight)`` baseline; tracking the command
    adds the rest (upright first, then walk)."""
    asset = env.scene[asset_cfg.name]
    data = asset.data

    # standing: base world-z above the nominal standing height. Use an explicit POSITIVE margin so
    # the gate DECAYS SMOOTHLY as the base sags below stand_height (gives a gradient to lift the base
    # back up). stand_height is negative in this sim frame, so the old `stand_height*0.5` margin was
    # <0 -> _tolerance fell back to a hard 0/1 indicator with no gradient in the sag band (the robot
    # got stuck in a stable crouch earning ~0 with nothing pulling it up).
    h = data.root_pos_w.torch[:, 2]
    standing = _tolerance(h, lower=stand_height, upper=float("inf"), margin=stand_margin)
    # upright: -projected_gravity_z in body frame (~1 upright, 0 on its side)
    up = -data.projected_gravity_b.torch[:, 2]
    upright = _tolerance(up, lower=upright_min, upper=float("inf"), margin=upright_min)
    stand_gate = standing * upright

    # move (0..1): command-aware LINEAR speed reward (HumanoidBench-style). Reward the velocity
    # COMPONENT ALONG THE COMMANDED DIRECTION, ramping linearly 0->1 as it reaches the commanded
    # speed. This is the key difference from a loose Gaussian velocity-tracking term: undershoot is
    # penalized PROPORTIONALLY, so leaning to fake a little transient velocity scores low and only a
    # real, sustained gait at ~command speed scores full. (HumanoidBench demands >=1 m/s COM speed,
    # unfakeable by leaning; we demand the commanded speed, up to 0.8 m/s here.) `tracking_std` is
    # no longer used (kept in the signature for cfg compatibility).
    cmd = env.command_manager.get_command(command_name)[:, :2]          # (N,2) commanded planar vel (yaw frame)
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w.torch),
                                 asset.data.root_lin_vel_w.torch[:, :3])[:, :2]  # (N,2) actual planar vel
    cmd_speed = torch.norm(cmd, dim=1)                                  # (N,)
    moving = cmd_speed > 0.1
    cmd_dir = cmd / cmd_speed.clamp_min(1e-6).unsqueeze(-1)
    v_along = (vel_yaw * cmd_dir).sum(dim=1)                            # progress speed toward the command
    move_go = (v_along / cmd_speed.clamp_min(1e-6)).clamp(0.0, 1.0)     # linear 0->1 at commanded speed
    speed = torch.norm(vel_yaw, dim=1)
    move_stand = _tolerance(speed, lower=0.0, upper=0.0, margin=0.5)    # reward stillness when told to stand
    move = torch.where(moving, move_go, move_stand)

    # small-control factor in [0.8, 1] (mild preference for small actions -> smoother gait)
    act = env.action_manager.action
    ctrl = _tolerance(torch.norm(act, dim=-1) / act.shape[-1], lower=0.0, upper=0.0, margin=1.0)
    small_control = (4.0 + ctrl) / 5.0

    return stand_gate * small_control * ((1.0 - move_weight) + move_weight * move)


def upright_posture(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """UNGATED torso-uprightness bonus in [0,1] (1 = torso vertical, 0 = on its side). This is the
    cold-start fix: the multiplicative ``stand_gate`` (standing×upright) is ~0 with a near-flat
    gradient when the robot is down, so a from-scratch / fallen policy gets no signal to stand back
    up. Added as a small *additive* term, this gives a smooth monotonic gradient toward vertical
    from ANY tilt. It also rewards holding a steady vertical torso (lower IMU tilt) -> smoother
    motion / better sim-to-real. Kept small so walking (the gated move term) still dominates."""
    asset = env.scene[asset_cfg.name]
    return torch.clamp(-asset.data.projected_gravity_b.torch[:, 2], 0.0, 1.0)


def feet_stance_width(
    env: ManagerBasedRLEnv,
    lower: float = 0.23,
    upper: float = 0.30,
    margin: float = 0.10,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_ankle_roll"),
) -> torch.Tensor:
    """Reward a stable base of support: horizontal foot-to-foot distance kept in the target BAND
    [lower, upper], decaying to ~0 for `margin` beyond either edge. This both (a) gives a gradient to
    widen from the narrow neutral (~0.17 m, feet under the hips = tips easily) toward the band, and
    (b) actively DISCOURAGES over-widening into the splits (reward falls off above `upper`) -- a hard
    upper limit, since a very wide/low stance would otherwise be ultra-stable and get farmed via
    episode survival. Peaks in [0.23, 0.30] m; ~0.1 by 0.40 m."""
    asset = env.scene[asset_cfg.name]
    foot_xy = asset.data.body_pos_w.torch[:, asset_cfg.body_ids, :2]     # (N, 2, 2) horizontal pos of both feet
    sep = (foot_xy[:, 0, :] - foot_xy[:, 1, :]).norm(dim=-1)             # (N,) foot-to-foot horizontal distance
    return _tolerance(sep, lower=lower, upper=upper, margin=margin)


def base_lin_accel_xy_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize horizontal base linear acceleration for a SMOOTH walk.

    Uses the root body's world-frame linear acceleration (Isaac Lab ``body_lin_acc_w``,
    body index 0). Squaring the x/y components discourages jerky base motion — i.e.
    "penalize fast IMU changes in X/Y" — so the gait stays smooth rather than stompy.
    """
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.body_lin_acc_w.torch[:, 0, :2]), dim=1)


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt).torch[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time.torch[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time.torch[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time.torch[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward

def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history.torch[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w.torch[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w.torch), asset.data.root_lin_vel_w.torch[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w.torch[:, 2])
    return torch.exp(-ang_vel_error / std**2)
