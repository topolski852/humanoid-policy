from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

from humanoid_policy_assets.actuators.limits import joint_effort_limits

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_vel_excess(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_vel: float = 8.0,
) -> torch.Tensor:
    """Penalize joint speed above ``max_vel`` (rad/s, summed over joints).

    This is a HARDWARE SAFETY constraint, not a smoothness preference. The ESC cannot track its
    encoder much above ~13 rad/s at the joint: it raises ``ERROR_ENCODER_FAULT`` (0x2000) and
    then floods EMCY until the bus drops and needs a power cycle. The measA-full bundle did
    exactly that 41 s into its first run -- ``right_knee_pitch`` ramped 1.7 -> 5.8 -> 13.0 rad/s
    and took ``can_right_leg`` down with 51,484 EMCY frames in 5.5 s (humanoid-control
    ``docs/measurements/REPORT_2026-09-25_measA.md`` section 1).

    Threshold rationale: smoothA, the best-behaved bundle on hardware, reaches a median
    per-joint max of **6.90 rad/s while WALKING** (peaks 7.1-8.8), against only 1.08 rad/s while
    standing. An earlier version of this term used the standing figure and hinged at 2.0, which
    penalised normal gait continuously and produced a shuffle. 8.0 rad/s sits above smoothA's
    walking range and 5 rad/s below the observed fault, so it prices hardware-fault trajectories
    rather than locomotion. See the note on ``dof_vel_excess`` in the biped ``RewardsCfg``.

    Deliberately NOT ``isaaclab.envs.mdp.joint_vel_limits``, which measures against
    ``soft_joint_vel_limits`` -- an articulation field fed by ``velocity_limit_sim``, which this
    robot's explicit actuators never set. The limit that matters is a property of the ESC
    firmware, so it belongs in the config as an explicit number rather than being inherited from
    a solver field. Paired with a termination at a higher ceiling; see ``TerminationsCfg``.
    """
    asset = env.scene[asset_cfg.name]
    vel = asset.data.joint_vel.torch[:, asset_cfg.joint_ids]
    return torch.sum(torch.relu(vel.abs() - max_vel), dim=1)


def joint_torque_saturation(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize torque DEMANDED beyond what the actuator can deliver (N·m, summed over joints).

    Measured motivation: on hardware the knees demand p95 28 N·m and sit at or over their cap
    on 52.1% / 33.3% of policy ticks (humanoid-control
    ``docs/measurements/REPORT_2026-09-23_smoothA.md`` sec 2). Raising the cap 6.0 -> 11.0 fixed
    the STATIC droop and did nothing for the dynamic demand, which is above even the motor's
    physical ceiling -- so this stopped being a limit-configuration problem and became a gait
    problem, i.e. training-side. This term is the direct lever: it prices the overshoot itself.

    Deliberately NOT ``isaaclab.envs.mdp.applied_torque_limits``, which computes
    ``|applied_torque - computed_torque|``. That identity only isolates the clip on a stock
    actuator. :class:`StickSlipDelayedPDActuator` overwrites ``applied_effort`` with
    ``clipped_PD - friction``, so the stock term would report clip-deficit PLUS friction -- a
    roughly 4.3 N·m floor across the 12 legs that is present whether or not anything saturates,
    swamping the signal we actually want.

    ``computed_torque`` is the pre-clip PD output, so ``relu(|tau| - limit)`` is exactly the
    unmet demand. It is zero whenever the joint is inside its limit, which keeps the term silent
    on a policy that already has headroom.

    Complements rather than replaces ``dof_torques_l2``: that penalizes torque MAGNITUDE
    everywhere and is dominated by the bulk of normal operation, this one is scoped to the peaks
    that touch the ceiling.
    """
    asset = env.scene[asset_cfg.name]
    tau = asset.data.computed_torque.torch[:, asset_cfg.joint_ids]
    limit = joint_effort_limits(asset)[:, asset_cfg.joint_ids]
    return torch.sum(torch.relu(tau.abs() - limit), dim=1)


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
