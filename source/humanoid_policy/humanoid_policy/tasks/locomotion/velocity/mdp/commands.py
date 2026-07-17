"""Velocity command that also logs walking + stability READBACK metrics.

Subclasses Isaac Lab's ``UniformVelocityCommand`` to additionally emit, per episode:
  - ``Metrics/base_velocity/tracked_speed``    — mean actual speed achieved *in the
    commanded direction* (velocity projected onto the command). Works for ANY command
    direction (fwd/back/strafe), so it is the omnidirectional "is it locomoting" signal.
  - ``Metrics/base_velocity/commanded_speed``  — mean commanded planar speed; the
    denominator for tracked_ratio = tracked_speed / commanded_speed (0 statue -> 1 tracker).
  - ``Metrics/base_velocity/forward_speed``    — mean actual forward base speed (kept as a
    plain readback; only meaningful when the command happens to be forward)
  - ``Metrics/base_velocity/base_accel_rms``   — RMS horizontal base linear acceleration
    (smoothness / "fast IMU X/Y changes")
  - ``Metrics/base_velocity/rocking_rms``      — RMS roll/pitch base angular velocity
    (stability / rocking)

Any ``"/"``-containing metric key is written verbatim as a TensorBoard scalar by
rsl_rl (``logger.py``), so an external process (the eureka gated fitness) can read
whether the robot is really moving and how stable/smooth it is — without a second
Isaac process.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.utils.configclass import configclass

__all__ = ["WalkMetricsVelocityCommand", "WalkMetricsVelocityCommandCfg"]


class WalkMetricsVelocityCommand(UniformVelocityCommand):
    """``UniformVelocityCommand`` + per-episode walking/stability readback metrics."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        # extra per-episode metrics — finalized in reset(), logged by the base CommandTerm
        self.metrics["tracked_speed"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["commanded_speed"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["forward_speed"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["base_accel_rms"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["rocking_rms"] = torch.zeros(self.num_envs, device=self.device)
        # per-episode running sums (cleared at reset)
        self._tracked_sum = torch.zeros(self.num_envs, device=self.device)
        self._cmd_speed_sum = torch.zeros(self.num_envs, device=self.device)
        self._fwd_sum = torch.zeros(self.num_envs, device=self.device)
        self._accel_sq_sum = torch.zeros(self.num_envs, device=self.device)
        self._rock_sq_sum = torch.zeros(self.num_envs, device=self.device)

    def _update_metrics(self):
        # keep the base error_vel_xy/yaw accumulation + _step_count increment
        super()._update_metrics()
        data = self.robot.data
        vel_xy = data.root_lin_vel_b.torch[:, :2]           # actual planar velocity (base frame)
        cmd_xy = self.vel_command_b[:, :2]                   # commanded planar velocity (base frame)
        cmd_norm = torch.norm(cmd_xy, dim=1)
        # achieved speed IN the commanded direction (projection). Omnidirectional: fwd/back/
        # strafe all count as positive when tracked. 0 for standing-command envs (cmd_norm~0).
        proj = torch.where(
            cmd_norm > 1e-3,
            torch.sum(vel_xy * cmd_xy, dim=1) / cmd_norm.clamp_min(1e-3),
            torch.zeros_like(cmd_norm),
        )
        self._tracked_sum += proj
        self._cmd_speed_sum += cmd_norm
        # actual forward base speed (base-frame x) — plain readback
        self._fwd_sum += vel_xy[:, 0]
        # horizontal base linear acceleration magnitude^2 (IMU-like x/y jerk)
        acc_xy = data.body_lin_acc_w.torch[:, 0, :2]
        self._accel_sq_sum += torch.sum(torch.square(acc_xy), dim=1)
        # roll/pitch angular velocity magnitude^2 (rocking)
        rock_xy = data.root_ang_vel_b.torch[:, :2]
        self._rock_sq_sum += torch.sum(torch.square(rock_xy), dim=1)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        if env_ids is None:
            env_ids = slice(None)
        # finalize our per-episode means BEFORE super().reset() reads/logs self.metrics.
        # _step_count is still the episode length here (base reset zeros it afterward).
        denom = self._step_count[env_ids].clamp_min(1.0)
        self.metrics["tracked_speed"][env_ids] = self._tracked_sum[env_ids] / denom
        self.metrics["commanded_speed"][env_ids] = self._cmd_speed_sum[env_ids] / denom
        self.metrics["forward_speed"][env_ids] = self._fwd_sum[env_ids] / denom
        self.metrics["base_accel_rms"][env_ids] = torch.sqrt(self._accel_sq_sum[env_ids] / denom)
        self.metrics["rocking_rms"][env_ids] = torch.sqrt(self._rock_sq_sum[env_ids] / denom)
        extras = super().reset(env_ids)
        # zero our running sums for the next episode
        self._tracked_sum[env_ids] = 0.0
        self._cmd_speed_sum[env_ids] = 0.0
        self._fwd_sum[env_ids] = 0.0
        self._accel_sq_sum[env_ids] = 0.0
        self._rock_sq_sum[env_ids] = 0.0
        return extras


@configclass
class WalkMetricsVelocityCommandCfg(UniformVelocityCommandCfg):
    """``UniformVelocityCommandCfg`` wired to the metrics-logging command class."""

    class_type: type = WalkMetricsVelocityCommand
