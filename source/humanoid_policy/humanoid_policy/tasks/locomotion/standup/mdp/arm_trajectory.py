"""Scripted arm trajectory, exposed to the policy as a phase command.

The G1 squat<->stand task keeps the arms **out of the action space** (12 legs + waist_pitch =
13 DoF). The hands-down contact in the captured safety squat is incidental -- stability comes
from feet and shins, the arms carry no load (xr_teleoperate ``docs/HANDOFF.md`` S5) -- but the
arms still have to *finish* in the pose, and swinging 14 arm joints through a 5 s transition is a
real inertial disturbance the legs must reject.

So: drive the arms along a deterministic time-parameterized path, and put the path's **phase** in
the observation. The policy can then anticipate the disturbance instead of only reacting to it,
and the arm motion stays reproducible on hardware (it is a joint-space interpolation the deploy
side can replay open-loop).

The term writes joint position targets directly on the articulation. Those targets persist through
the decimated physics steps, and the ``JointPositionAction`` term only writes the 13 policy joints,
so the two never fight over a joint.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["ArmTrajectoryCommand", "ArmTrajectoryCommandCfg"]


def _smoothstep(x: torch.Tensor) -> torch.Tensor:
    """C1-continuous 0->1 ramp (zero velocity at both ends), for x already clamped to [0, 1]."""
    return x * x * (3.0 - 2.0 * x)


class ArmTrajectoryCommand(CommandTerm):
    """Interpolates the arms from ``start_pose`` to ``end_pose`` and emits the phase as a command.

    Command vector (per env): ``[phase, phase_rate]``.

    * ``phase`` in [0, 1] -- the smoothstepped progress of the arm motion.
    * ``phase_rate`` -- its time derivative, normalized by ``1/duration_s``; peaks mid-motion and
      is zero while waiting and after arrival. This is the term that tells the policy *when the
      shove is coming*, which raw phase alone conveys only implicitly.
    """

    cfg: "ArmTrajectoryCommandCfg"

    def __init__(self, cfg: "ArmTrajectoryCommandCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]

        names = list(cfg.start_pose.keys())
        missing = [n for n in names if n not in cfg.end_pose]
        if missing:
            raise ValueError(f"joints in start_pose but not end_pose: {missing}")
        self._joint_ids, resolved = self.robot.find_joints(names, preserve_order=True)
        self._start = torch.tensor([cfg.start_pose[n] for n in resolved], device=self.device, dtype=torch.float32)
        self._end = torch.tensor([cfg.end_pose[n] for n in resolved], device=self.device, dtype=torch.float32)
        unknown = [n for n in cfg.waypoint_offset if n not in resolved]
        if unknown:
            raise ValueError(f"waypoint_offset names joints not in the trajectory: {unknown}")
        self._waypoint = torch.tensor(
            [cfg.waypoint_offset.get(n, 0.0) for n in resolved], device=self.device, dtype=torch.float32
        )

        self._command = torch.zeros(self.num_envs, 2, device=self.device)
        self.metrics["arm_phase"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        return (
            f"ArmTrajectoryCommand: {len(self._joint_ids)} joints, "
            f"delay {self.cfg.delay_s}s -> duration {self.cfg.duration_s}s"
        )

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _raw_phase(self) -> torch.Tensor:
        """Linear, unclamped-then-clamped progress from episode time."""
        t = self._env.episode_length_buf.to(torch.float32) * self._env.step_dt
        return torch.clamp((t - self.cfg.delay_s) / self.cfg.duration_s, 0.0, 1.0)

    def _update_command(self):
        u = self._raw_phase()
        phase = _smoothstep(u)
        # d/du smoothstep = 6u(1-u); the command carries it scaled by 1/duration -> ~unit range
        rate = 6.0 * u * (1.0 - u)
        self._command[:, 0] = phase
        self._command[:, 1] = rate

        targets = self._start.unsqueeze(0) + phase.unsqueeze(1) * (self._end - self._start).unsqueeze(0)
        # Waypoint detour: a half-sine bump, zero at both endpoints, so the path bows away from
        # the straight line in joint space without moving either end of it. This is what keeps the
        # hands out of the thighs; see the cfg field docs.
        if bool(self._waypoint.any()):
            bump = torch.sin(torch.pi * phase)
            targets = targets + bump.unsqueeze(1) * self._waypoint.unsqueeze(0)
        self.robot.set_joint_position_target(targets, joint_ids=self._joint_ids)

    def _resample_command(self, env_ids: Sequence[int]):
        # Nothing to sample: the trajectory is deterministic and keyed to episode time.
        pass

    def _update_metrics(self):
        self.metrics["arm_phase"] = self._command[:, 0]


@configclass
class ArmTrajectoryCommandCfg(CommandTermCfg):
    """Config for :class:`ArmTrajectoryCommand`."""

    class_type: type = ArmTrajectoryCommand

    asset_name: str = "robot"
    start_pose: dict = {}
    """``{joint_name: radians}`` at phase 0 (the spawn pose's arm configuration)."""
    end_pose: dict = {}
    """``{joint_name: radians}`` at phase 1. Must cover the same joints as ``start_pose``."""
    waypoint_offset: dict = {}
    """``{joint_name: radians}`` added as a half-sine bump peaking at mid-motion (zero at both
    ends, so the endpoints are untouched).

    Not cosmetic. A straight joint-space line from the G1's stand pose to the captured safety
    squat drives **both hands through the thighs and knees** -- 22 of 61 samples in collision, up
    to 50 mm penetration -- even though both endpoints are clean. Swinging the shoulders out and
    back over the motion routes the hands around the folding legs. Validate any change with
    ``xr_teleoperate/tools/check_self_collision.py --from <stand>.yaml --to <squat>.yaml``.
    """
    delay_s: float = 0.5
    """Seconds of hold before the arms start moving (lets the spawn transient settle)."""
    duration_s: float = 3.0
    """Seconds spent traversing start -> end."""

    def __post_init__(self):
        # The trajectory is time-keyed, not resampled; make the resample interval effectively never.
        self.resampling_time_range = (1.0e9, 1.0e9)
