"""Stick-slip + delayed PD actuator — the real leg-motor plant, ported into Isaac Lab.

This is the trainer-side twin of the bench-validated forward model in the sibling repo
``humanoid-tuner`` (``sim/motor_model.py`` :class:`FrictionModel` +
``sim/isaac/substrate.py`` application pattern). It closes three gaps the stock
``ImplicitActuatorCfg`` walk plant left open (and which are the leading suspects for the
deployed policy's on-robot instability):

  1. **Reflected inertia** — set via the actuator ``armature`` in the robot cfg (solver
     param, not this class): legs 0.025, ankles 0.0137 kg·m² (measured motor+gearbox).
  2. **Stick-slip joint friction** — Karnopp static hold + Stribeck slip + viscous. Stock
     implicit PD has none; the real geared joint has a ~0.4 N·m breakaway deadzone.
  3. **Command->response latency** — 7.2 ms (legs) / 12 ms (ankles), via the inherited
     :class:`DelayedPDActuator` per-physics-step delay buffer.

Fidelity note: the walk env approximates the firmware PID with Isaac's explicit PD
(``stiffness``·err_pos + ``damping``·err_vel), NOT the tuner's exact 2 kHz
FirmwarePositionController (no integrator / torque EMA). We keep that PD unchanged — it IS
the policy<->robot contract — and only subtract friction on top, exactly as
``substrate.py`` does: ``effort = tau_ctrl - FrictionModel.torque(vel, drive=tau_ctrl,
load=0)``. External load is 0 here (the multibody solver owns joint/gravity load), so the
load-dependent friction term is inactive — matching substrate.py's ``tau_load=0.0``.

Friction levels (coulomb/breakaway/viscous) are held as per-env×joint tensors so domain
randomization can scale them for joint-to-joint 3D-printed spread (see the
``randomize_stickslip_friction`` event).
"""

from __future__ import annotations

import torch

from isaaclab.actuators import DelayedPDActuator, DelayedPDActuatorCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.types import ArticulationActions


def stickslip_friction_torque(
    vel: torch.Tensor,
    drive: torch.Tensor,
    coulomb: torch.Tensor,
    breakaway: torch.Tensor,
    stribeck_vel: float,
    viscous: torch.Tensor,
    stick_vel: float,
) -> torch.Tensor:
    """Karnopp stick-slip dry friction — a torch port of humanoid-tuner ``FrictionModel.torque``.

    All of ``vel``, ``drive`` and the friction-level tensors are (num_envs, num_joints);
    ``stribeck_vel``/``stick_vel`` are scalars. ``tau_load`` is fixed at 0 (multibody solver
    owns the load), so ``fc``/``fs`` carry no load term — reproducing substrate.py exactly.

      * stick regime (|vel| < stick_vel): friction opposes the drive torque, saturating at
        the breakaway level ``fs`` -> the static-hold deadzone.
      * slip regime: Stribeck kinetic curve relaxing breakaway -> coulomb, opposing motion,
        plus viscous ``viscous·vel``.
    """
    fc = coulomb
    fs = torch.maximum(breakaway, coulomb)

    stuck = vel.abs() < stick_vel
    # static: hold against drive, saturating at +-fs (the stiction deadzone)
    static = torch.minimum(torch.maximum(drive, -fs), fs)
    # kinetic: Stribeck curve (breakaway -> coulomb) opposing motion, + viscous
    kinetic = (fc + (fs - fc) * torch.exp(-((vel / stribeck_vel) ** 2))) * torch.sign(vel) + viscous * vel
    return torch.where(stuck, static, kinetic)


class StickSlipDelayedPDActuator(DelayedPDActuator):
    """Delayed explicit PD whose delivered torque has the sticky-gearbox friction removed.

    Inherits the delay buffer + effort-limit clip from :class:`DelayedPDActuator`; the only
    addition is subtracting :func:`stickslip_friction_torque` from the (delayed, clipped) PD
    output each physics step, mirroring humanoid-tuner ``sim/isaac/substrate.py``.
    """

    cfg: "StickSlipDelayedPDActuatorCfg"

    def __init__(self, cfg: "StickSlipDelayedPDActuatorCfg", *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        # Friction levels as (num_envs, num_joints) tensors (per-joint dicts or a group float),
        # so the DR event can scale them per env/joint. Reuse the base parser for dict/regex/float.
        self.coulomb = self._parse_joint_parameter(cfg.coulomb, 0.0)
        self.breakaway = self._parse_joint_parameter(cfg.breakaway, 0.0)
        self.viscous = self._parse_joint_parameter(cfg.viscous, 0.0)
        # velocity-scale constants (not randomized): plain floats
        self.stribeck_vel = float(cfg.stribeck_vel)
        self.stick_vel = float(cfg.stick_vel)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # DelayedPDActuator.compute: delay the setpoints, run IdealPDActuator PD, clip to
        # effort_limit -> control_action.joint_efforts is the clamped commanded torque (tau_ctrl).
        control_action = super().compute(control_action, joint_pos, joint_vel)
        tau_ctrl = control_action.joint_efforts
        friction = stickslip_friction_torque(
            joint_vel, tau_ctrl, self.coulomb, self.breakaway, self.stribeck_vel, self.viscous, self.stick_vel
        )
        # net delivered torque = commanded - friction (drive=tau_ctrl, load=0), per substrate.py.
        net = tau_ctrl - friction
        self.applied_effort = net
        control_action.joint_efforts = net
        return control_action


@configclass
class StickSlipDelayedPDActuatorCfg(DelayedPDActuatorCfg):
    """Config for :class:`StickSlipDelayedPDActuator`.

    Adds the stick-slip friction parameters on top of the delayed-PD config. Each field may
    be a single float (whole group) or a ``{joint_regex: value}`` dict. ``min_delay`` /
    ``max_delay`` (inherited) are the command latency in **physics steps**; with sim dt =
    5 ms that is the per-reset-randomized transport lag (also serves as the latency DR).
    """

    class_type: type = StickSlipDelayedPDActuator

    coulomb: dict[str, float] | float | None = None
    """Sliding Coulomb friction level F_c (N·m)."""

    breakaway: dict[str, float] | float | None = None
    """Static/stiction peak F_s (N·m); >= coulomb gives the breakaway hump."""

    stribeck_vel: float = 0.05
    """Velocity scale over which stiction decays to sliding Coulomb (rad/s)."""

    viscous: dict[str, float] | float | None = None
    """Viscous damping b (N·m·s/rad)."""

    stick_vel: float = 0.02
    """|vel| below this is treated as the static (stick) regime (rad/s)."""
