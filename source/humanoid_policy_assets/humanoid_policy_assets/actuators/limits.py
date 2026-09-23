"""Per-joint effort limits, read back off a live articulation.

Why this exists: on an **explicit** actuator (which the modeled walk plant uses) the real
torque clamp lives on the actuator group as ``effort_limit``, NOT on ``asset.data``.
``asset.data.joint_effort_limits`` reports ``effort_limit_sim``, which
``robots/humanoid.py`` deliberately leaves at the explicit-actuator default (1e9) so PhysX
does not double-clip on top of the actuator's own clamp. Reading it would silently give you
"no limit" instead of the 12.0 / 7.0 N·m contract caps.

Anything that needs to reason about torque headroom -- the saturation reward, the sim-side
saturation metric in ``scripts/rsl_rl/eval_plant_compare.py`` -- should go through here so
there is exactly one place that knows that.
"""

from __future__ import annotations

import torch


def joint_effort_limits(asset) -> torch.Tensor:
    """``(num_envs, num_joints)`` tensor of each joint's actuator effort limit.

    Assembled by scattering every actuator group's ``effort_limit`` into a full-width buffer
    at that group's ``joint_indices``. Joints not covered by any group get ``inf`` (never
    saturating) rather than 0, so an unmapped joint reads as "no limit" instead of
    "permanently saturated" -- the latter would silently dominate any penalty built on this.

    The result is cached on the asset, since effort limits are fixed after startup.
    """
    cached = getattr(asset, "_humanoid_effort_limit_buf", None)
    if cached is not None:
        return cached

    num_envs = asset.data.joint_pos.shape[0]
    num_joints = asset.data.joint_pos.shape[1]
    buf = torch.full((num_envs, num_joints), float("inf"), device=asset.device)

    for group in asset.actuators.values():
        limit = group.effort_limit
        ids = group.joint_indices
        if isinstance(ids, slice):  # a group covering every joint
            ids = torch.arange(num_joints, device=asset.device)
        buf[:, ids] = limit.to(buf.dtype)

    asset._humanoid_effort_limit_buf = buf
    return buf
