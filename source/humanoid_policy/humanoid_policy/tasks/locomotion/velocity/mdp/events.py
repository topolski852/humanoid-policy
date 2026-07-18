from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

from isaaclab.assets import Articulation
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the joint default positions which may be different from URDF due to calibration errors.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if pos_distribution_params is not None:
        pos = asset.data.default_joint_pos.torch.clone()
        pos = _randomize_prop_by_op(
            pos, pos_distribution_params, env_ids, joint_ids, operation=operation, distribution=distribution
        )[env_ids][:, joint_ids]

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        asset.data.default_joint_pos.torch[env_ids, joint_ids] = pos


def randomize_stickslip_friction(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    friction_distribution_params: tuple[float, float] = (0.7, 1.3),
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Scale the stick-slip friction levels of the custom actuator model per env/joint.

    Covers joint-to-joint 3D-printed spread: we have ONE bench unit of each motor type, so the
    fitted coulomb/breakaway/viscous are nominals -- randomize +-30% (default) around them. The
    three levels scale together by a SINGLE factor per (env, joint) since they share one physical
    gearbox. Startup-only: these are plain tensors on the actuator (not PhysX solver params), so
    we scale them in place.

    Only actuators exposing the stick-slip attrs (``coulomb``) are touched; implicit-baseline runs
    are a no-op.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    for actuator in asset.actuators.values():
        if not hasattr(actuator, "coulomb"):
            continue
        # one shared scale per (env, joint) across the actuator's joints
        scale = _randomize_prop_by_op(
            torch.ones_like(actuator.coulomb),
            friction_distribution_params,
            env_ids,
            slice(None),
            operation="scale",
            distribution=distribution,
        )
        for attr in ("coulomb", "breakaway", "viscous"):
            getattr(actuator, attr)[env_ids] *= scale[env_ids]


def randomize_actuator_torque_constant(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    torque_constant_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the friction parameters used in joint friction model.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    # sample joint properties from the given ranges and set into the physics simulation
    # -- friction
    if torque_constant_params is not None:
        for actuator in asset.actuators.values():
            actuator_joint_ids = [joint_id in joint_ids for joint_id in actuator.joint_indices]
            if sum(actuator_joint_ids) > 0:
                stiffness = actuator.stiffness.to(asset.device).clone()
                damping = actuator.damping.to(asset.device).clone()
                scale = _randomize_prop_by_op(
                    torch.ones_like(stiffness, device=asset.device),
                    torque_constant_params,
                    env_ids,
                    actuator_joint_ids,
                    operation=operation,
                    distribution=distribution,
                )
                stiffness[env_ids[:, None], actuator_joint_ids] *= scale[env_ids[:, None], actuator_joint_ids]
                damping[env_ids[:, None], actuator_joint_ids] *= scale[env_ids[:, None], actuator_joint_ids]

                asset.write_joint_stiffness_to_sim(stiffness, joint_ids=actuator.joint_indices, env_ids=env_ids)
                asset.write_joint_damping_to_sim(damping, joint_ids=actuator.joint_indices, env_ids=env_ids)
