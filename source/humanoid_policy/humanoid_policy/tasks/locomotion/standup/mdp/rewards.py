from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_off_ground(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize feet leaving the ground (returns the count of airborne feet, 0..n_feet).

    In a squat -> stand both feet should stay planted the whole time, so this discourages the
    policy from lifting/kicking a leg to generate momentum. Complements ``feet_slide`` (which only
    acts while a foot is loaded): a foot lifted clear of the ground escapes ``feet_slide`` but is
    caught here.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history.torch[:, :, sensor_cfg.body_ids, :]
    in_contact = forces.norm(dim=-1).max(dim=1)[0] > threshold  # [envs, n_feet]
    return torch.sum((~in_contact).float(), dim=1)


_pose_target_cache: dict = {}


def track_joint_pose_exp(
    env: "ManagerBasedRLEnv",
    target: dict,
    std: float = 0.7,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Dense reward for matching a target joint pose (``{joint_name: radians}``).

    Returns ``exp(-mean_sq_err / std**2)`` in [0, 1], peaking at 1 when the joints match ``target``.
    Rewards reaching the *standing* pose (loaded from the pose library) rather than just gaining
    height. The joint index map is resolved by name once and cached (sim joint order is interleaved,
    so never assume the config order). Empty ``target`` returns 1s (no-op) so a missing pose library
    degrades gracefully instead of crashing the env.
    """
    asset = env.scene[asset_cfg.name]
    device = env.device  # torch device (asset.data.*.device is warp's Device, not torch's)
    key = (id(env), asset_cfg.name, tuple(sorted(target.items())))
    cache = _pose_target_cache.get(key)
    if cache is None:
        names = list(asset.data.joint_names)
        order = [i for i in range(len(names)) if names[i] in target]
        idx = torch.tensor(order, device=device, dtype=torch.long)
        vals = torch.tensor([target[names[i]] for i in order], device=device, dtype=torch.float32)
        _pose_target_cache[key] = cache = (idx, vals)
    idx, vals = cache
    if idx.numel() == 0:
        return torch.ones(env.num_envs, device=device)
    q = asset.data.joint_pos[:]  # materialize ProxyArray -> torch (num_envs, n_joints)
    err = torch.mean(torch.square(q[:, idx] - vals), dim=1)
    return torch.exp(-err / (std * std))


def base_height_exp(
    env: "ManagerBasedRLEnv",
    target_height: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Exponential reward peaking when the base reaches ``target_height`` (flat-ground world z).

    Complements a coarse ``base_height_l2`` penalty: the L2 term supplies a gradient from the
    deep squat all the way up, while this term adds a sharp bonus for settling at standing height.
    """
    asset = env.scene[asset_cfg.name]
    base_height = asset.data.root_pos_w[:, 2]
    return torch.exp(-torch.square(base_height - target_height) / (std**2))
