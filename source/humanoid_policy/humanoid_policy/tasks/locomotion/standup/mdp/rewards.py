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
    reference: dict | None = None,
    std: float = 0.4,
    min_delta: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Dense reward for matching a target joint pose (``{joint_name: radians}``), in [0, 1].

    Measures per-joint closeness normalized by each joint's ``|reference - target|`` travel (the
    squat->stand motion, clamped to ``min_delta``), then averages the per-joint exponentials:
    ``mean_j exp(-((q_j - target_j)/denom_j)**2 / std**2)``.

    Why normalized + per-joint-averaged rather than a single mean-squared error: the big sagittal
    joints (knee/hip_pitch, ~1.5 rad of travel) otherwise dominate the error and drown out the small
    lateral 'feet-in' joints (hip_roll/ankle_roll, ~0.2 rad), so the robot could score ~full reward
    with its feet still wide. Normalizing puts every joint on the same 0..1 "fraction of the way to
    the stand pose" scale, so the feet count as much as the knees. Averaging per-joint also makes the
    reward climb only once most joints match, so the feet are drawn in as it reaches standing.

    ``reference`` is the start (squat) pose; if omitted, falls back to unnormalized absolute error.
    The joint index map + denominators are resolved by name once and cached (sim joint order is
    interleaved). Empty ``target`` returns 1s so a missing pose library degrades gracefully.
    """
    asset = env.scene[asset_cfg.name]
    device = env.device  # torch device (asset.data.*.device is warp's Device, not torch's)
    ref = reference or {}
    key = (id(env), asset_cfg.name, tuple(sorted(target.items())), tuple(sorted(ref.items())), min_delta)
    cache = _pose_target_cache.get(key)
    if cache is None:
        names = list(asset.data.joint_names)
        order = [i for i in range(len(names)) if names[i] in target]
        idx = torch.tensor(order, device=device, dtype=torch.long)
        vals = torch.tensor([target[names[i]] for i in order], device=device, dtype=torch.float32)
        if ref:
            denom = torch.tensor(
                [max(abs(ref.get(names[i], target[names[i]]) - target[names[i]]), min_delta) for i in order],
                device=device, dtype=torch.float32,
            )
        else:
            denom = torch.ones(len(order), device=device, dtype=torch.float32)
        _pose_target_cache[key] = cache = (idx, vals, denom)
    idx, vals, denom = cache
    if idx.numel() == 0:
        return torch.ones(env.num_envs, device=device)
    q = asset.data.joint_pos[:]  # materialize ProxyArray -> torch (num_envs, n_joints)
    e = (q[:, idx] - vals) / denom
    return torch.mean(torch.exp(-(e * e) / (std * std)), dim=1)


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
