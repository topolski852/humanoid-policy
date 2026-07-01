from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


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
