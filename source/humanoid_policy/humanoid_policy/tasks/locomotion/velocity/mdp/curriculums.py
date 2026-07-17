"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w.torch[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())


def command_forward_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str = "base_velocity",
    fwd_min: float = 0.3,
    start_max: float = 0.4,
    end_max: float = 1.0,
    num_steps: int = 48000,
) -> float:
    """Ramp the forward-command MAX from ``start_max`` to ``end_max`` over training.

    The policy starts by only having to track a slow forward walk (a gentle exit from
    the "stand still" reward basin) and faces the full commanded speed once it has
    learned to move. Mutates the LIVE command ranges in place — the velocity command
    re-reads ``cfg.ranges.lin_vel_x`` on every resample, so later resamples pick up the
    wider range. The lower bound stays at ``fwd_min`` so the robot is always commanded
    forward (never to stand).

    Args:
        num_steps: env steps over which the max ramps ``start_max`` -> ``end_max``.

    Returns:
        The current forward-command max (logged as ``Curriculum/command_forward_levels``).
    """
    term = env.command_manager.get_term(command_name)
    frac = min(1.0, float(env.common_step_counter) / float(max(1, num_steps)))
    new_max = start_max + frac * (end_max - start_max)
    term.cfg.ranges.lin_vel_x = (fwd_min, new_max)
    return new_max
