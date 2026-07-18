"""Thin vec-env adapter over Isaac Lab's ManagerBasedRLEnv for the TD-MPC2 trainer.

Unlike RslRlVecEnvWrapper (which merges terminated|truncated into one `dones` tensor), this
adapter surfaces the RAW 5-tuple from `ManagerBasedRLEnv.step` — keeping `terminated` and
`time_out` SEPARATE (needed for correct terminal-value bootstrapping) — and exposes BOTH obs
groups: the deployable 45-dim noisy `policy` obs and the clean 48-dim privileged `critic` obs
(critic = policy + true base_lin_vel).

IMPORTANT (verified in isaaclab source): the env auto-resets done envs INSIDE `step`, and the
returned obs for a done env is the RESET obs — the true terminal obs is overwritten and is NOT
in `extras`. This adapter does not try to reconstruct it; it just returns the `terminated` /
`time_out` flags so the replay buffer can avoid sampling windows that cross an episode boundary.
"""

from __future__ import annotations

import torch


class TdmpcVecEnv:
    """Minimal raw-tuple adapter. `env` is the gym.make result; we drive `env.unwrapped`."""

    def __init__(self, env):
        self.env = env
        self.uenv = env.unwrapped
        self.num_envs: int = int(self.uenv.num_envs)
        self.device = self.uenv.device
        self.num_obs: int = 45
        self.num_priv_obs: int = 48
        self.num_actions: int = int(self.uenv.action_manager.total_action_dim)
        self.step_dt: float = float(self.uenv.step_dt)

    def _split_obs(self, obs_dict) -> tuple[torch.Tensor, torch.Tensor]:
        return obs_dict["policy"], obs_dict["critic"]

    def reset(self) -> tuple[torch.Tensor, torch.Tensor]:
        obs_dict, _ = self.uenv.reset()
        return self._split_obs(obs_dict)

    def step(self, action: torch.Tensor):
        """Return (obs_policy(N,45), obs_critic(N,48), reward(N,), terminated(N,) bool,
        time_out(N,) bool, extras). `obs_*` are post-reset for done envs (see module note)."""
        obs_dict, reward, terminated, time_out, extras = self.uenv.step(action)
        obs_p, obs_c = self._split_obs(obs_dict)
        return obs_p, obs_c, reward, terminated.bool(), time_out.bool(), extras

    def close(self):
        self.env.close()
