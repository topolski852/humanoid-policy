"""TD-MPC2 implicit world model — adapted from the official MIT repo, simplified for our setting.

Single-task, state-only, non-episodic, non-multitask. Encoder/dynamics/reward/π/Q operate on the
shared latent z = encode(obs). Q-ensemble is a plain ModuleList (no tensordict/vmap); target Qs are
a deepcopy soft-updated via Polyak. `cfg` must carry the derived fields the agent attaches:
`obs_dim`, `action_dim`, `bin_size`.
"""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn

from .common import init
from .common import layers
from .common import math as tdmath


class WorldModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d_za = cfg.latent_dim + cfg.action_dim
        self._encoder = layers.state_encoder(cfg)
        self._dynamics = layers.mlp(d_za, 2 * [cfg.mlp_dim], cfg.latent_dim, act=layers.SimNorm(cfg))
        self._reward = layers.mlp(d_za, 2 * [cfg.mlp_dim], max(cfg.num_bins, 1))
        self._pi = layers.mlp(cfg.latent_dim, 2 * [cfg.mlp_dim], 2 * cfg.action_dim)
        self._Qs = nn.ModuleList(
            [layers.mlp(d_za, 2 * [cfg.mlp_dim], max(cfg.num_bins, 1), dropout=cfg.dropout) for _ in range(cfg.num_q)]
        )
        self.apply(init.weight_init)
        init.zero_([self._reward[-1].weight])
        for q in self._Qs:
            init.zero_([q[-1].weight])

        self.register_buffer("log_std_min", torch.tensor(float(cfg.log_std_min)))
        self.register_buffer("log_std_dif", torch.tensor(float(cfg.log_std_max) - float(cfg.log_std_min)))

        self._target_Qs = deepcopy(self._Qs)
        for p in self._target_Qs.parameters():
            p.requires_grad_(False)

    # -- keep target Qs in eval mode --------------------------------------------
    def train(self, mode: bool = True):
        super().train(mode)
        self._target_Qs.train(False)
        return self

    @torch.no_grad()
    def soft_update_target_Q(self):
        for p, tp in zip(self._Qs.parameters(), self._target_Qs.parameters()):
            tp.data.lerp_(p.data, self.cfg.tau)

    @property
    def total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # -- model heads ------------------------------------------------------------
    def encode(self, obs):
        return self._encoder(obs)

    def next(self, z, a):
        return self._dynamics(torch.cat([z, a], dim=-1))

    def reward(self, z, a):
        return self._reward(torch.cat([z, a], dim=-1))

    def pi(self, z):
        """Gaussian policy prior (reparameterized, tanh-squashed to [-1,1])."""
        mean, log_std = self._pi(z).chunk(2, dim=-1)
        log_std = tdmath.log_std(log_std, self.log_std_min, self.log_std_dif)
        eps = torch.randn_like(mean)
        log_prob = tdmath.gaussian_logprob(eps, log_std)
        size = eps.shape[-1]
        scaled_log_prob = log_prob * size
        action = mean + eps * log_std.exp()
        mean, action, log_prob = tdmath.squash(mean, action, log_prob)
        entropy_scale = scaled_log_prob / (log_prob + 1e-8)
        info = {
            "mean": mean,
            "log_std": log_std,
            "entropy": -log_prob,
            "scaled_entropy": -log_prob * entropy_scale,
        }
        return action, info

    def Q(self, z, a, return_type: str = "min", target: bool = False):
        assert return_type in {"min", "avg", "all"}
        za = torch.cat([z, a], dim=-1)
        qs = self._target_Qs if target else self._Qs
        out = torch.stack([q(za) for q in qs], dim=0)  # (num_q, ..., num_bins)
        if return_type == "all":
            return out
        qidx = torch.randperm(self.cfg.num_q, device=out.device)[:2]
        Q = tdmath.two_hot_inv(out[qidx], self.cfg)  # (2, ..., 1)
        return Q.min(0).values if return_type == "min" else Q.sum(0) / 2
