"""Vectorized GPU replay buffer for TD-MPC2 over Isaac Lab's fixed-N parallel envs.

Replaces upstream TD-MPC2's torchrl per-episode buffer. Stores N time-contiguous per-env rings
and samples length-(horizon+1) obs windows (horizon actions/rewards) that NEVER cross an episode
boundary — the critical correctness requirement, because Isaac auto-resets done envs in-step and
the true terminal obs is lost (verified in isaaclab source). Pure torch tensors, no torchrl, so
it works against torch 2.11 / tensordict 0.13 without the upstream pins.

Boundary rule: a transition stored at time index t carries `terminated[t]`, `time_out[t]`, and
`done[t] = terminated | time_out`. A sampled window starts at s, uses transitions s..s+H-1 and obs
s..s+H (H=horizon). It is valid iff:
  * no `done` on the INTERIOR transitions s..s+H-2 (an interior reset would break obs continuity /
    the imagined-rollout targets); AND
  * the BOUNDARY transition s+H-1 is not a `time_out` (a truncation's next-obs s+H is a reset obs,
    but `terminated=0` there so the target would wrongly bootstrap V of a reset state).
A `terminated` boundary transition at s+H-1 IS allowed: `(1-terminated)=0` zeros the bootstrap, so
obs s+H (a reset obs) is never used — and this is the ONLY way a `terminated=1` transition (and the
"terminal has no future" signal) ever enters the loss. Excluding it (the earlier rule) made the
`(1-terminated)` mask dead code and the value fn over-value near-collapse states.

Terminal-value bootstrap: `sample` returns `terminated` per step so the agent masks the value
target `r + gamma*(1-terminated)*V(next)`.
"""

from __future__ import annotations

import math

import torch


class SequenceReplayBuffer:
    def __init__(self, cfg, num_envs: int, obs_dim: int, priv_obs_dim: int, act_dim: int, device):
        self.device = device
        self.N = int(num_envs)
        self.H = int(cfg.horizon)
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.use_priv = bool(cfg.use_privileged_critic)
        self.priv_dim = int(priv_obs_dim) if self.use_priv else 0
        # per-env ring capacity so total ~= buffer_size
        self.cap = max(self.H + 2, math.ceil(int(cfg.buffer_size) / self.N))

        f = lambda *s: torch.empty(*s, device=device)  # noqa: E731
        self.obs = f(self.cap, self.N, self.obs_dim)
        self.priv = f(self.cap, self.N, self.priv_dim) if self.use_priv else None
        self.action = f(self.cap, self.N, self.act_dim)
        self.reward = f(self.cap, self.N)
        self.terminated = torch.zeros(self.cap, self.N, dtype=torch.bool, device=device)
        self.time_out = torch.zeros(self.cap, self.N, dtype=torch.bool, device=device)
        self.done = torch.zeros(self.cap, self.N, dtype=torch.bool, device=device)
        # TD-M(PC)²: planner mean/std stored from day 1 (unused until the toggle is on)
        self.plan_mean = f(self.cap, self.N, self.act_dim)
        self.plan_std = f(self.cap, self.N, self.act_dim)

        self.head = 0          # next write row (shared across envs — they advance in lockstep)
        self.size = 0          # number of valid rows (<= cap)
        self._env_ids = torch.arange(self.N, device=device)

    def __len__(self):
        return self.size * self.N

    @torch.no_grad()
    def add(self, obs, action, reward, terminated, time_out, priv=None, plan_mean=None, plan_std=None):
        """Store one env-step transition batch (all shape (N, ...) / (N,))."""
        t = self.head
        self.obs[t] = obs
        if self.use_priv:
            self.priv[t] = priv
        self.action[t] = action
        self.reward[t] = reward
        self.terminated[t] = terminated
        self.time_out[t] = time_out
        self.done[t] = terminated | time_out
        if plan_mean is not None:
            self.plan_mean[t] = plan_mean
            self.plan_std[t] = plan_std
        self.head = (self.head + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    @torch.no_grad()
    def _valid_starts(self):
        """Boolean mask of window-start slots whose [s, s+H] window is fully in-buffer, contiguous
        (no ring-wrap over head), has no INTERIOR done (positions s..s+H-2), and whose BOUNDARY
        transition s+H-1 is not a time_out. A `terminated` boundary IS allowed (bootstrap masked)."""
        if self.size < self.H + 1:
            return None
        # candidate start rows: those with H full transitions + 1 trailing obs ahead of them,
        # none wrapping across `head`. Work in "logical age" order to avoid the wrap seam.
        # oldest logical row index -> physical row:
        start_logical = (self.head - self.size) % self.cap
        # physical rows in chronological order
        order = (start_logical + torch.arange(self.size, device=self.device)) % self.cap  # (size,)
        done_chrono = self.done[order]          # (size, N) in chronological order
        timeout_chrono = self.time_out[order]   # (size, N)
        H = self.H
        n = self.size
        if n - H < 1:
            return None
        # cumulative done count for the INTERIOR test "any done in [p, p+H-2]"
        z = torch.zeros(1, self.N, device=self.device)
        csum = torch.cat([z, torch.cumsum(done_chrono.float(), dim=0)], dim=0)  # (n+1, N)
        # valid start positions p in [0, n-H-1]; window transitions p..p+H-1
        p = torch.arange(0, n - H, device=self.device)  # ensures p+H <= n-1 (obs at p+H exists)
        interior_dones = csum[p + H - 1] - csum[p]        # dones in [p, p+H-2] (interior only)
        boundary_timeout = timeout_chrono[p + H - 1]      # (len(p), N) time_out at the last transition
        valid_chrono = (interior_dones == 0) & (~boundary_timeout)  # (len(p), N)
        return order, p, valid_chrono  # physical order map + chronological start positions + mask

    @torch.no_grad()
    def sample(self, batch_size: int, horizon: int | None = None):
        """Return a dict of stacked sequences:
          obs:        (H+1, B, obs_dim)
          priv:       (H+1, B, priv_dim)  (only if use_privileged_critic)
          action:     (H,   B, act_dim)
          reward:     (H,   B)
          terminated: (H,   B)
          plan_mean/plan_std: (H, B, act_dim)
        Windows never cross an episode boundary. Returns None if not enough data yet.
        """
        H = self.H if horizon is None else int(horizon)
        assert H == self.H, "buffer built for a fixed horizon"
        vs = self._valid_starts()
        if vs is None:
            return None
        order, p_positions, valid_chrono = vs  # order:(size,), p_positions:(P,), valid:(P,N)
        # flatten valid (position, env) pairs and sample B of them
        flat_valid = valid_chrono.reshape(-1)  # (P*N,)
        valid_idx = flat_valid.nonzero(as_tuple=False).squeeze(-1)
        if valid_idx.numel() == 0:
            return None
        pick = valid_idx[torch.randint(valid_idx.numel(), (batch_size,), device=self.device)]
        P = p_positions.numel()
        pos = p_positions[pick // self.N]        # (B,) chronological start position
        env = pick % self.N                      # (B,) env index
        # chronological positions p..p+H -> physical rows via `order`
        steps = torch.arange(H + 1, device=self.device)  # (H+1,)
        chrono = pos.unsqueeze(0) + steps.unsqueeze(1)    # (H+1, B) chronological indices
        rows = order[chrono]                              # (H+1, B) physical rows
        b = torch.arange(batch_size, device=self.device)
        # gather: rows (H+1,B), env (B,) -> broadcast env over the H+1 axis
        env_b = env.unsqueeze(0).expand(H + 1, batch_size)  # (H+1,B)
        out = {
            "obs": self.obs[rows, env_b],                       # (H+1,B,obs)
            "action": self.action[rows[:H], env_b[:H]],         # (H,B,act)
            "reward": self.reward[rows[:H], env_b[:H]],         # (H,B)
            "terminated": self.terminated[rows[:H], env_b[:H]].float(),  # (H,B)
            "plan_mean": self.plan_mean[rows[:H], env_b[:H]],
            "plan_std": self.plan_std[rows[:H], env_b[:H]],
        }
        if self.use_priv:
            out["priv"] = self.priv[rows, env_b]                # (H+1,B,priv)
        return out
