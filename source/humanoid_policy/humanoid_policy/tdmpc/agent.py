"""TD-MPC2 agent — adapted from the official MIT repo for our vectorized Isaac Lab setting.

Key adaptations vs upstream tdmpc2/tdmpc2.py:
  - device-agnostic (no cuda:0 hardcode), eager only (no torch.compile / cudagraphs), single-task,
    non-episodic, non-multitask.
  - `act_pi(obs_batch)` — BATCHED policy-prior action over N envs, used for data collection (planning
    is not needed to collect and vectorizes trivially through the prior).
  - `_update(batch)` — takes the dict from our SequenceReplayBuffer.sample (not torchrl).
  - Actions are normalized to [-1,1]; the trainer maps to env-raw by ×cfg.act_env_scale.
  - `plan()` (MPPI) kept for eval/P2, operating on a single obs.
  - `update_pi` uses the live Qs then clears their grads (upstream used a detached Q copy).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .common import math as tdmath
from .common.scale import RunningScale
from .world_model import WorldModel


class TDMPC2(torch.nn.Module):
    def __init__(self, cfg, obs_dim: int, action_dim: int, device):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(device)
        # derived fields the model/math read off cfg
        cfg.obs_dim = int(obs_dim)
        cfg.action_dim = int(action_dim)
        cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins - 1)
        assert cfg.latent_dim % cfg.simnorm_dim == 0, "latent_dim must be divisible by simnorm_dim"

        self.model = WorldModel(cfg).to(self.device)
        self.optim = torch.optim.Adam([
            {"params": self.model._encoder.parameters(), "lr": cfg.lr * cfg.enc_lr_scale},
            {"params": self.model._dynamics.parameters()},
            {"params": self.model._reward.parameters()},
            {"params": self.model._Qs.parameters()},
        ], lr=cfg.lr)
        self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=cfg.lr, eps=1e-5)
        self.model.eval()
        self.scale = RunningScale(cfg, self.device)
        self.discount = float(cfg.discount)
        self.iterations = cfg.mppi_iterations + 2 * int(action_dim >= 20)
        self._prev_mean = torch.zeros(cfg.horizon, action_dim, device=self.device)

    # ---- inference ------------------------------------------------------------
    @torch.no_grad()
    def act_pi(self, obs, eval_mode=False):
        """Batched policy-prior action (N, action_dim) in [-1,1]. Used for collection."""
        z = self.model.encode(obs)
        action, info = self.model.pi(z)
        if eval_mode:
            action = info["mean"]
        return action

    @torch.no_grad()
    def _estimate_value(self, z, actions):
        G, discount = 0, 1
        for t in range(self.cfg.horizon):
            reward = tdmath.two_hot_inv(self.model.reward(z, actions[t]), self.cfg)
            z = self.model.next(z, actions[t])
            G = G + discount * reward
            discount = discount * self.discount
        action, _ = self.model.pi(z)
        return G + discount * self.model.Q(z, action, return_type="avg")

    @torch.no_grad()
    def plan(self, obs, t0=False, eval_mode=False):
        """MPPI over the latent model for a SINGLE obs (1, obs_dim). Returns (action_dim,) in [-1,1]."""
        H, A, dev = self.cfg.horizon, self.cfg.action_dim, self.device
        z0 = self.model.encode(obs)  # (1, latent)
        # policy-prior seed trajectories
        pi_actions = torch.empty(H, self.cfg.num_pi_trajs, A, device=dev)
        _z = z0.repeat(self.cfg.num_pi_trajs, 1)
        for t in range(H - 1):
            pi_actions[t], _ = self.model.pi(_z)
            _z = self.model.next(_z, pi_actions[t])
        pi_actions[-1], _ = self.model.pi(_z)

        z = z0.repeat(self.cfg.num_samples, 1)
        mean = torch.zeros(H, A, device=dev)
        std = torch.full((H, A), self.cfg.max_std, device=dev)
        if not t0:
            mean[:-1] = self._prev_mean[1:]
        actions = torch.empty(H, self.cfg.num_samples, A, device=dev)
        actions[:, : self.cfg.num_pi_trajs] = pi_actions

        for _ in range(self.iterations):
            r = torch.randn(H, self.cfg.num_samples - self.cfg.num_pi_trajs, A, device=dev)
            actions[:, self.cfg.num_pi_trajs:] = (mean.unsqueeze(1) + std.unsqueeze(1) * r).clamp(-1, 1)
            value = self._estimate_value(z, actions).nan_to_num(0)
            elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
            elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]
            max_value = elite_value.max(0).values
            score = torch.exp(self.cfg.temperature * (elite_value - max_value))
            score = score / score.sum(0)
            mean = (score.unsqueeze(0) * elite_actions).sum(dim=1) / (score.sum(0) + 1e-9)
            std = ((score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2).sum(dim=1)
                   / (score.sum(0) + 1e-9)).sqrt().clamp(self.cfg.min_std, self.cfg.max_std)

        rand_idx = tdmath.gumbel_softmax_sample(score.squeeze(1))
        acts = torch.index_select(elite_actions, 1, rand_idx).squeeze(1)
        a, astd = acts[0], std[0]
        if not eval_mode:
            a = a + astd * torch.randn(A, device=dev)
        self._prev_mean.copy_(mean)
        return a.clamp(-1, 1)

    @torch.no_grad()
    def plan_batch(self, obs, eval_mode=True):
        """Vectorized MPPI over N envs at once. obs (N, obs_dim) -> action (N, action_dim) in [-1,1].

        Same MPPI as `plan` but with an explicit env batch dim N (flattened into the sample batch for
        the model forward passes), so it runs the planner for all collection/eval envs in parallel.
        """
        cfg, dev = self.cfg, self.device
        N = obs.shape[0]
        H, A, S, Np, lat = cfg.horizon, cfg.action_dim, cfg.num_samples, cfg.num_pi_trajs, cfg.latent_dim
        z0 = self.model.encode(obs)  # (N, lat)

        pi_actions = torch.empty(H, N, Np, A, device=dev)
        _z = z0.unsqueeze(1).expand(N, Np, lat).reshape(N * Np, lat)
        for t in range(H - 1):
            a_t, _ = self.model.pi(_z)
            pi_actions[t] = a_t.view(N, Np, A)
            _z = self.model.next(_z, a_t)
        a_last, _ = self.model.pi(_z)
        pi_actions[-1] = a_last.view(N, Np, A)

        z = z0.unsqueeze(1).expand(N, S, lat).reshape(N * S, lat)  # (N*S, lat)
        mean = torch.zeros(H, N, A, device=dev)
        std = torch.full((H, N, A), cfg.max_std, device=dev)
        actions = torch.empty(H, N, S, A, device=dev)
        actions[:, :, :Np] = pi_actions

        for _ in range(self.iterations):
            r = torch.randn(H, N, S - Np, A, device=dev)
            actions[:, :, Np:] = (mean.unsqueeze(2) + std.unsqueeze(2) * r).clamp(-1, 1)
            value = self._estimate_value(z, actions.reshape(H, N * S, A)).squeeze(-1).view(N, S).nan_to_num(0)
            elite_idx = torch.topk(value, cfg.num_elites, dim=1).indices          # (N, E)
            ei = elite_idx.view(1, N, cfg.num_elites, 1).expand(H, N, cfg.num_elites, A)
            elite_actions = torch.gather(actions, 2, ei)                          # (H, N, E, A)
            elite_value = torch.gather(value, 1, elite_idx)                       # (N, E)
            score = torch.exp(cfg.temperature * (elite_value - elite_value.max(1, keepdim=True).values))
            score = score / (score.sum(1, keepdim=True) + 1e-9)                   # (N, E)
            sc = score.view(1, N, cfg.num_elites, 1)
            mean = (sc * elite_actions).sum(2) / (sc.sum(2) + 1e-9)               # (H, N, A)
            std = ((sc * (elite_actions - mean.unsqueeze(2)) ** 2).sum(2) / (sc.sum(2) + 1e-9)).sqrt()
            std = std.clamp(cfg.min_std, cfg.max_std)

        mu0, std0 = mean[0], std[0]           # planner action distribution at t=0 (for TD-M(PC)²)
        a0 = mu0
        if not eval_mode:
            a0 = a0 + std0 * torch.randn(N, A, device=dev)
        return a0.clamp(-1, 1), mu0, std0

    # ---- learning -------------------------------------------------------------
    def update_pi(self, zs, plan_mean=None, plan_std=None):
        cfg = self.cfg
        action, info = self.model.pi(zs)
        qs = self.model.Q(zs, action, return_type="avg")
        self.scale.update(qs[0])
        qs = self.scale(qs)
        rho = torch.pow(cfg.rho, torch.arange(len(qs), device=self.device))
        # base TD-MPC2 policy loss (entropy_coef*log_pi - Q, weighted by rho over the horizon)
        q_loss = (-(cfg.entropy_coef * info["scaled_entropy"] + qs).mean(dim=(1, 2)) * rho).mean()
        pi_loss = q_loss
        prior_loss = torch.zeros((), device=self.device)
        if cfg.use_tdmpc2_square and plan_mean is not None:
            # TD-M(PC)² "residual": pull the policy sample toward the planner's Gaussian (mu,std).
            H = plan_mean.shape[0]
            pis = action[:H]                                   # (H,B,A) policy-sampled actions
            std = plan_std.clamp_min(cfg.min_std)              # (H,B,A)
            eps = (pis - plan_mean) / std
            logp = (-0.5 * eps.pow(2) - std.log() - 0.9189385175704956).mean(dim=-1)  # (H,B) per-dim mean
            if float(self.scale.value) > cfg.scale_threshold:
                logp = logp / self.scale.value
            prior_loss = -(logp.mean(dim=1) * rho[:H]).mean()  # maximize log-lik -> minimize -logp
            pi_loss = q_loss + (cfg.prior_coef * cfg.action_dim / cfg.prior_dof_ref) * prior_loss
        pi_loss.backward()
        pi_grad_norm = torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), cfg.grad_clip_norm)
        self.pi_optim.step()
        self.pi_optim.zero_grad(set_to_none=True)
        return {"pi_loss": pi_loss.detach(), "prior_loss": prior_loss.detach(), "pi_grad_norm": pi_grad_norm,
                "pi_entropy": info["entropy"].detach().mean(), "pi_scale": self.scale.value.mean()}

    @torch.no_grad()
    def _td_target(self, next_z, reward, terminated):
        action, _ = self.model.pi(next_z)
        return reward + self.discount * (1 - terminated) * self.model.Q(next_z, action, return_type="min", target=True)

    def _update(self, obs, action, reward, terminated, plan_mean=None, plan_std=None):
        """obs (H+1,B,obs_dim); action (H,B,A); reward,terminated (H,B,1)."""
        cfg = self.cfg
        with torch.no_grad():
            next_z = self.model.encode(obs[1:])
            td_targets = self._td_target(next_z, reward, terminated)

        self.model.train()
        zs = torch.empty(cfg.horizon + 1, cfg.batch_size, cfg.latent_dim, device=self.device)
        z = self.model.encode(obs[0])
        zs[0] = z
        consistency_loss = 0
        for t, (_action, _next_z) in enumerate(zip(action.unbind(0), next_z.unbind(0))):
            z = self.model.next(z, _action)
            consistency_loss = consistency_loss + F.mse_loss(z, _next_z) * cfg.rho ** t
            zs[t + 1] = z

        _zs = zs[:-1]
        qs = self.model.Q(_zs, action, return_type="all")        # (num_q,H,B,num_bins)
        reward_preds = self.model.reward(_zs, action)            # (H,B,num_bins)

        reward_loss, value_loss = 0, 0
        for t, (rp, rew, tdt, qs_t) in enumerate(
            zip(reward_preds.unbind(0), reward.unbind(0), td_targets.unbind(0), qs.unbind(1))
        ):
            reward_loss = reward_loss + tdmath.soft_ce(rp, rew, cfg).mean() * cfg.rho ** t
            for q1 in qs_t.unbind(0):
                value_loss = value_loss + tdmath.soft_ce(q1, tdt, cfg).mean() * cfg.rho ** t

        consistency_loss = consistency_loss / cfg.horizon
        reward_loss = reward_loss / cfg.horizon
        value_loss = value_loss / (cfg.horizon * cfg.num_q)
        total_loss = (cfg.consistency_coef * consistency_loss
                      + cfg.reward_coef * reward_loss
                      + cfg.value_coef * value_loss)

        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip_norm)
        self.optim.step()
        self.optim.zero_grad(set_to_none=True)

        pi_info = self.update_pi(zs.detach(), plan_mean=plan_mean, plan_std=plan_std)
        # clear stray grads on Q/model params accumulated by the pi-loss (upstream used a detached Q copy)
        self.optim.zero_grad(set_to_none=True)
        self.model.soft_update_target_Q()

        self.model.eval()
        info = {"consistency_loss": consistency_loss.detach(), "reward_loss": reward_loss.detach(),
                "value_loss": value_loss.detach(), "total_loss": total_loss.detach(), "grad_norm": grad_norm}
        info.update(pi_info)
        return {k: (v.detach().mean() if isinstance(v, torch.Tensor) else torch.tensor(float(v))) for k, v in info.items()}

    def update(self, batch):
        """batch: dict from SequenceReplayBuffer.sample. Returns loss info dict (scalars)."""
        obs = batch["obs"]
        action = batch["action"]
        reward = batch["reward"].unsqueeze(-1)         # (H,B,1)
        terminated = batch["terminated"].unsqueeze(-1)  # (H,B,1)
        pm = batch.get("plan_mean") if self.cfg.use_tdmpc2_square else None
        ps = batch.get("plan_std") if self.cfg.use_tdmpc2_square else None
        return self._update(obs, action, reward, terminated, plan_mean=pm, plan_std=ps)

    # ---- checkpoint -----------------------------------------------------------
    def save(self, fp):
        torch.save({"model": self.model.state_dict(), "scale": self.scale.state_dict()}, fp)

    def load(self, fp):
        sd = torch.load(fp, map_location=self.device, weights_only=False) if isinstance(fp, str) else fp
        self.model.load_state_dict(sd["model"])
        if "scale" in sd:
            self.scale.load_state_dict(sd["scale"])
