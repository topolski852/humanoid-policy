"""Standalone correctness test for SequenceReplayBuffer — NO Isaac/sim needed.

Feeds a synthetic vectorized stream with scripted per-env dones. Each env's obs[...,0] is an
EPISODE-LOCAL step counter that increments by 1 each step and RESETS to 0 at a done. Therefore:
  - within any window that does NOT cross a boundary, consecutive obs differ by exactly +1;
  - a window that crossed a boundary would show a reset (a diff != +1).
So asserting "all intra-window diffs == 1" simultaneously verifies (a) no interior done leaks
into a sampled window and (b) obs continuity. Also exercises ring-wrap (more steps than capacity)
and prints VRAM.

Run: .venv/bin/python scripts/tdmpc/check_buffer.py
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "source", "humanoid_policy"))
from humanoid_policy.tdmpc.buffer import SequenceReplayBuffer  # noqa: E402


def run(device, use_priv):
    N, H = 8, 3
    obs_dim, priv_dim, act_dim = 4, 6, 2
    cfg = SimpleNamespace(horizon=H, buffer_size=N * 50, use_privileged_critic=use_priv)
    buf = SequenceReplayBuffer(cfg, N, obs_dim, priv_dim, act_dim, device)
    assert buf.cap == 50, buf.cap  # ceil(400/8)

    torch.manual_seed(0)
    counter = torch.zeros(N, device=device)           # episode-local step per env
    steps = 130                                        # > cap (50) to force ring-wrap
    # scripted episode lengths per env so dones land at different, known places
    ep_len = torch.tensor([5, 7, 9, 11, 13, 4, 6, 8], device=device, dtype=torch.float)
    since = torch.zeros(N, device=device)

    for _ in range(steps):
        obs = torch.zeros(N, obs_dim, device=device)
        obs[:, 0] = counter
        priv = torch.zeros(N, priv_dim, device=device) if use_priv else None
        if use_priv:
            priv[:, 0] = counter
        action = torch.randn(N, act_dim, device=device)
        reward = torch.randn(N, device=device)
        since += 1
        done = since >= ep_len                          # scripted terminations
        terminated = done & (torch.rand(N, device=device) < 0.5)  # split term vs timeout
        time_out = done & ~terminated
        buf.add(obs, action, reward, terminated, time_out, priv=priv,
                plan_mean=action, plan_std=torch.ones_like(action))
        # advance synthetic env: reset counter at done, else +1
        counter = torch.where(done, torch.zeros_like(counter), counter + 1)
        since = torch.where(done, torch.zeros_like(since), since)

    # sample many windows and validate
    total = 0
    for _ in range(200):
        batch = buf.sample(batch_size=256)
        assert batch is not None
        obs = batch["obs"]                              # (H+1, B, obs_dim)
        c = obs[..., 0]                                 # (H+1, B) episode-local counters
        diffs = c[1:] - c[:-1]                          # (H, B)
        bad = (diffs != 1.0)
        assert not bad.any(), f"cross-boundary/discontinuity: {int(bad.sum())} of {diffs.numel()} steps"
        # shape checks
        assert batch["action"].shape == (H, 256, act_dim)
        assert batch["reward"].shape == (H, 256)
        assert batch["terminated"].shape == (H, 256)
        if use_priv:
            assert batch["priv"].shape == (H + 1, 256, priv_dim)
            assert torch.equal(batch["priv"][..., 0], c)  # priv counter must match obs counter
        total += diffs.numel()

    vram = torch.cuda.memory_allocated(device) / 1e6 if str(device).startswith("cuda") else 0.0
    print(f"  [{device}, priv={use_priv}] OK: {total} intra-window steps validated, "
          f"len(buf)={len(buf)}, cap/env={buf.cap}, VRAM={vram:.1f} MB")


def main():
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for use_priv in (True, False):
        run(dev, use_priv)
    print("check_buffer: ALL PASS")


if __name__ == "__main__":
    main()
