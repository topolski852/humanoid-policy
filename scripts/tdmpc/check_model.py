"""Smoke test for the adapted TD-MPC2 world model + agent — NO Isaac/sim needed.

Builds the agent, runs several `update` steps on dummy horizon batches, and exercises the
batched policy prior and the MPPI planner. Asserts finite losses and correct shapes on the
actual torch 2.11 / no-torchrl stack.

Run: .venv/bin/python scripts/tdmpc/check_model.py
"""

from __future__ import annotations

import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "source", "humanoid_policy"))
from humanoid_policy.tdmpc.config import TdmpcAgentCfg  # noqa: E402
from humanoid_policy.tdmpc.agent import TDMPC2  # noqa: E402


def main():
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    cfg = TdmpcAgentCfg()
    cfg.batch_size = 128          # smaller for a fast check
    obs_dim, act_dim = 45, 12
    agent = TDMPC2(cfg, obs_dim, act_dim, dev)
    print(f"  world-model params: {agent.model.total_params:,} | latent={cfg.latent_dim} horizon={cfg.horizon}")

    H, B = cfg.horizon, cfg.batch_size
    for step in range(5):
        batch = {
            "obs": torch.randn(H + 1, B, obs_dim, device=dev),
            "action": (2 * torch.rand(H, B, act_dim, device=dev) - 1),
            "reward": torch.randn(H, B, device=dev),
            "terminated": (torch.rand(H, B, device=dev) < 0.05).float(),
        }
        info = agent.update(batch)
        for k, v in info.items():
            assert math.isfinite(float(v)), f"non-finite {k}={v} at step {step}"
    print("  update(): 5 steps, all losses finite | " +
          " ".join(f"{k}={float(v):.3f}" for k, v in info.items() if "loss" in k))

    # batched policy prior over N envs
    obs_n = torch.randn(32, obs_dim, device=dev)
    a = agent.act_pi(obs_n, eval_mode=False)
    assert a.shape == (32, act_dim) and float(a.abs().max()) <= 1.0 + 1e-5, (a.shape, float(a.abs().max()))
    print(f"  act_pi: shape={tuple(a.shape)} range=[{float(a.min()):.2f},{float(a.max()):.2f}] (expect [-1,1])")

    # MPPI planner on a single obs
    a1 = agent.plan(torch.randn(1, obs_dim, device=dev), t0=True, eval_mode=True)
    assert a1.shape == (act_dim,) and float(a1.abs().max()) <= 1.0 + 1e-5, (a1.shape, float(a1.abs().max()))
    print(f"  plan (MPPI): shape={tuple(a1.shape)} range=[{float(a1.min()):.2f},{float(a1.max()):.2f}]")
    print("check_model: ALL PASS")


if __name__ == "__main__":
    main()
