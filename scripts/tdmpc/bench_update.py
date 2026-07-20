"""Benchmark TD-MPC2 update throughput: eager vs torch.compile — NO Isaac/sim needed.

Times N `agent.update(batch)` calls with a fixed dummy batch (real shapes: horizon+1, batch 256)
under three configs and prints updates/sec + the implied env-steps/sec at updates_per_step=16:
  1. eager (baseline)
  2. compile the 5 world-model methods (encode/next/reward/pi/Q) individually
  3. compile _update end-to-end (mode='reduce-overhead', cudagraphs)
Also times plan_batch (MPPI collection) eager vs the method-compiled model.

Run: .venv/bin/python scripts/tdmpc/bench_update.py
"""

from __future__ import annotations

import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "source", "humanoid_policy"))
from humanoid_policy.tdmpc.config import TdmpcAgentCfg  # noqa: E402
from humanoid_policy.tdmpc.agent import TDMPC2  # noqa: E402

DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
OBS, ACT = 45, 12
UPS = 16  # updates_per_step for the sps projection


def make_batch(cfg):
    H, B = cfg.horizon, cfg.batch_size
    return {
        "obs": torch.randn(H + 1, B, OBS, device=DEV),
        "action": (2 * torch.rand(H, B, ACT, device=DEV) - 1),
        "reward": torch.randn(H, B, device=DEV),
        "terminated": (torch.rand(H, B, device=DEV) < 0.05).float(),
        "plan_mean": (2 * torch.rand(H, B, ACT, device=DEV) - 1),
        "plan_std": torch.rand(H, B, ACT, device=DEV) + 0.05,
    }


def time_updates(agent, batch, n=300, warmup=30):
    for _ in range(warmup):
        agent.update(batch)
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(n):
        agent.update(batch)
    torch.cuda.synchronize()
    dt = time.time() - t
    ups = n / dt
    # env-steps/sec at 32 envs, UPS updates/iter: sps = 32 / (collect + UPS/ups_persec)
    return ups, dt / n * 1000.0  # updates/sec, ms/update


def build(cfg_mods=None):
    torch.manual_seed(0)
    cfg = TdmpcAgentCfg()
    cfg.batch_size = 256
    cfg.use_tdmpc2_square = True  # matches the live run (prior loss path active)
    if cfg_mods:
        cfg_mods(cfg)
    return TDMPC2(cfg, OBS, ACT, DEV)


def main():
    print(f"device={DEV} torch={torch.__version__}")
    cfg = TdmpcAgentCfg(); cfg.batch_size = 256

    # 1) eager
    agent = build()
    batch = make_batch(agent.cfg)
    ups0, ms0 = time_updates(agent, batch)
    print(f"[eager]            {ups0:6.1f} upd/s   {ms0:5.2f} ms/update")

    # 2) compile the 5 model methods individually
    try:
        agent2 = build()
        m = agent2.model
        for name in ("encode", "next", "reward", "pi", "Q"):
            setattr(m, "_c_" + name, torch.compile(getattr(m, name)))
        # monkeypatch: route calls through compiled versions
        m.encode = m._c_encode; m.next = m._c_next; m.reward = m._c_reward
        m.pi = m._c_pi; m.Q = m._c_Q
        ups1, ms1 = time_updates(agent2, make_batch(agent2.cfg))
        print(f"[compile-methods]  {ups1:6.1f} upd/s   {ms1:5.2f} ms/update   ({ups1/ups0:.2f}x)")
    except Exception as e:
        print(f"[compile-methods]  FAILED: {type(e).__name__}: {str(e)[:180]}")

    # 3) compile _update end-to-end, reduce-overhead (cudagraphs)
    try:
        agent3 = build()
        agent3._update = torch.compile(agent3._update, mode="reduce-overhead")
        ups2, ms2 = time_updates(agent3, make_batch(agent3.cfg))
        print(f"[compile-_update]  {ups2:6.1f} upd/s   {ms2:5.2f} ms/update   ({ups2/ups0:.2f}x)")
    except Exception as e:
        print(f"[compile-_update]  FAILED: {type(e).__name__}: {str(e)[:180]}")

    print(f"\nProjection @ 32 envs, updates_per_step={UPS} (updates dominate):")
    print(f"  eager ~{32/(UPS/ups0):.0f} sps from updates alone (collection extra)")


if __name__ == "__main__":
    main()
