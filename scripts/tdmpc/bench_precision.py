"""Benchmark TF32 / bf16-autocast (x eager/compile) for the TD-MPC2 update — NO Isaac/sim.

Answers: on top of torch.compile, do TF32 and bf16 autocast add throughput, and does bf16 keep
the losses sane (finite + close to fp32)? Prints upd/s and the final losses per config.

Run: .venv/bin/python scripts/tdmpc/bench_precision.py
"""

from __future__ import annotations

import os
import sys
import time
from contextlib import nullcontext

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "source", "humanoid_policy"))
from humanoid_policy.tdmpc.config import TdmpcAgentCfg  # noqa: E402
from humanoid_policy.tdmpc.agent import TDMPC2  # noqa: E402

DEV = torch.device("cuda:0")
OBS, ACT = 45, 12


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


def build(compile_update):
    torch.manual_seed(0)
    cfg = TdmpcAgentCfg()
    cfg.batch_size = 256
    cfg.use_tdmpc2_square = True
    a = TDMPC2(cfg, OBS, ACT, DEV)
    if compile_update:
        a._update = torch.compile(a._update, mode="reduce-overhead")
    return a


def run(label, compile_update, tf32, bf16, n=300, warmup=30):
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    agent = build(compile_update)
    batch = make_batch(agent.cfg)
    ctx = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if bf16 else nullcontext
    info = None
    for _ in range(warmup):
        with ctx():
            info = agent.update(batch)
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(n):
        with ctx():
            info = agent.update(batch)
    torch.cuda.synchronize()
    ups = n / (time.time() - t)
    losses = {k: float(v) for k, v in info.items() if "loss" in k}
    finite = all(torch.isfinite(torch.tensor(v)) for v in losses.values())
    print(f"[{label:28s}] {ups:6.1f} upd/s   finite={finite}  "
          f"cons={losses.get('consistency_loss', 0):.4f} val={losses.get('value_loss', 0):.4f} "
          f"rew={losses.get('reward_loss', 0):.4f}")
    return ups


def main():
    print(f"device={torch.cuda.get_device_name(0)} torch={torch.__version__}\n")
    base = run("eager fp32 (baseline)", False, False, False)
    run("eager + TF32",               False, True,  False)
    run("eager + TF32 + bf16",        False, True,  True)
    print()
    c = run("compile fp32 (current)",     True,  False, False)
    run("compile + TF32",             True,  True,  False)
    run("compile + TF32 + bf16",      True,  True,  True)
    print(f"\nbaseline eager fp32 = {base:.0f} upd/s ; compile fp32 = {c:.0f} upd/s ({c/base:.2f}x)")


if __name__ == "__main__":
    main()
