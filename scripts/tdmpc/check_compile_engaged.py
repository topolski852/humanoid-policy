"""Companion to check_compile_adam.py: did torch.compile ACTUALLY engage, or silently fall back?

check_compile_adam showed eager and compiled producing bit-identical results. That is the
right answer if compilation is numerically faithful — but it is ALSO what you would see if
`torch.compile` errored and fell back to eager, which agent.py's docstring says it does
silently. Distinguish the two with (a) dynamo's compiled-frame counter and (b) steady-state
wall time per update. The port claims ~3.6x faster updates from compiling.
"""

from __future__ import annotations

import time

import torch

from humanoid_policy.tdmpc.agent import TDMPC2
from humanoid_policy.tdmpc.config import TdmpcAgentCfg

OBS_DIM, ACT_DIM = 45, 12


def batch(cfg, device, gen):
    H, B = cfg.horizon, cfg.batch_size
    return {
        "obs": torch.randn(H + 1, B, OBS_DIM, device=device, generator=gen),
        "action": torch.randn(H, B, ACT_DIM, device=device, generator=gen).clamp(-1, 1),
        "reward": torch.randn(H, B, device=device, generator=gen) * 0.1,
        "terminated": torch.zeros(H, B, device=device),
        "plan_mean": torch.randn(H, B, ACT_DIM, device=device, generator=gen).clamp(-1, 1),
        "plan_std": torch.full((H, B, ACT_DIM), 0.3, device=device),
    }


def bench(compile_on, device="cuda:0", warmup=15, iters=60):
    import torch._dynamo as dynamo
    dynamo.reset()
    dynamo.utils.counters.clear()
    torch.manual_seed(0)
    cfg = TdmpcAgentCfg(); cfg.compile = compile_on
    cfg.use_tdmpc2_square = True; cfg.plan_collection = True
    agent = TDMPC2(cfg, OBS_DIM, ACT_DIM, device)
    gen = torch.Generator(device=device); gen.manual_seed(1234)
    b = batch(cfg, device, gen)

    for _ in range(warmup):
        agent.update(b)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        agent.update(b)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    ok_frames = dynamo.utils.counters["frames"].get("ok", 0)
    graph_breaks = sum(dynamo.utils.counters["graph_break"].values())
    return dt, ok_frames, graph_breaks


if __name__ == "__main__":
    print(f"torch {torch.__version__}\n")
    e_dt, e_ok, _ = bench(False)
    c_dt, c_ok, c_gb = bench(True)
    print(f"{'arm':<14}{'ms/update':>12}{'dynamo frames ok':>19}{'graph breaks':>15}")
    print("-" * 60)
    print(f"{'eager':<14}{e_dt*1e3:>12.2f}{e_ok:>19}{'-':>15}")
    print(f"{'compiled':<14}{c_dt*1e3:>12.2f}{c_ok:>19}{c_gb:>15}")
    print(f"\nspeedup: {e_dt/c_dt:.2f}x   (the port's docstring claims ~3.6x)")
    if c_ok == 0:
        print("*** compile did NOT engage (0 compiled frames) — it fell back to eager.")
    else:
        print(f"OK: torch.compile engaged ({c_ok} compiled frames).")
