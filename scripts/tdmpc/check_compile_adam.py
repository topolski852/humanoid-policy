"""D4 check: is Adam still correct when `_update` is wrapped in torch.compile(reduce-overhead)?

WHY. The official TD-MPC2 constructs BOTH optimizers with `capturable=True`
(tdmpc2.py:29,31) precisely because it compiles `_update` with mode="reduce-overhead",
which drives CUDA graphs. Our port compiles the same region (agent.py:53) but omits
`capturable` (agent.py:36-42). With `capturable=False` Adam keeps its `step` counter as a
CPU scalar; anything CPU-side inside a graph-captured region executes once at capture and
is then frozen. A frozen `step` freezes the bias corrections
`1-beta1**step` / `1-beta2**step`, which silently pins the effective learning rate at its
first-step value forever. Every fleet since 07-20 ran with --compile.

This does NOT need Isaac: the agent only needs obs/action dims and a batch dict, so the
whole thing runs on synthetic data in seconds.

Primary signal is the step counter, which is RNG-independent and unambiguous. Loss and
parameter comparisons are secondary because torch.compile legitimately consumes RNG in a
different order than eager, so small divergence there is expected and is NOT evidence of a
bug on its own.

Usage:  .venv/bin/python scripts/tdmpc/check_compile_adam.py
        .venv/bin/python scripts/tdmpc/check_compile_adam.py --fix   # re-test with capturable=True
"""

from __future__ import annotations

import argparse
import sys

import torch

from humanoid_policy.tdmpc.agent import TDMPC2
from humanoid_policy.tdmpc.config import TdmpcAgentCfg

OBS_DIM, ACT_DIM = 45, 12


def make_cfg(compile_on: bool):
    cfg = TdmpcAgentCfg()
    cfg.compile = compile_on
    cfg.use_tdmpc2_square = True
    cfg.plan_collection = True
    return cfg


def synth_batch(cfg, device, gen):
    """A batch shaped exactly like SequenceReplayBuffer.sample()."""
    H, B = cfg.horizon, cfg.batch_size
    return {
        "obs": torch.randn(H + 1, B, OBS_DIM, device=device, generator=gen),
        "action": torch.randn(H, B, ACT_DIM, device=device, generator=gen).clamp(-1, 1),
        "reward": torch.randn(H, B, device=device, generator=gen) * 0.1,
        "terminated": torch.zeros(H, B, device=device),
        "plan_mean": torch.randn(H, B, ACT_DIM, device=device, generator=gen).clamp(-1, 1),
        "plan_std": torch.full((H, B, ACT_DIM), 0.3, device=device),
    }


def adam_steps(optim):
    """The `step` counter Adam actually used, per param group."""
    out = []
    for p, st in optim.state.items():
        if "step" in st:
            s = st["step"]
            out.append(float(s.item()) if torch.is_tensor(s) else float(s))
    return out


def run(compile_on: bool, n_updates: int, device: str, capturable: bool | None = None):
    torch.manual_seed(0)
    cfg = make_cfg(compile_on)
    agent = TDMPC2(cfg, OBS_DIM, ACT_DIM, device)
    if capturable is not None:
        # rebuild both optimizers with the reference's setting
        agent.optim = torch.optim.Adam([
            {"params": agent.model._encoder.parameters(), "lr": cfg.lr * cfg.enc_lr_scale},
            {"params": agent.model._dynamics.parameters()},
            {"params": agent.model._reward.parameters()},
            {"params": agent.model._Qs.parameters()},
        ], lr=cfg.lr, capturable=capturable)
        agent.pi_optim = torch.optim.Adam(agent.model._pi.parameters(), lr=cfg.lr, eps=1e-5,
                                          capturable=capturable)

    gen = torch.Generator(device=device); gen.manual_seed(1234)
    batches = [synth_batch(cfg, device, gen) for _ in range(n_updates)]

    losses = []
    for i, b in enumerate(batches):
        torch.manual_seed(10_000 + i)          # same dropout/randperm draws in both arms
        info = agent.update(b)
        losses.append(float(info["total_loss"]))
    torch.cuda.synchronize()

    w = torch.cat([p.detach().flatten() for p in agent.model._dynamics.parameters()])
    return {
        "losses": losses,
        "model_steps": adam_steps(agent.optim),
        "pi_steps": adam_steps(agent.pi_optim),
        "dyn_w_norm": float(w.norm()),
        "dyn_w_sum": float(w.sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=40)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fix", action="store_true", help="also test compiled + capturable=True")
    a = ap.parse_args()

    print(f"torch {torch.__version__} | device {a.device} | {a.updates} updates on synthetic batches\n")

    print("--- eager (compile=False) ...")
    eager = run(False, a.updates, a.device)
    print("--- compiled (compile=True, capturable unset == our production path) ...")
    comp = run(True, a.updates, a.device)

    rows = [("eager", eager), ("compiled (ours)", comp)]
    if a.fix:
        print("--- compiled + capturable=True (the reference's setting) ...")
        rows.append(("compiled+capturable", run(True, a.updates, a.device, capturable=True)))

    print(f"\n{'arm':<22}{'adam step (model)':>19}{'adam step (pi)':>16}{'loss[0]':>10}{'loss[-1]':>10}{'|W_dyn|':>11}")
    print("-" * 88)
    for name, r in rows:
        ms = set(r["model_steps"]); ps = set(r["pi_steps"])
        msd = f"{min(ms):.0f}" if len(ms) == 1 else f"MIXED {sorted(ms)[:3]}"
        psd = f"{min(ps):.0f}" if len(ps) == 1 else f"MIXED {sorted(ps)[:3]}"
        print(f"{name:<22}{msd:>19}{psd:>16}{r['losses'][0]:>10.4f}{r['losses'][-1]:>10.4f}{r['dyn_w_norm']:>11.4f}")

    print(f"\nEXPECTED adam step after {a.updates} updates: {a.updates}")
    verdict = 0
    for name, r in rows:
        got = min(set(r["model_steps"])) if r["model_steps"] else -1
        if abs(got - a.updates) > 0.5:
            print(f"  *** {name}: step counter is {got:.0f}, NOT {a.updates} -> bias correction is wrong")
            verdict = 1
        else:
            print(f"  OK  {name}: step counter advanced correctly ({got:.0f})")

    # secondary: did the weights actually move, and do the two arms stay in the same regime?
    d = abs(comp["dyn_w_norm"] - eager["dyn_w_norm"]) / max(eager["dyn_w_norm"], 1e-9)
    print(f"\nsecondary — |W_dyn| relative difference eager vs compiled: {d*100:.3f}%")
    print("  (small differences are EXPECTED: torch.compile consumes RNG in a different order.")
    print("   A large gap, or a frozen step counter above, is what would indicate a real bug.)")
    return verdict


if __name__ == "__main__":
    sys.exit(main())
