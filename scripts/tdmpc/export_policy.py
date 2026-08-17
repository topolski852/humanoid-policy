"""Export a TD-MPC2 checkpoint as a deployable policy bundle for humanoid-control.

WHAT GETS EXPORTED, AND WHAT DOES NOT. humanoid-control's `OnnxPolicy`/`TorchPolicy`
(humanoid_control/policy.py) call a single-input model once per control tick:
`sess.run(None, {input: obs(1,45)})[0]` -> action, first 12 taken. That is a feedforward
map, so only TD-MPC2's **policy prior** (`encode -> pi -> tanh(mean)`) can be exported this
way. The MPPI planner cannot: at H=6 it is 512 samples x 6 steps x 6 iterations = 18,432
world-model forward passes per 40 ms tick.

That distinction is not free. Measured on model_best.pt of the 2026-08-16 run, 193 episodes:

    policy path                     %eps fall   falls/min    fwd    x cmd   tilt
    MPPI H=6 (what we graded)          10.9%       0.54     0.431   1.44   8.66
    policy PRIOR only (this export)    17.1%       0.72     0.402   1.34  10.13
    PPO reference (deployed today)      6.2%       0.23     0.281   0.94   9.70

So exporting the prior costs ~6 pp of fall rate versus the graded number, and is still worse
than the PPO policy currently deployed. Deploying the planner would need MPPI on the target
at 25 Hz -- a separate project, not an export.

ACTION UNITS. The agent acts in [-1,1]; the env multiplies by `act_env_scale` (4.0), which
equals the walk env's `_ACTION_RAW_LIMIT`. The deploy contract's formula is
`target = clip(action)*action_scale + default_pose` with `action_scale` 0.25 and limits +-4,
i.e. it expects the RAW action. So the exported model emits `tanh(mean) * act_env_scale`,
matching what PPO's exporter emits and what the robot already consumes.

The contract/yaml are copied from `deploy/walk/` rather than regenerated: the TD-MPC2 walk env
inherits the PPO env's `ObservationsCfg` and `ActionsCfg` unchanged (verified 2026-08-12), so
joint order, gains, limits, obs layout and action scale are all identical. Only the network and
its provenance differ.

Usage (no Isaac, no GPU):
    .venv/bin/python scripts/tdmpc/export_policy.py \
        --checkpoint logs/tdmpc/tdmpc_biped/2026-08-16_13-50-26/model_best.pt \
        --out deploy/walk_tdmpc
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import torch
import torch.nn as nn

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

from humanoid_policy.tdmpc.agent import TDMPC2  # noqa: E402
from humanoid_policy.tdmpc.config import TdmpcAgentCfg  # noqa: E402

OBS_DIM, ACT_DIM = 45, 12


class ExportedPrior(nn.Module):
    """obs (N,45) -> raw action (N,12) in the deploy contract's +-act_env_scale units.

    Mirrors `TDMPC2.act_pi(obs, eval_mode=True)` exactly: encode, take the policy-prior mean,
    tanh-squash it (world_model.pi applies `math.squash`, which is tanh on the mean), then scale
    to raw units. No sampling, no dropout — deterministic.
    """

    def __init__(self, encoder: nn.Module, pi: nn.Module, act_scale: float, act_dim: int):
        super().__init__()
        self.encoder = encoder
        self.pi = pi
        self.act_scale = float(act_scale)
        self.act_dim = int(act_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        z = self.encoder(obs)
        mu = self.pi(z)[..., : self.act_dim]      # chunk(2)[0] == the mean half
        return torch.tanh(mu) * self.act_scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default=os.path.join(REPO, "deploy", "walk_tdmpc"))
    ap.add_argument("--contract-from", default=os.path.join(REPO, "deploy", "walk"),
                    help="copy the joint/gain/limit contract from this bundle (same robot).")
    ap.add_argument("--samples", type=int, default=4096, help="random obs used to verify parity.")
    a = ap.parse_args()

    cfg = TdmpcAgentCfg()
    agent = TDMPC2(cfg, OBS_DIM, ACT_DIM, "cpu")
    agent.load(a.checkpoint)
    agent.model.eval()

    net = ExportedPrior(agent.model._encoder, agent.model._pi, cfg.act_env_scale, ACT_DIM).eval()

    # ---- parity: exported net must equal the agent's own deterministic action --------------
    torch.manual_seed(0)
    obs = torch.randn(a.samples, OBS_DIM)
    with torch.no_grad():
        ref = agent.act_pi(obs, eval_mode=True) * cfg.act_env_scale
        got = net(obs)
    err = (ref - got).abs().max().item()
    print(f"[export] parity vs agent.act_pi(eval_mode=True): max abs err {err:.3e}")
    assert err < 1e-5, "exported net does not match the agent"
    print(f"[export] action range over {a.samples} random obs: "
          f"[{got.min():.2f}, {got.max():.2f}] (contract limit +-{cfg.act_env_scale})")

    os.makedirs(a.out, exist_ok=True)

    # ---- TorchScript ----------------------------------------------------------------------
    ts = torch.jit.trace(net, torch.randn(1, OBS_DIM))
    ts_path = os.path.join(a.out, "policy.pt")
    ts.save(ts_path)

    # ---- ONNX -----------------------------------------------------------------------------
    onnx_path = os.path.join(a.out, "policy.onnx")
    torch.onnx.export(net, torch.randn(1, OBS_DIM), onnx_path,
                      input_names=["obs"], output_names=["actions"],
                      dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
                      opset_version=17)

    # ---- verify BOTH artefacts round-trip to the same numbers ------------------------------
    with torch.no_grad():
        ts_out = ts(obs)
    print(f"[export] torchscript  max abs err vs net: {(ts_out - got).abs().max().item():.3e}")
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name
        onnx_out = torch.tensor(sess.run(None, {name: obs.numpy()})[0])
        e = (onnx_out - got).abs().max().item()
        print(f"[export] onnxruntime  max abs err vs net: {e:.3e}"
              + ("  <-- exceeds 1e-4, investigate" if e > 1e-4 else ""))
    except ImportError:
        print("[export] onnxruntime not installed — ONNX written but NOT verified. "
              "humanoid-control loads it via onnxruntime, so verify before flashing.")

    # ---- contract + yaml (same robot; only the net changed) --------------------------------
    for fn in ("leg_policy_contract.json", "policy_latest.yaml"):
        src = os.path.join(a.contract_from, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(a.out, fn))
    cpath = os.path.join(a.out, "leg_policy_contract.json")
    if os.path.exists(cpath):
        with open(cpath) as f:
            c = json.load(f)
        c["_meta"] = {
            "source": "humanoid-policy scripts/tdmpc/export_policy.py (TD-MPC2 policy prior)",
            "checkpoint": os.path.relpath(os.path.abspath(a.checkpoint), REPO),
            "algorithm": "TD-MPC2 (policy prior only — MPPI planner NOT exported)",
            "measured_prior": {"pct_episodes_with_fall": 17.1, "real_falls_per_min": 0.72,
                               "forward_speed_mean": 0.402, "cmd_vx": 0.3, "torso_tilt_deg": 10.13},
            "measured_with_mppi_h6": {"pct_episodes_with_fall": 10.9, "real_falls_per_min": 0.539,
                                      "forward_speed_mean": 0.431},
            "ppo_reference_currently_deployed": {"pct_episodes_with_fall": 6.2,
                                                 "real_falls_per_min": 0.23,
                                                 "forward_speed_mean": 0.281},
            "warning": "This prior falls MORE than the PPO policy already deployed (17.1% vs 6.2% "
                       "of episodes). Do not flash it as a replacement without a bench comparison.",
            "note": "obs/action layout inherited unchanged from the PPO walk env, so joint order, "
                    "gains, limits and action_scale below are unmodified.",
        }
        with open(cpath, "w") as f:
            json.dump(c, f, indent=2)

    print(f"\n[export] bundle written to {os.path.relpath(a.out, REPO)}/")
    for fn in sorted(os.listdir(a.out)):
        print(f"           {fn:<28} {os.path.getsize(os.path.join(a.out, fn)):>9,} bytes")
    print("\n[export] load with humanoid_control.policy.load_policy(<path>/policy.onnx)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
