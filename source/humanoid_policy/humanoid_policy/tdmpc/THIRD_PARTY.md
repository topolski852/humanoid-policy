# Third-party attribution

The model primitives, world model, and agent in this package are **adapted from TD-MPC2**
(Nicklas Hansen, Hao Su, Xiaolong Wang), released under the **MIT License**:

- Upstream: https://github.com/nicklashansen/tdmpc2  (paper: arXiv:2310.16828)
- Files adapted here: `common/{layers,math,scale,init}.py`, `world_model.py`, `agent.py`
  (from upstream `common/{layers,math,scale,init}.py`, `common/world_model.py`, `tdmpc2.py`).

Adaptations for this repo (see docs/proactive_world_model/DESIGN.md and the approved plan):
- Single-task, state-only, non-episodic, non-multitask (multitask/pixel/episodic branches removed).
- The tensordict/`from_modules`+`torch.vmap` Q-ensemble is replaced by a plain `nn.ModuleList`
  (target Qs = deepcopy soft-updated) so it runs on torch 2.11 without the upstream
  `torchrl`/`tensordict` version pins.
- Device-agnostic; eager only (no `torch.compile`/CUDA-graphs); actions normalized to [-1,1] and
  mapped to the env's raw JointPosition convention by the trainer.
- The upstream torchrl replay buffer is replaced by our own `buffer.SequenceReplayBuffer`.

The MIT license permits this adaptation provided the copyright notice is retained. Related
successors also MIT and available if adopted later: TD-M(PC)²
(https://github.com/DarthUtopian/tdmpc_square_public) and Newt
(https://github.com/nicklashansen/newt).

---
MIT License — Copyright (c) 2023 Nicklas Hansen. See the upstream LICENSE for full text.
