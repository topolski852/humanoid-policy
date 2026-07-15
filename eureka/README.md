# eureka — automated reward-weight tuner (API-free)

Brings the humanoid-tuner's Eureka loop into this trainer: an **evolutionary search
over the walk reward-term weights**, where each candidate is a real training run that
is **stopped early the moment its reward plateaus**, then graded by a weight-invariant
ground-truth fitness, and the winners are evolved into the next generation. No Claude
API key needed — the "propose/reflect" step is a local genetic search, not an LLM call.

It **wraps** the existing rsl_rl trainer (`scripts/rsl_rl/train.py`) — it does not
modify it. All Eureka code lives in this folder.

## The loop

```
 propose weights ──► train (plateau early-stop) ──► grade (TB fitness) ──► evolve ──┐
      ▲  local evolutionary search           weight-invariant ground truth          │
      └────────────────────────────────────────────────────────────────────────────┘
```

- **Propose** (`propose_local.py`): mutate / crossover / random over the 17 walk
  reward-term weights; signs fixed (a penalty can't become a bonus); `0` prunes a term.
- **Train** (`run.py`): launch `train.py` with the weights as Hydra overrides
  (`env.rewards.<term>.weight=...`); poll `Train/mean_reward` and kill the run once it
  plateaus (`--inner-patience`).
- **Grade** (`evaluate.py`): read the run's TensorBoard and compute a fitness that the
  reward weights **cannot** influence — `Metrics/success_rate`, velocity-tracking error
  (`Metrics/base_velocity/error_vel_*`), fall rate (`Episode_Termination/base_orientation`),
  and survival (`Train/mean_episode_length`). That independence is what makes ranking honest.
- **Evolve** (`search.py`): keep the best, evolve the rest, stop when the search plateaus
  (`--patience`).

## Run it

```bash
# 1) verify TB -> fitness on an existing run, NO training/API:
OMNI_KIT_ACCEPT_EULA=YES .venv/bin/python -m eureka.search --dry-run

# 2) cheap end-to-end smoke (real training, minutes):
OMNI_KIT_ACCEPT_EULA=YES .venv/bin/python -m eureka.search \
    --profile fast --max_iterations 40 --inner-patience 8 --iterations 1 --candidates 2

# 3) the overnight search:
OMNI_KIT_ACCEPT_EULA=YES .venv/bin/python -m eureka.search \
    --profile fast --max_iterations 250 --iterations 8 --candidates 6 \
    --patience 3 --backend subprocess --log eureka_history.jsonl
```

Output: `eureka_best.json` (winning weights + fitness + a ready-to-paste Hydra
override string) and, with `--log`, a JSONL of every candidate. Per-candidate run dirs
land at `logs/rsl_rl/<experiment>/<ts>_eureka-g<g>-c<i>/`.

**Reproduce the winner** — paste its `hydra_override` onto a normal full training run,
or bake the weights back into `RewardsCfg`.

## Backends (staged)

- `--backend subprocess` (default, **shipped**): one `train.py` per candidate. Fully
  additive, correct baseline. Pays ~30 min Isaac startup **per candidate**.
- `--backend persistent` (**scaffold**, `run_persistent.py`): one long-lived Isaac
  process, weights mutated live between training bouts — startup paid **once**. Build +
  cross-check against subprocess before trusting it overnight.

## Notes

- Fitness weights and the term/sign table are in `config.py` — the one place to edit to
  retarget a task (e.g. standup) or reweight the objective.
- A Claude proposer stub (`propose.py`) is kept as the drop-in point for when an API key
  is available; the local search is the default and needs no credentials.
