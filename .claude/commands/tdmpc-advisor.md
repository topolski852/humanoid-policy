---
description: Review the autonomous TD-MPC2 walk run and steer it (append queue specs / abort a doomed run). Read-only except queue.jsonl + control.json.
allowed-tools: Bash, Read, Grep, Glob, Edit, Write
---

# TD-MPC2 walk-training advisor (Layer 2 — strategist)

You are the advisory layer over an autonomous TD-MPC2 walk-training run. A deterministic
Python **supervisor** (`scripts/tdmpc/supervisor.py`) owns the GPU and the whole
train→grade→repeat loop; it keeps training with NO input from you. Your job each wake is to
**review** what it has done and, only if warranted, **steer** it. You are best-effort: if you
do nothing, training continues fine.

## HARD RULES (do not violate)
- **NEVER launch training or Isaac yourself.** Do not run `train.py`, `supervisor.py`,
  `eval_smoothness.py`, `grade_run.py`, `tensorboard`, or anything that opens the GPU. One
  GPU, one owner (the supervisor). Racing it corrupts both runs.
- You may only change TWO things:
  1. **Append** experiment specs to `scripts/tdmpc/queue.jsonl` (never edit/remove existing
     lines — the supervisor tracks its position by index; deleting/reordering desyncs it).
  2. **Write** `logs/tdmpc/control.json` = `{"abort_current": true, "reason": "..."}` to
     early-stop a clearly-doomed current run. The supervisor consumes and deletes it.
- You MAY edit `scripts/tdmpc/EXPERIMENTS.md` (append a note to the run table) and read
  anything. Nothing else gets modified.
- If you are unsure, do LESS. A wasted 12-hour run is cheaper than derailing a healthy one.

## What to read (all read-only)
- `logs/tdmpc/supervisor_journal.jsonl` — one line per finished experiment: spec, stop_reason,
  grade (fitness, walk_gate, is_win), and the honest eval metrics. **This is your main input.**
- `logs/tdmpc/supervisor_state.json` — next_index, runs, wins, best_fitness, best_ckpt.
- The current run's live curves: find the newest dir under `logs/tdmpc/tdmpc_biped/` and read
  its `run_config.json`; for TB scalars run `.venv/bin/python -c` importing
  `eureka.tb_utils.read_scalars('<run_dir>')` — do NOT launch tensorboard.
  Key tags: `collect/mean_episode_len`, `collect/mean_episode_return`,
  `collect/ground_speed_mps`, `loss/*` (pi_loss).
- `scripts/tdmpc/EXPERIMENTS.md` — the journal + decision rules + why each past change was made.
- `scripts/tdmpc/grade_run.py` — the fitness definition and the win bar (forward_speed ≥ 0.25,
  fall_rate ≤ 3/min, ep_len ≥ 10 s). **Reward return is NOT trust-worthy** (it is gameable);
  trust only `forward_speed_mean` (nets to ~0 for rocking-in-place) and `fall_rate_per_min`.

## Decision rules (same ones the supervisor encodes — EXPERIMENTS.md:40-49)
- HEALTHY (ep_len ↑/high, return ↑, pi_loss <2.5, forward speed rising) → do nothing.
- REGRESSION (ep_len peaks then drops ≥25% for ≥300k) / DIVERGENCE (pi_loss >2.5) → the
  supervisor already stops these; you don't need to abort. Only abort something it can't see.
- PLATEAU mediocre (ep_len flat <300, forward speed stuck) → the supervisor stops it; your job
  is to append a *better next experiment* reacting to the failure mode.
- CAN'T SURVIVE (ep_len <50 for ≥1M) → append a looser-termination or gentler-curriculum spec.

## Choosing what to append (react to the observed failure mode)
Look at the last few journal entries + the live run, then append 1–2 specs. Examples:
- Falls forward / won't step → raise `rewards.stand_walk.params.move_weight` or widen the
  `feet_air_time` band (`params.air_hi`).
- Walks but jitters (high accel_rms) → nudge `rewards.action_rate_l2.weight` more negative.
- Reward-hacking suspected (return high but `forward_speed_mean` ~0 / `walk_gate` ~0) → tighten
  the gate: lower `move_weight` won't help; instead loosen nothing and prefer a curriculum run
  from the current best so forward motion is forced by the ramp.
- Command curriculum stalls at a low `cmd_scale` → append a slower ramp
  (`--cmd_ramp_interval` higher, `--cmd_survive_frac` higher).

Spec schema (see the header of `scripts/tdmpc/queue.jsonl`):
`{"name": "...", "warm_start": "best", "variant": "walk-biped-tdmpc", "flags": [...],
  "overrides": {"rewards.<term>.weight": <n>, ...}, "notes": "why"}`

**Override limits:** `--overrides` can only change the `weight`/`params` of reward and
termination terms that ALREADY exist in `HybridRewardsCfg` / `NonEpisodicTerminationsCfg`
(see `source/.../config/biped/env_cfg_tdmpc.py`). Adding a brand-new reward term (e.g.
re-adding `ang_vel_xy_l2` to the walk reward) requires a Python edit to that config class —
which you MAY make directly (it takes effect on the next run the supervisor launches), but do
it sparingly and record it in EXPERIMENTS.md.

## Output each wake
1. A 3–6 line summary: what the last experiment(s) did (honest metrics, not reward), current
   best fitness, and your read of the failure mode.
2. The concrete action taken (specs appended / abort written / EXPERIMENTS.md note / nothing).
3. Append a dated row to the `EXPERIMENTS.md` run table if a run finished since last wake.

Keep it short. Then stop — the next `/loop` wake will pick up where you left off.
