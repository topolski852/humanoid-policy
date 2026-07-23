# Week-long autonomous TD-MPC2 walk run — RUNBOOK

One-page operator guide for the unattended train→test→adjust→repeat harness. Everything runs
from the repo root: `/home/nse/humanoid/humanoid-policy`.

## What it is
Two layers:
- **Supervisor (Layer 1, `supervisor.py`)** — owns the GPU. Launches each experiment from
  `queue.jsonl` warm-started from the best genuine walker so far, monitors it against the
  EXPERIMENTS.md decision rules (early-stops plateau/regression/divergence/can't-survive),
  **grades it with an honest, reward-hack-proof fitness** (a high-return faller scores ~0),
  preserves genuine winners, prunes disk, journals, and repeats. Idles after **3 wins**
  (`--max_wins`) or **40 runs** (`--max_runs`). Needs no human/LLM to keep going.
- **Advisor (Layer 2, `/tdmpc-advisor` via `/loop`)** — optional Claude review that only
  appends specs to `queue.jsonl` or drops an abort flag. If it dies, training is unaffected.

Honest signals (cannot be faked by reward weights): `forward_speed_mean` (body-frame forward
velocity — ~0 for rocking in place) and `fall_rate_per_min`, from `eval_smoothness.py`. The
win bar is defined in `grade_run.py`: forward ≥ 0.25 m/s, ≤ 3 falls/min, ≥ 10 s episodes.

## Before you leave (pre-flight)
```bash
cd /home/nse/humanoid/humanoid-policy
pgrep -af 'tdmpc/train.py|scripts/rsl_rl/train.py'    # MUST be empty — the supervisor owns the GPU;
                                                       # a second training process will OOM the 16 GB card.
sudo nvidia-smi -pm 1                                  # persistence mode (steady week-long GPU)
ls logs/tdmpc/_preserved/stand_v4_best.pt             # the initial warm-start MUST exist
df -h .                                                # need headroom; harness prunes to ~64 MB/run
rm -f logs/tdmpc/SUPERVISOR_DONE                       # clear any stale done-sentinel
# optional sanity (uses the GPU briefly — do this BEFORE starting the loop, not during):
.venv/bin/python scripts/tdmpc/grade_run.py --checkpoint logs/tdmpc/_preserved/stand_v4_best.pt --num_envs 8 --steps 200
#   -> expect walk_gate ~0, is_win=false (a stand is not a walk). Confirms the grader is honest.
```

## Start it (detached, survives logout)
```bash
cd /home/nse/humanoid/humanoid-policy
nohup bash scripts/tdmpc/run_supervisor.sh > logs/tdmpc/_runlogs/wrapper.log 2>&1 &
```
The wrapper restarts the supervisor on any crash (backoff up to 5 min) until the DONE
sentinel appears. Forward supervisor args through the wrapper, e.g. to run the full week
without idling: `... run_supervisor.sh --max_wins 99`.

## Start the Claude layers (optional — training runs fine without them)
Three tiers, each a separate **local** detached Claude Code session (they read/write local files;
a cloud/scheduled agent can't). Both auto-editing tiers need `--dangerously-skip-permissions` so
they don't prompt. Launch each in its own `tmux` window:

- **Layer 2 — advisor (sensor, every 6 h):** diagnoses, appends queue specs, and files structured
  machinery-bug alerts to `logs/tdmpc/ADVISOR_ALERTS.md`. Never edits code.
  ```
  cd /home/nse/humanoid/humanoid-policy && claude --dangerously-skip-permissions
  /loop 6h /tdmpc-advisor
  ```
- **Layer 3 — engineer (actuator, every 24 h):** reads the advisor's alerts and, ONLY when a bug is
  corroborated across **≥3 distinct runs**, applies + git-commits ONE scoped code fix
  (trainer.py / grade_run.py / env_cfg_tdmpc.py — auto-applies on the next run). Compile-checked,
  logged to `logs/tdmpc/ENGINEER_LOG.md`. supervisor.py bugs it leaves for you.
  ```
  cd /home/nse/humanoid/humanoid-policy && claude --dangerously-skip-permissions
  /loop 24h /tdmpc-engineer
  ```
Both are safe to skip. If they die, training continues; you just lose adaptivity/self-healing.
The three-tier split = **supervisor executes, advisor senses, engineer acts (only on corroborated
bugs)** — so no single component can churn the code or wedge the GPU.

**What to check on return:** `logs/tdmpc/ADVISOR_ALERTS.md` (bugs seen) and `logs/tdmpc/ENGINEER_LOG.md`
(fixes made — each has a git commit you can review/revert).

## Monitor (from anywhere, read-only — do NOT open tensorboard while training runs the GPU)
```bash
tail -f logs/tdmpc/_runlogs/supervisor.log            # live supervisor decisions
cat logs/tdmpc/supervisor_state.json                  # runs, wins, best_fitness, best_ckpt
tail -n 3 logs/tdmpc/supervisor_journal.jsonl | python3 -m json.tool   # last experiments + honest metrics
ls -t logs/tdmpc/_preserved/walk_win_*.pt 2>/dev/null | head           # preserved winners (best first)
```
`best_ckpt` in the state file is your current best walking policy. Winners are copied to
`logs/tdmpc/_preserved/walk_{win,best}_<fitness>_<ts>.pt`.

## Stop it
```bash
touch logs/tdmpc/SUPERVISOR_DONE                       # tell the wrapper to stop restarting
pkill -f scripts/tdmpc/supervisor.py                  # kill the supervisor + its training child
```

## After a machine reboot
State is persisted, so just re-run the start command — it resumes at the next queue index
with the same best-checkpoint pointer:
```bash
cd /home/nse/humanoid/humanoid-policy && rm -f logs/tdmpc/SUPERVISOR_DONE
nohup bash scripts/tdmpc/run_supervisor.sh > logs/tdmpc/_runlogs/wrapper.log 2>&1 &
```
(A reboot is the one failure the nohup wrapper can't survive on its own — this is the manual
step for it.)

## Watch a candidate walk (when you're back, or on a second machine — uses the GPU)
```bash
.venv/bin/python scripts/tdmpc/eval_smoothness.py \
  --checkpoint logs/tdmpc/_preserved/<best>.pt --plan --num_envs 4 --cmd_vx 0.3   # GUI (omit --headless)
```

## Knobs (edit the start command)
- `--max_wins N` (default 3) — idle-at-cap threshold; raise to keep improving all week.
- `--max_runs N` (default 40) — hard safety cap.
- `--max_env_steps N` (default 10_000_000) — per-run budget (~12.5 h); early-stop cuts bad runs sooner.
- `--min_judge_steps N` (default 2_000_000) — grace before plateau/regression rules apply (runs under-saturated ~1M).
- Queue: append specs to `scripts/tdmpc/queue.jsonl` (never edit/reorder existing lines).

## Files
- Runtime state (gitignored, in `logs/tdmpc/`): `supervisor_state.json`, `supervisor_journal.jsonl`,
  `control.json`, `SUPERVISOR_DONE`, `_overrides/`, `_runlogs/`.
- Preserved winners: `logs/tdmpc/_preserved/walk_*.pt`.
- Per-run: `logs/tdmpc/tdmpc_biped/<ts>/` holds `run_config.json`, `grade.json`,
  `eval_metrics.json`, `model_best.pt` (+ one pruned checkpoint).

## Troubleshooting
- **Every run grades ~0 / no wins:** expected early — the stand isn't a walk yet. Check the
  journal's honest `forward_speed_mean` trend across runs, not the training return.
- **Runs die instantly (CRASH exit):** check the newest `logs/tdmpc/_runlogs/<name>_*.log` for a
  Python/Isaac traceback; the supervisor retries twice then moves on and journals it.
- **Disk filling:** pruning keeps `model_best.pt` + one checkpoint per graded run (~64 MB). If a
  run is killed before grading, its intermediate checkpoints aren't pruned — clear old
  `logs/tdmpc/tdmpc_biped/<ts>/model_<N>.pt` manually (never delete a whole run dir you didn't
  inspect).
- **GPU Xid faults:** `dmesg | grep -i xid`; the wrapper + crash-retry will relaunch, but a
  wedged GPU needs a reboot (then re-run the start command).
