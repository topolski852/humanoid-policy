#!/usr/bin/env python3
"""One-glance status for a supervisor fleet. Read-only, no Isaac, no GPU.

Why this exists: the per-run log under logs/tdmpc/_runlogs/ is NOT a reliable progress signal.
The supervisor redirects the training process's stdout to a file, so Python block-buffers it and
`print()` output appears in ~8 kB chunks — a healthy run can look frozen for many minutes. The
TensorBoard event file is written by SummaryWriter on its own schedule, so that is what this reads.

Usage:
    .venv/bin/python scripts/tdmpc/fleet_status.py                 # one shot (default tag: lean)
    .venv/bin/python scripts/tdmpc/fleet_status.py --watch         # refresh every 60 s
    .venv/bin/python scripts/tdmpc/fleet_status.py --tag stability # a different fleet
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
from eureka.tb_utils import read_scalars  # noqa: E402

RUNS_DIR = os.path.join(REPO_ROOT, "logs", "tdmpc", "tdmpc_biped")
STATE_DIR = os.path.join(REPO_ROOT, "logs", "tdmpc")
SUP_LOG = os.path.join(STATE_DIR, "_runlogs", "supervisor.log")
WIN_LEAN_DEG = 15.0      # keep in step with grade_run.py
WIN_FWD_SPEED = 0.25


def _tail_mean(series, span):
    """Mean of a [(step, value)] series over the last `span` env-steps."""
    if not series:
        return None
    cutoff = series[-1][0] - span
    vals = [v for s, v in series if s >= cutoff]
    return sum(vals) / len(vals) if vals else None


def _live_proc():
    """(pid, elapsed) of the running trainer, or (None, None)."""
    try:
        out = subprocess.check_output(
            ["ps", "-eo", "pid,etime,cmd"], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return None, None
    for line in out.splitlines():
        if ("bootstrap.py" in line or "train.py" in line) and "grep" not in line:
            parts = line.split()
            return parts[0], parts[1]
    return None, None


def _live_arm(tag):
    """(name, run_dir) of the most recently launched spec, from the supervisor's own (flushed) log.

    run_dir comes from the supervisor's "run dir:" line rather than "newest directory in
    tdmpc_biped/", because touching an older run's files (writing a grade, say) makes that stale
    directory the newest and the status silently reports the previous arm's numbers as the live
    one's. run_dir is None until the supervisor has discovered it (~1-2 min after launch).
    """
    if not os.path.exists(SUP_LOG):
        return None, None
    name = run_dir = None
    with open(SUP_LOG, errors="ignore") as f:
        for line in f:
            if f"journal=supervisor_journal_{tag}.jsonl" in line:
                name = run_dir = None             # this fleet started here; forget earlier fleets
            m = re.search(r"launch '([^']+)'", line)
            if m:
                name, run_dir = m.group(1), None  # new arm -> its dir is not known yet
            m = re.search(r"run dir: (\S+)", line)
            if m:
                run_dir = m.group(1)
    return name, run_dir


def _recent_sps(run_dir, tag="collect/ground_speed_mps", points=60):
    """Steps/sec from the RECENT slope of the event file's wall-clock times.

    Not `collect/env_steps_per_sec` — that scalar is total/(elapsed since train() started), a
    cumulative average that still carries the multi-minute pretrain burst, so early in a run it
    under-reports the true rate by ~10x and the ETA comes out absurd.
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ev = sorted(glob.glob(os.path.join(run_dir, "events.out.tfevents.*")))
    if not ev:
        return None
    ea = EventAccumulator(ev[-1], size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return None
    s = ea.Scalars(tag)[-points:]
    if len(s) < 2:
        return None
    dt, dstep = s[-1].wall_time - s[0].wall_time, s[-1].step - s[0].step
    return dstep / dt if dt > 1.0 else None


def _fmt_eta(run_dir, step, budget):
    sps = _recent_sps(run_dir)
    if not sps or step >= budget:
        return ""
    return f"  eta ~{(budget - step) / sps / 3600.0:.1f} h  ({sps:.0f} steps/s)"


def show(tag):
    queue = os.path.join(REPO_ROOT, "scripts", "tdmpc", f"{tag}_queue.jsonl")
    specs = []
    if os.path.exists(queue):
        specs = [json.loads(l) for l in open(queue)
                 if l.strip() and not l.lstrip().startswith("#")]

    journal = os.path.join(STATE_DIR, f"supervisor_journal_{tag}.jsonl")
    graded = []
    if os.path.exists(journal):
        graded = [json.loads(l) for l in open(journal) if l.strip()]

    pid, elapsed = _live_proc()
    arm, run = _live_arm(tag)
    print(f"\n\033[1mFLEET '{tag}'\033[0m  {len(graded)}/{len(specs)} graded"
          + (f"   running: {arm} (pid {pid}, up {elapsed})" if pid else "   \033[33mno trainer running\033[0m"))

    if graded:
        print("\n  GRADED")
        for e in graded:
            g = e.get("grade", {})
            raw = g.get("_raw", {})
            lean = raw.get("torso_lean_fwd_deg")
            leans = "  n/a" if lean is None else f"{lean:+6.1f}"
            flag = "" if lean is None or lean <= WIN_LEAN_DEG else "  <- LEANING"
            print(f"    {e['spec']['name']:20s} fwd {raw.get('forward_speed_mean', 0):5.3f} m/s"
                  f"   lean {leans} deg   fall/min {raw.get('fall_rate_per_min', '?')}"
                  f"   win={g.get('is_win')}{flag}")

    if not pid and not run:
        return
    if run is None:
        print(f"\n  LIVE  {arm}   starting up — Isaac boot + 400k-step seed + the 20k-update "
              f"pretrain burst run ~6 min before the first scalar appears")
        return
    sc = read_scalars(run)
    speed = sc.get("collect/ground_speed_mps", [])
    lean = sc.get("collect/torso_lean_deg", [])
    ret = sc.get("collect/mean_episode_return", [])
    eplen = sc.get("collect/mean_episode_len", [])

    budget = 3_000_000
    cfgp = os.path.join(run, "run_config.json")
    if os.path.exists(cfgp):
        budget = int(json.load(open(cfgp)).get("max_env_steps", budget))

    print(f"\n  LIVE  {arm or '?'}   {os.path.relpath(run, REPO_ROOT)}")
    if not speed:
        print("    no scalars yet — still seeding / in the pretrain burst "
              "(the burst is ~20k updates and logs nothing until it ends)")
        return
    step = speed[-1][0]
    print(f"    step {step/1e6:5.2f}M / {budget/1e6:.2f}M  ({100*step/budget:4.1f}%)"
          + _fmt_eta(run, step, budget))
    r100 = _tail_mean(speed, 100_000)
    print(f"    speed  {speed[-1][1]:5.3f} m/s   last100k {r100:5.3f}"
          f"   {'OK' if r100 and r100 > 0.15 else 'LOW - check it is still walking'}")
    if lean:
        l100 = _tail_mean(lean, 100_000)
        verdict = "OK" if abs(l100) <= WIN_LEAN_DEG else "OVER BAR"
        print(f"    lean  {lean[-1][1]:+6.1f} deg   last100k {l100:+6.1f}"
              f"   {verdict}  (bar {WIN_LEAN_DEG:.0f}, 07-24 winner was +27.3)")
    else:
        print("    lean   (no torso_lean_deg tag — run predates the posture telemetry)")
    if ret:
        print(f"    return {ret[-1][1]:6.2f}   ep_len {eplen[-1][1]:.0f}"
              f"   {'(500 = no collapses)' if eplen and eplen[-1][1] >= 499 else ''}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", default="lean", help="supervisor --run_tag of the fleet (default: lean)")
    p.add_argument("--watch", action="store_true", help="refresh until interrupted")
    p.add_argument("--every", type=int, default=60, help="seconds between refreshes with --watch")
    a = p.parse_args()
    while True:
        if a.watch:
            print("\033[2J\033[H", end="")     # clear
        show(a.tag)
        if not a.watch:
            return
        print(f"\n  refreshing every {a.every}s — ctrl-C to stop")
        time.sleep(a.every)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
