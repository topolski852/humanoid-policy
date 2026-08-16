"""Health check for a live TD-MPC2 run. Deterministic, no LLM, no GPU, no Isaac.

Both real problems in this project were caught by a human eyeballing TensorBoard, not by any
automation: the 2026-08-15 reward hack (mean_episode_len fell off 500) and the 2026-08-16 plateau
(peak at 4.65M then regression). Neither needed judgement to DETECT -- one is an invariant, the
other is a rolling-mean comparison with a z-test. Both are codified here.

Deliberately split by whether the verdict is ambiguous:

  CRITICAL -- an invariant is broken and the run is producing garbage. Safe to stop automatically.
    canary      mean_episode_len must sit at 500. Below it, hard_collapse is firing. Every healthy
                run in this project's history held exactly 500.0; take-1 of the long run sat at
                443.9 and falling because an ungated reward term paid a collapsed robot to slide.
    divergence  pi_loss > 1.5 sustained = value overoptimism (the project's own threshold).
    stalled     TensorBoard has not advanced -- the trainer is hung or dead.

  WARN -- the run is healthy but no longer improving. Stopping is a VALUE judgement about whether
    more GPU is worth it, so this escalates to a human and never auto-stops.
    plateau     improvement slope has decayed to ~0.
    regression  a rolling-mean peak followed by a statistically significant decline.

RETURN IS NEVER USED. Take-1 of the long run had the highest episode_return ever recorded in this
project while collapsing in ~22% of episodes.

Usage:
    .venv/bin/python scripts/tdmpc/watch_run.py                 # newest run, human-readable
    .venv/bin/python scripts/tdmpc/watch_run.py --json          # machine-readable verdict
    .venv/bin/python scripts/tdmpc/watch_run.py --run <dir>
    .venv/bin/python scripts/tdmpc/watch_run.py --selftest      # replay known-outcome runs
Exit code: 0 OK, 1 WARN, 2 CRITICAL.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics as st
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from eureka.tb_utils import read_scalars  # noqa: E402

RUNS = os.path.join(REPO, "logs", "tdmpc", "tdmpc_biped")

EP_LEN_FLOOR = 495.0      # below this, hard_collapse is firing
PI_LOSS_DIVERGE = 1.5     # the project's own overoptimism threshold
STALE_SECS = 2400         # no new TB point in 40 min = hung
WARMUP_STEPS = 400_000    # ignore the untrained head
ROLL_STEPS = 250_000      # rolling-mean window
Z_REGRESSION = -2.0       # peak-to-now, in SEs of the rolling mean
PLATEAU_SLOPE = 0.004     # m/s per 1M steps; below this speed has stopped improving
MIN_STEPS_TO_JUDGE = 2_000_000


def _roll(series, window_steps):
    if len(series) < 4:
        return [], 0.0
    step = max(series[1][0] - series[0][0], 1)
    W = max(3, int(window_steps / step))
    if len(series) <= W:
        return [], 0.0
    xs = [a for a, _ in series]
    vs = [b for _, b in series]
    roll = [(xs[i], sum(vs[i - W + 1:i + 1]) / W) for i in range(W - 1, len(vs))]
    resid = [vs[i] - roll[i - W + 1][1] for i in range(W - 1, len(vs))]
    se = (st.pstdev(resid) / math.sqrt(W)) if len(resid) > 1 else 0.0
    return roll, se


def _slope(series, lo, hi):
    p = [(a, b) for a, b in series if lo <= a < hi and not math.isnan(b)]
    if len(p) < 10:
        return None
    n = len(p); mx = sum(a for a, _ in p) / n; my = sum(b for _, b in p) / n
    num = sum((a - mx) * (b - my) for a, b in p); den = sum((a - mx) ** 2 for a, _ in p)
    return num / den * 1e6 if den else None


def check(run_dir: str, now: float | None = None) -> dict:
    sc = read_scalars(run_dir)
    spd = [(a, b) for a, b in sc.get("collect/ground_speed_mps", []) if not math.isnan(b)]
    if not spd:
        return {"verdict": "OK", "reasons": ["no scalars yet (seeding / pretrain burst)"],
                "run": os.path.basename(run_dir), "step": 0}
    step = spd[-1][0]
    out = {"run": os.path.basename(run_dir), "step": step, "reasons": [], "metrics": {}}
    crit, warn = [], []

    # ---- CRITICAL: TB stalled -------------------------------------------------------------
    ev = sorted(glob.glob(os.path.join(run_dir, "events.out.tfevents.*")))
    if ev:
        age = (now or time.time()) - os.path.getmtime(ev[-1])
        out["metrics"]["tb_age_s"] = round(age)
        if age > STALE_SECS:
            crit.append(f"STALLED: no TensorBoard write in {age/60:.0f} min (trainer hung or dead)")

    # ---- CRITICAL: episode-length canary --------------------------------------------------
    ep = [(a, b) for a, b in sc.get("collect/mean_episode_len", []) if not math.isnan(b) and a > WARMUP_STEPS]
    if ep:
        recent = [b for a, b in ep if a >= step - 300_000] or [ep[-1][1]]
        m = sum(recent) / len(recent)
        out["metrics"]["ep_len"] = round(m, 1)
        if m < EP_LEN_FLOOR:
            crit.append(f"CANARY: mean_episode_len {m:.0f} < {EP_LEN_FLOOR:.0f} -- hard_collapse is "
                        f"firing. Every healthy run in this project held exactly 500. Suspect a "
                        f"farmable (ungated, positive) reward term.")

    # ---- CRITICAL: value divergence -------------------------------------------------------
    pl = [b for a, b in sc.get("loss/pi_loss", []) if not math.isnan(b)]
    if len(pl) >= 3:
        out["metrics"]["pi_loss"] = round(pl[-1], 3)
        if all(abs(v) > PI_LOSS_DIVERGE for v in pl[-3:]):
            crit.append(f"DIVERGENCE: |pi_loss| {pl[-1]:.2f} > {PI_LOSS_DIVERGE} sustained")

    # ---- WARN: plateau / regression (judgement -- never auto-stop) -------------------------
    if step >= MIN_STEPS_TO_JUDGE:
        s = [(a, b) for a, b in spd if a > WARMUP_STEPS]
        roll, se = _roll(s, ROLL_STEPS)
        if roll:
            peak = max(roll, key=lambda p: p[1]); cur = roll[-1]
            z = (cur[1] - peak[1]) / se if se else 0.0
            out["metrics"].update({"speed_peak": round(peak[1], 4), "speed_peak_at": peak[0],
                                   "speed_now": round(cur[1], 4), "speed_z": round(z, 1)})
            if z <= Z_REGRESSION and peak[0] < step - ROLL_STEPS:
                warn.append(f"REGRESSION: ground_speed peaked {peak[1]:.3f} @ {peak[0]/1e6:.2f}M, "
                            f"now {cur[1]:.3f} (z {z:+.1f})")
        sl = _slope(s, step - 1e6, step)
        if sl is not None:
            out["metrics"]["speed_slope_last1M"] = round(sl, 4)
            if sl < PLATEAU_SLOPE:
                warn.append(f"PLATEAU: ground_speed slope {sl:+.4f} m/s per 1M over the last 1M "
                            f"(< {PLATEAU_SLOPE}) -- improvement has stopped")
        fl = [(a, b) for a, b in sc.get("collect/falls_per_min", []) if not math.isnan(b) and a > WARMUP_STEPS]
        if fl:
            fs = _slope(fl, step - 1e6, step)
            out["metrics"]["falls_per_min"] = round(fl[-1][1], 2)
            out["metrics"]["falls_slope_last1M"] = round(fs, 4) if fs is not None else None
            if fs is not None and fs > 0:
                warn.append(f"FALLS RISING: falls/min slope {fs:+.3f} per 1M over the last 1M")

    out["verdict"] = "CRITICAL" if crit else ("WARN" if warn else "OK")
    out["reasons"] = crit + warn or ["healthy"]
    out["action"] = ("STOP the run and notify" if crit else
                     "NOTIFY a human -- stopping is a value judgement, do not auto-stop" if warn
                     else "continue")
    return out


def newest_run():
    ds = [d for d in glob.glob(os.path.join(RUNS, "*")) if os.path.isdir(d)]
    return max(ds, key=os.path.getmtime) if ds else None


def selftest():
    """Replay runs whose outcome we already know. This is the only proof the thresholds work."""
    cases = [
        ("2026-08-15_13-40-31", "CRITICAL", "long-run take 1: ungated reward term, ep_len 444"),
        ("2026-08-15_17-19-04", "WARN", "long-run v2: peaked 4.65M then regressed"),
        ("2026-08-15_02-17-59", "OK", "h6-flat: healthy to budget"),
        ("2026-08-13_03-09-36", "OK", "lean-C: healthy to budget"),
    ]
    print(f"{'run':<22}{'expect':>10}{'got':>10}   note")
    print("-" * 92)
    ok = True
    for name, expect, note in cases:
        d = os.path.join(RUNS, name)
        if not os.path.isdir(d):
            print(f"{name:<22}{expect:>10}{'MISSING':>10}   {note}"); continue
        # freeze time so the staleness rule does not fire on finished runs
        ev = sorted(glob.glob(os.path.join(d, "events.out.tfevents.*")))
        r = check(d, now=os.path.getmtime(ev[-1]) + 60 if ev else None)
        got = r["verdict"]; ok &= (got == expect)
        print(f"{name:<22}{expect:>10}{got:>10}   {note}")
        for x in r["reasons"][:2]:
            print(f"{'':<42}   - {x[:78]}")
    print("\nSELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=None)
    p.add_argument("--json", action="store_true")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        return selftest()
    d = a.run or newest_run()
    if not d:
        print("no runs found"); return 0
    r = check(d)
    if a.json:
        print(json.dumps(r, indent=2))
    else:
        print(f"[{r['verdict']}] {r['run']} @ {r['step']/1e6:.2f}M")
        for x in r["reasons"]:
            print(f"   - {x}")
        if r.get("metrics"):
            print("   " + "  ".join(f"{k}={v}" for k, v in r["metrics"].items()))
        print(f"   ACTION: {r['action']}")
    return {"OK": 0, "WARN": 1, "CRITICAL": 2}[r["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
