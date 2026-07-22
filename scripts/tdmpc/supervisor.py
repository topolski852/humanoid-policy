"""Autonomous TD-MPC2 walk-training supervisor (Layer 1 — the authoritative executor).

Runs the week-long train -> objectively-test -> adjust -> repeat loop with NO human or LLM in
the critical path. Per experiment it:

  1. pops the next spec from queue.jsonl (Claude may append more — see advisor_prompt.md),
  2. launches train.py warm-started from the best genuine walker so far,
  3. monitors TensorBoard against the EXPERIMENTS.md decision rules and early-stops on
     plateau / regression / divergence / can't-survive (or a Claude abort, or a hang),
  4. GRADES it with grade_run.py — the honest, reward-hack-proof fitness (a high-return
     faller scores ~0), preserving the checkpoint only if it genuinely walks better,
  5. prunes the run's intermediate checkpoints (disk safety for a week of runs),
  6. journals the outcome, then loops — until `wins >= MAX_WINS` (idle-at-cap) or the queue
     is exhausted.

State (supervisor_state.json) is persisted every iteration so the nohup wrapper
(run_supervisor.sh) can restart this process after a crash and pick up exactly where it left
off. TD-MPC2 has no mid-run resume, so a crashed *training* run is relaunched fresh,
warm-started from its own latest checkpoint.

Run from the repo root:  .venv/bin/python scripts/tdmpc/supervisor.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
from eureka.tb_utils import read_scalars, plateaued  # noqa: E402  (standalone: os/glob + lazy TB)

VENV_PY = os.path.join(REPO_ROOT, ".venv", "bin", "python")
TRAIN = os.path.join("scripts", "tdmpc", "train.py")
GRADE = os.path.join("scripts", "tdmpc", "grade_run.py")
EXPERIMENT_DIR = os.path.join(REPO_ROOT, "logs", "tdmpc", "tdmpc_biped")   # where train.py writes runs
PRESERVED = os.path.join(REPO_ROOT, "logs", "tdmpc", "_preserved")
RUNLOGS = os.path.join(REPO_ROOT, "logs", "tdmpc", "_runlogs")
STATE_DIR = os.path.join(REPO_ROOT, "logs", "tdmpc")

QUEUE = os.path.join(REPO_ROOT, "scripts", "tdmpc", "queue.jsonl")  # tracked; advisor appends here
STATE = os.path.join(STATE_DIR, "supervisor_state.json")
JOURNAL = os.path.join(STATE_DIR, "supervisor_journal.jsonl")
CONTROL = os.path.join(STATE_DIR, "control.json")
DONE_SENTINEL = os.path.join(STATE_DIR, "SUPERVISOR_DONE")     # wrapper stops restarting when present
OVERRIDES_DIR = os.path.join(STATE_DIR, "_overrides")

INITIAL_WARMSTART = os.path.join(PRESERVED, "stand_v4_best.pt")   # the proven stand seed

# --- defaults (CLI-overridable) ---
DEF_MAX_ENV_STEPS = 10_000_000     # official TD-MPC2 per-task budget
DEF_POLL_SECS = 120
DEF_MAX_WINS = 3                   # idle-at-cap: stop after this many genuine wins
DEF_MAX_RUNS = 40                  # hard safety cap on total runs
DEF_MIN_JUDGE_STEPS = 800_000      # don't apply plateau/regression rules before this (per EXPERIMENTS)
DEF_STALE_SECS = 2400              # TB hasn't advanced in this long -> treat run as hung
DEF_CRASH_RETRIES = 2             # relaunch a crashed training run this many times


# ------------------------------------------------------------------------------------------- utils
def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _log(msg: str) -> None:
    print(f"[supervisor {datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _read_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return default
    except Exception as e:
        _log(f"WARNING could not read {path}: {e}")
        return default


def _load_queue() -> list[dict]:
    """queue.jsonl: one experiment spec per line (comments/blank lines skipped)."""
    specs = []
    if not os.path.exists(QUEUE):
        return specs
    with open(QUEUE) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                specs.append(json.loads(line))
            except Exception as e:
                _log(f"WARNING queue.jsonl line {i+1} unparseable, skipped: {e}")
    return specs


def _series(sc: dict, tag: str) -> tuple[list[int], list[float]]:
    pts = sc.get(tag, [])
    return [s for s, _ in pts], [v for _, v in pts]


def _resolve_pi_loss_tag(sc: dict) -> str | None:
    for t in sc:
        if re.search(r"pi.*loss|loss.*pi", t, re.I):
            return t
    return None


# ------------------------------------------------------------------------- decision rules (EXPERIMENTS.md)
WALK_FLOOR_MPS = 0.15   # recent-mean ground speed below this = "not actually walking" (cmd is 0.3-0.8)


def _recent_mean(steps, vals, cur_step, window) -> float | None:
    r = [v for s, v in zip(steps, vals) if s >= cur_step - window]
    return sum(r) / len(r) if r else None


def evaluate_run(sc: dict, min_judge_steps: int, expects_locomotion: bool = False) -> tuple[str, str]:
    """Return ('continue'|'stop', reason) from the live TB scalars, per EXPERIMENTS.md:40-49.
    Divergence can fire early; the health/plateau/regression rules wait until min_judge_steps.

    `expects_locomotion` (True for the WALK env) switches the primary signal from episode length
    to GROUND SPEED + RETURN: the walk env is NON-episodic, so `mean_episode_len` is pinned at 500
    and is blind to a run that plateaus into a stable near-stand (return flat, speed ~0). For those
    runs the honest, un-fakeable signal is `collect/ground_speed_mps`. (Discovered live on the
    2026-07-22 walk run: ep_len=500 said 'HEALTHY' while speed sat at ~0.06 and return had flatlined.)
    """
    steps, ep = _series(sc, "collect/mean_episode_len")
    cur_step = steps[-1] if steps else 0

    # DIVERGENCE (both phases): value-overoptimism (pi_loss > 2.5), sustained over the last few points.
    pt = _resolve_pi_loss_tag(sc)
    if pt:
        _, pl = _series(sc, pt)
        if len(pl) >= 3 and all(v > 2.5 for v in pl[-3:]):
            return "stop", f"DIVERGENCE pi_loss>2.5 sustained (last={pl[-1]:.2f}) @ {cur_step}"

    if cur_step < min_judge_steps or len(ep) < 5:
        return "continue", f"warming ({cur_step} steps)"

    # --- LOCOMOTION phase (non-episodic walk): judge on ground speed + return, NOT ep_len ---
    if expects_locomotion:
        rsteps, ret = _series(sc, "collect/mean_episode_return")
        sp_steps, sp = _series(sc, "collect/ground_speed_mps")
        spd = _recent_mean(sp_steps, sp, cur_step, 200_000) if sp else None
        # PLATEAU_NONWALKING: return stopped improving AND it isn't actually walking -> stuck in a
        # non-locomotion optimum (the classic "upright but won't step"). Change the reward, don't wait.
        if ret and plateaued(ret, patience=300, min_delta=0.3) and spd is not None and spd < WALK_FLOOR_MPS:
            return "stop", (f"PLATEAU_NONWALKING return flat ~{ret[-1]:.2f} + speed {spd:.3f}"
                            f"<{WALK_FLOOR_MPS} (not walking) @ {cur_step}")
        # SPEED_REGRESSION: it DID sustain a walk (mean speed once cleared the floor over a window)
        # then lost it for >=300k steps -> gait collapsed back toward a stand.
        if sp:
            best_win = max((_recent_mean(sp_steps, sp, s, 200_000) or 0.0) for s in sp_steps)
            if best_win >= WALK_FLOOR_MPS and spd is not None and spd < 0.5 * best_win \
                    and cur_step >= min_judge_steps + 300_000:
                return "stop", (f"SPEED_REGRESSION mean speed {best_win:.3f}->{spd:.3f} "
                                f"(gait lost >=300k) @ {cur_step}")
        return "continue", (f"HEALTHY return={ret[-1] if ret else float('nan'):.2f} "
                            f"speed={spd if spd is not None else float('nan'):.3f} @ {cur_step}")

    # --- STAND phase (episodic): ep_len IS the signal ---
    # CAN'T SURVIVE: ep_len stuck < 50 for >= 1M steps -> bounds too tight.
    win_1m = [(s, v) for s, v in zip(steps, ep) if s >= cur_step - 1_000_000]
    if cur_step >= 1_000_000 and win_1m and all(v < 50 for _, v in win_1m):
        return "stop", f"CANT_SURVIVE ep_len<50 for last 1M (cur={ep[-1]:.0f}) @ {cur_step}"

    # REGRESSION: peaked then dropped >=25% and stayed down for >=300k steps.
    peak = max(ep)
    peak_step = steps[ep.index(peak)]
    recent = [v for s, v in zip(steps, ep) if s >= cur_step - 300_000]
    if peak > 50 and cur_step - peak_step >= 300_000 and recent and max(recent) < 0.75 * peak:
        return "stop", f"REGRESSION ep_len {peak:.0f}->{ep[-1]:.0f} (>=25% for 300k) @ {cur_step}"

    # PLATEAU mediocre: ep_len flat AND low (<300). ~1 pt / 1000 steps -> patience 500 ~= 500k steps.
    if plateaued(ep, patience=500, min_delta=10.0) and ep[-1] < 300:
        return "stop", f"PLATEAU_MEDIOCRE ep_len flat ~{ep[-1]:.0f}(<300) @ {cur_step}"

    return "continue", f"HEALTHY ep_len={ep[-1]:.0f} @ {cur_step}"


# --------------------------------------------------------------------------------- process control
def _terminate(proc: subprocess.Popen) -> None:
    """SIGINT (lets simulation_app.close() run) -> SIGTERM -> SIGKILL on the group. (from eureka/run.py)"""
    if proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
        for sig, wait in ((signal.SIGINT, 60), (signal.SIGTERM, 30), (signal.SIGKILL, 5)):
            os.killpg(pgid, sig)
            for _ in range(wait):
                if proc.poll() is not None:
                    return
                time.sleep(1)
    except ProcessLookupError:
        pass


def _find_run_dir(since: float) -> str | None:
    """The logs/tdmpc/tdmpc_biped/<ts>/ dir train.py just created (newest, mtime >= launch)."""
    cands = [d for d in glob.glob(os.path.join(EXPERIMENT_DIR, "*"))
             if os.path.isdir(d) and os.path.getmtime(d) >= since - 5]
    return sorted(cands, key=os.path.getmtime)[-1] if cands else None


def _latest_ckpt(run_dir: str) -> str | None:
    ck = glob.glob(os.path.join(run_dir, "model_*.pt"))
    if not ck:
        return None
    # numeric model_<N>.pt first (most-trained), else model_best.pt
    numbered = [(int(m.group(1)), p) for p in ck
                if (m := re.search(r"model_(\d+)\.pt$", os.path.basename(p)))]
    if numbered:
        return max(numbered)[1]
    return os.path.join(run_dir, "model_best.pt") if os.path.exists(os.path.join(run_dir, "model_best.pt")) else ck[0]


# ------------------------------------------------------------------------------- launch + monitor
def _resolve_warm_start(spec: dict, best_ckpt: str | None) -> str | None:
    ws = spec.get("warm_start", "best")
    if ws in (None, "none", "scratch"):
        return None
    if ws == "best":
        return best_ckpt or INITIAL_WARMSTART
    if ws in ("stand", "stand_v4"):
        return INITIAL_WARMSTART
    return ws if os.path.isabs(ws) else os.path.join(REPO_ROOT, ws)


def launch_and_monitor(spec: dict, cfg, best_ckpt: str | None,
                       warm_override: str | None = None) -> tuple[str | None, str, str]:
    """Launch one training run and monitor it. Returns (run_dir, reason, kind) where kind is
    'stopped' (a rule/abort/finish — grade it) or 'crash' (unexpected exit — caller may retry)."""
    os.makedirs(OVERRIDES_DIR, exist_ok=True)
    os.makedirs(RUNLOGS, exist_ok=True)
    name = spec.get("name", "exp")
    variant = spec.get("variant", "walk-biped-tdmpc")
    # the stand env is episodic (ep_len is the signal); every walk/curriculum variant is
    # non-episodic -> judge on ground speed + return instead (see evaluate_run).
    expects_locomotion = "stand" not in variant
    max_steps = int(spec.get("max_env_steps", cfg.max_env_steps))
    warm = warm_override if warm_override is not None else _resolve_warm_start(spec, best_ckpt)

    cmd = [VENV_PY, TRAIN, "--variant", variant, "--num_envs", str(spec.get("num_envs", 32)),
           "--tdmpc2_square", "--compile", "--updates_per_step", str(spec.get("updates_per_step", 16)),
           "--max_env_steps", str(max_steps), "--seed", str(spec.get("seed", 0)), "--headless"]
    if warm:
        cmd += ["--init_checkpoint", warm]
    overrides = spec.get("overrides") or {}
    if overrides:
        ovpath = os.path.join(OVERRIDES_DIR, f"{name}_{_now()}.json")
        with open(ovpath, "w") as f:
            json.dump(overrides, f, indent=2)
        cmd += ["--overrides", ovpath]
    cmd += list(spec.get("flags", []))     # e.g. ["--cmd_curriculum", "--cmd_survive_frac", "0.35"]

    logpath = os.path.join(RUNLOGS, f"{name}_{_now()}.log")
    _log(f"launch '{name}' variant={variant} warm={os.path.basename(warm) if warm else 'scratch'} "
         f"budget={max_steps} flags={spec.get('flags', [])} overrides={list(overrides)}")
    _log(f"  cmd: {' '.join(cmd)}")
    _log(f"  stdout -> {logpath}")
    env = dict(os.environ, OMNI_KIT_ACCEPT_EULA="YES")
    logf = open(logpath, "w")
    t0 = time.time()
    proc = subprocess.Popen(cmd, env=env, cwd=REPO_ROOT, stdout=logf, stderr=subprocess.STDOUT,
                            start_new_session=True)

    run_dir = None
    last_step, last_progress_t = -1, time.time()
    try:
        while True:
            rc = proc.poll()
            if rc is not None:
                # process exited on its own: natural finish (rc 0) or crash (rc != 0)
                if run_dir is None:
                    run_dir = _find_run_dir(t0)
                if rc == 0:
                    return run_dir, "training finished (budget reached)", "stopped"
                return run_dir, f"CRASH exit={rc}", "crash"

            if run_dir is None:
                run_dir = _find_run_dir(t0)
                time.sleep(cfg.poll_secs)
                continue

            # abort flag from the Claude advisor
            ctl = _read_json(CONTROL, {})
            if ctl.get("abort_current"):
                _terminate(proc)
                _clear_control()
                return run_dir, f"ABORT_BY_ADVISOR: {ctl.get('reason', '')}", "stopped"

            sc = read_scalars(run_dir)
            steps, _ = _series(sc, "collect/mean_episode_len")
            cur_step = steps[-1] if steps else 0
            if cur_step > last_step:
                last_step, last_progress_t = cur_step, time.time()
            elif time.time() - last_progress_t > cfg.stale_secs:
                _terminate(proc)
                return run_dir, f"HUNG no TB progress {cfg.stale_secs}s (stuck @ {cur_step})", "crash"

            action, reason = evaluate_run(sc, cfg.min_judge_steps, expects_locomotion)
            if action == "stop":
                _log(f"  decision: STOP — {reason}")
                _terminate(proc)
                return run_dir, reason, "stopped"

            time.sleep(cfg.poll_secs)
    finally:
        _terminate(proc)
        logf.close()


def run_experiment(spec: dict, cfg, best_ckpt: str | None) -> tuple[str | None, str]:
    """Run one experiment with bounded crash-retry (relaunch warm-started from its own latest
    checkpoint). Returns (run_dir, reason) for a graded stop."""
    warm_override = None
    for attempt in range(cfg.crash_retries + 1):
        run_dir, reason, kind = launch_and_monitor(spec, cfg, best_ckpt, warm_override)
        if kind == "stopped":
            return run_dir, reason
        # crash: relaunch from the latest checkpoint of the crashed run, if any
        _log(f"  {reason} (attempt {attempt+1}/{cfg.crash_retries+1})")
        latest = _latest_ckpt(run_dir) if run_dir else None
        if attempt < cfg.crash_retries:
            wait = 30 * (attempt + 1)
            _run_gpu_healthcheck()
            _log(f"  backoff {wait}s then relaunch"
                 + (f" warm-started from {os.path.basename(latest)}" if latest else " (no ckpt yet)"))
            time.sleep(wait)
            warm_override = latest if latest else warm_override
        else:
            return run_dir, reason + " (retries exhausted)"
    return run_dir, reason


def _run_gpu_healthcheck() -> None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
        _log(f"  nvidia-smi: {out.stdout.strip() or out.stderr.strip()}")
    except Exception as e:
        _log(f"  nvidia-smi unavailable: {e}")


# ------------------------------------------------------------------------------------ grade + preserve
def grade_run(run_dir: str, cfg) -> dict:
    """Run grade_run.py (fresh Isaac process) on the run's model_best.pt; return the grade dict."""
    best = os.path.join(run_dir, "model_best.pt")
    ckpt = best if os.path.exists(best) else (_latest_ckpt(run_dir) or run_dir)
    grade_out = os.path.join(run_dir, "grade.json")
    cmd = [VENV_PY, GRADE, "--checkpoint", ckpt, "--cmd_vx", str(cfg.cmd_vx),
           "--num_envs", str(cfg.eval_envs), "--steps", str(cfg.eval_steps), "--out", grade_out]
    env = dict(os.environ, OMNI_KIT_ACCEPT_EULA="YES")
    _log(f"  grading {os.path.basename(ckpt)} ...")
    try:
        subprocess.run(cmd, env=env, cwd=REPO_ROOT, check=True, timeout=cfg.grade_timeout)
        return _read_json(grade_out, {"fitness": 0.0, "is_win": False, "error": "no grade.json"})
    except Exception as e:
        _log(f"  WARNING grade failed: {e}")
        return {"fitness": 0.0, "is_win": False, "error": str(e)}


def preserve_if_better(run_dir: str, grade: dict, state: dict) -> bool:
    """Copy model_best.pt into _preserved/ when this run beats the running best fitness. Returns
    True if it became the new best warm-start."""
    fit = float(grade.get("fitness", 0.0))
    if fit <= state.get("best_fitness", -1.0):
        return False
    os.makedirs(PRESERVED, exist_ok=True)
    src = os.path.join(run_dir, "model_best.pt")
    if not os.path.exists(src):
        return False
    tag = "win" if grade.get("is_win") else "best"
    dst = os.path.join(PRESERVED, f"walk_{tag}_{fit:.3f}_{_now()}.pt")
    shutil.copy2(src, dst)
    state["best_fitness"] = fit
    state["best_ckpt"] = dst
    _log(f"  NEW BEST fitness={fit:.3f} -> preserved {os.path.basename(dst)}")
    return True


def prune_run(run_dir: str, grade: dict) -> None:
    """Disk safety: keep model_best.pt + the single latest numbered checkpoint; delete the rest.
    Only touches model_<N>.pt files INSIDE this graded run dir — never a whole run/experiment dir."""
    numbered = [(int(m.group(1)), p) for p in glob.glob(os.path.join(run_dir, "model_*.pt"))
                if (m := re.search(r"model_(\d+)\.pt$", os.path.basename(p)))]
    if not numbered:
        return
    keep = max(numbered)[1]                       # the most-trained numbered checkpoint
    removed = 0
    for _, p in numbered:
        if p != keep:
            try:
                os.remove(p)
                removed += 1
            except OSError:
                pass
    _log(f"  pruned {removed} intermediate checkpoints (kept model_best.pt + {os.path.basename(keep)})")


def journal(entry: dict) -> None:
    os.makedirs(os.path.dirname(JOURNAL), exist_ok=True)
    with open(JOURNAL, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _clear_control() -> None:
    try:
        os.remove(CONTROL)
    except FileNotFoundError:
        pass


# --------------------------------------------------------------------------------------------- state
def load_state() -> dict:
    return _read_json(STATE, {"next_index": 0, "runs": 0, "wins": 0,
                              "best_fitness": -1.0, "best_ckpt": None})


def save_state(state: dict) -> None:
    with open(STATE, "w") as f:
        json.dump(state, f, indent=2)


# ---------------------------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description="Autonomous TD-MPC2 walk-training supervisor.")
    p.add_argument("--max_env_steps", type=int, default=DEF_MAX_ENV_STEPS)
    p.add_argument("--poll_secs", type=int, default=DEF_POLL_SECS)
    p.add_argument("--max_wins", type=int, default=DEF_MAX_WINS, help="idle-at-cap after N genuine wins.")
    p.add_argument("--max_runs", type=int, default=DEF_MAX_RUNS, help="hard safety cap on total runs.")
    p.add_argument("--min_judge_steps", type=int, default=DEF_MIN_JUDGE_STEPS)
    p.add_argument("--stale_secs", type=int, default=DEF_STALE_SECS)
    p.add_argument("--crash_retries", type=int, default=DEF_CRASH_RETRIES)
    p.add_argument("--cmd_vx", type=float, default=0.3, help="grading forward command (m/s).")
    p.add_argument("--eval_envs", type=int, default=64)
    p.add_argument("--eval_steps", type=int, default=1000)
    p.add_argument("--grade_timeout", type=int, default=1800)
    p.add_argument("--queue_grace_secs", type=int, default=1800,
                   help="if the queue is exhausted (not at cap), wait this long for the advisor to "
                        "append specs before idling.")
    cfg = p.parse_args()

    os.makedirs(STATE_DIR, exist_ok=True)
    if os.path.exists(DONE_SENTINEL):
        _log(f"DONE sentinel present ({DONE_SENTINEL}) — nothing to do. Remove it to resume.")
        return 0

    state = load_state()
    save_state(state)   # write immediately so the advisor can read the true queue index during run 1
    _log(f"start: {json.dumps(state)}  cap: wins>={cfg.max_wins} or runs>={cfg.max_runs}")

    empty_since = None
    while True:
        if state["wins"] >= cfg.max_wins or state["runs"] >= cfg.max_runs:
            reason = f"wins {state['wins']}>={cfg.max_wins}" if state["wins"] >= cfg.max_wins \
                else f"runs {state['runs']}>={cfg.max_runs}"
            _log(f"IDLE-AT-CAP reached ({reason}). best_fitness={state['best_fitness']:.3f} "
                 f"best={state['best_ckpt']}. Writing DONE sentinel and idling.")
            with open(DONE_SENTINEL, "w") as f:
                json.dump({"reason": reason, "state": state, "when": _now()}, f, indent=2)
            return 0

        queue = _load_queue()
        if state["next_index"] >= len(queue):
            # queue exhausted but not at cap: give the advisor a grace window to append more.
            if empty_since is None:
                empty_since = time.time()
                _log(f"queue exhausted at index {state['next_index']} (len {len(queue)}); "
                     f"waiting up to {cfg.queue_grace_secs}s for the advisor to append specs.")
            if time.time() - empty_since > cfg.queue_grace_secs:
                _log("no new specs appended — writing DONE sentinel and idling.")
                with open(DONE_SENTINEL, "w") as f:
                    json.dump({"reason": "queue exhausted", "state": state, "when": _now()}, f, indent=2)
                return 0
            time.sleep(min(cfg.poll_secs, 120))
            continue
        empty_since = None

        spec = queue[state["next_index"]]
        _log(f"=== experiment #{state['runs']+1} (queue idx {state['next_index']}): "
             f"{spec.get('name', '?')} — {spec.get('notes', '')}")
        entry = {"when": _now(), "index": state["next_index"], "spec": spec}
        try:
            run_dir, reason = run_experiment(spec, cfg, state.get("best_ckpt"))
            entry["run_dir"] = run_dir
            entry["stop_reason"] = reason
            if run_dir and os.path.isdir(run_dir):
                grade = grade_run(run_dir, cfg)
                entry["grade"] = {k: v for k, v in grade.items() if k != "metrics"}
                entry["metrics"] = grade.get("metrics")
                became_best = preserve_if_better(run_dir, grade, state)
                entry["became_best"] = became_best
                if grade.get("is_win"):
                    state["wins"] += 1
                    entry["is_win"] = True
                    _log(f"  WIN #{state['wins']} — fitness={grade.get('fitness'):.3f}")
                prune_run(run_dir, grade)
            else:
                entry["grade"] = {"error": "no run dir produced"}
                _log("  WARNING no run dir — skipping grade")
        except Exception as e:
            entry["error"] = repr(e)
            _log(f"  ERROR in experiment (continuing): {e!r}")

        state["runs"] += 1
        state["next_index"] += 1
        journal(entry)
        save_state(state)
        _log(f"  state: runs={state['runs']} wins={state['wins']} "
             f"best_fitness={state['best_fitness']:.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
