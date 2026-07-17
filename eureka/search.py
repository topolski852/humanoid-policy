"""The outer loop: propose weights -> train (early-stop on plateau) -> grade ->
evolve -> repeat, until the search plateaus.

    # verify TB->fitness on an existing run, no training/API:
    OMNI_KIT_ACCEPT_EULA=YES .venv/bin/python -m eureka.search --dry-run

    # real search (each candidate is a bounded training run; runs for hours):
    OMNI_KIT_ACCEPT_EULA=YES .venv/bin/python -m eureka.search \
        --variant walk-biped --profile fast --max_iterations 250 \
        --iterations 8 --candidates 6 --patience 3 --backend subprocess \
        --log eureka_history.jsonl

Ported from humanoid-tuner/policy/reward_search/loop.py; the inner "optimize gains"
step is replaced by a real training run + TB-graded fitness. Per-candidate results
are printed as they finish (each is expensive and rare), not a fast progress bar.
"""

from __future__ import annotations

import glob
import json
import os
import time

import numpy as np

from . import config as C
from . import evaluate as E
from .propose import Candidate
from .propose_local import propose_weights_local
from .run import run_candidate
from .tb_utils import iters_since_best


def _fmt_hms(s: float) -> str:
    s = int(s)
    return (f"{s // 3600}h{(s % 3600) // 60:02d}m{s % 60:02d}s" if s >= 3600
            else f"{s // 60}m{s % 60:02d}s")


def _print_candidate(cand: Candidate, elapsed: float) -> None:
    c = cand.components
    if "_raw" in c:
        r = c["_raw"]
        print(f"  {cand.name:18s} fitness={cand.fitness:6.4f} "
              f"gate={c.get('walk_gate', 0.0):.2f} track={c.get('tracked_ratio', 0.0):.2f} "
              f"(v={r.get('tracked_speed', 0.0):.2f}/{r.get('commanded_speed', 0.0):.2f})  "
              f"fall={r['fall_rate']:.2f} err_xy={r['err_vel_xy']:.3f} "
              f"ep_len={r['mean_ep_len']:.0f}/{r['max_ep_steps']}  "
              f"[{cand.stopped_reason}, {_fmt_hms(elapsed)}]")
    else:
        print(f"  {cand.name:18s} fitness={cand.fitness:6.4f}  ({c.get('error','?')})")


def _log(fp, gen: int, cand: Candidate) -> None:
    fp.write(json.dumps({
        "gen": gen, "name": cand.name, "fitness": cand.fitness,
        "components": cand.components, "weights": cand.weights,
        "run_dir": cand.run_dir, "stopped_reason": cand.stopped_reason,
    }) + "\n")
    fp.flush()


def _write_best(path: str, best: Candidate) -> None:
    with open(path, "w") as f:
        json.dump({
            "name": best.name, "fitness": best.fitness, "components": best.components,
            "weights": best.weights, "run_dir": best.run_dir,
            "hydra_override": C.hydra_override_str(best.weights),
        }, f, indent=2)


def _evaluate(cand: Candidate, cfg: C.SearchConfig, gen: int, idx: int) -> Candidate:
    cfg.run_name = f"eureka-g{gen}-c{idx}"
    cand.name = f"g{gen}c{idx}:{cand.name}"
    t0 = time.perf_counter()
    run_dir, reason = run_candidate(cand.weights, cfg)
    cand.run_dir, cand.stopped_reason = run_dir, reason
    if run_dir is None:
        cand.fitness, cand.components = float("-inf"), {"error": "no run dir produced"}
    else:
        cand.fitness, cand.components = E.score(run_dir)
    _print_candidate(cand, time.perf_counter() - t0)
    return cand


def _dry_run(cfg: C.SearchConfig) -> None:
    """Score the newest existing run dir for the variant — verifies TB->fitness."""
    exp = os.path.join("logs", "rsl_rl", C.VARIANT_EXPERIMENT[cfg.variant])
    dirs = sorted(glob.glob(os.path.join(exp, "*/")), key=os.path.getmtime)
    if not dirs:
        print(f"[dry-run] no existing run dirs under {exp}/ to score.")
        return
    run_dir = dirs[-1].rstrip("/")
    fit, comp = E.score(run_dir)
    print(f"[dry-run] scored {run_dir}\n  fitness = {fit:.4f}")
    for k, v in comp.items():
        if k != "_raw":
            print(f"    {k:9s} {v:.4f}")
    print(f"  raw: {json.dumps(comp.get('_raw', {}))}")
    print("  default-weight hydra override:\n    " + C.hydra_override_str(C.DEFAULT_WEIGHTS))


def main() -> None:
    cfg = C.build_config()
    if cfg.dry_run:
        _dry_run(cfg)
        return

    rng = np.random.default_rng(cfg.seed)
    history: list[Candidate] = []
    log_fp = open(cfg.log, "a") if cfg.log else None
    t_start = time.perf_counter()

    print(f"eureka reward-weight search   variant={cfg.variant} profile={cfg.profile} "
          f"backend={cfg.backend}   {cfg.iterations} gens x {cfg.candidates} candidates "
          f"(inner cap {cfg.max_iterations} iters)")

    # Generation 0: either seed from a known-good walker (skip the retrain) or train the
    # shipped default weights + random explorers.
    seed = _seed_from_best(cfg) if cfg.seed_best else None
    if seed is not None:
        print(f"Generation 0: SEEDED from {seed.name} fitness={seed.fitness:.4f} "
              f"(re-graded from {os.path.basename(seed.run_dir)}, no retrain)")
        history.append(seed)
        if log_fp:
            _log(log_fp, 0, seed)
    else:
        print("Generation 0: default weights + random explorers")
        gen0 = [Candidate(name="default", weights=dict(C.DEFAULT_WEIGHTS))]
        gen0 += propose_weights_local([], cfg.candidates - 1, rng, gen=0)
        for i, cand in enumerate(gen0):
            _evaluate(cand, cfg, 0, i)
            history.append(cand)
            if log_fp:
                _log(log_fp, 0, cand)
    best = max(history, key=lambda c: c.fitness)
    _write_best(cfg.best_out, best)

    stopped = "reached max generations"
    for gen in range(1, cfg.iterations + 1):
        print(f"Generation {gen}: evolving from best fitness {best.fitness:.4f}")
        gen_best = best.fitness
        for i, cand in enumerate(propose_weights_local(history, cfg.candidates, rng, gen=gen)):
            _evaluate(cand, cfg, gen, i)
            history.append(cand)
            if log_fp:
                _log(log_fp, gen, cand)
        best = max(history, key=lambda c: c.fitness)
        _write_best(cfg.best_out, best)
        if best.fitness > gen_best + cfg.min_delta:
            print(f"  gen {gen}: new best fitness={best.fitness:.4f}  ({best.name})")
        # outer plateau: same detector, on the per-generation best fitness
        gen_bests = _gen_best_curve(history, cfg)
        stale, _ = iters_since_best(gen_bests, cfg.min_delta)
        if cfg.patience and stale >= cfg.patience:
            stopped = f"converged (no improvement for {cfg.patience} generations, gen {gen})"
            break

    if log_fp:
        log_fp.close()
    best = max(history, key=lambda c: c.fitness)
    _write_best(cfg.best_out, best)
    print(f"\n[stop] {stopped}   total {_fmt_hms(time.perf_counter() - t_start)}")
    print(f"BEST: {best.name}  fitness={best.fitness:.4f}")
    print(f"  reproduce with:\n    {C.hydra_override_str(best.weights)}")
    print(f"wrote {cfg.best_out}" + (f" and {cfg.log}" if cfg.log else ""))


def _seed_from_best(cfg: C.SearchConfig) -> Candidate | None:
    """Load the saved best (cfg.best_out) as a pre-scored gen-0 Candidate, RE-GRADED with
    the current fitness by re-reading its TB run dir — so the search can evolve from a
    known-good walker without retraining it. Returns None if unavailable/incomplete."""
    path = cfg.best_out
    if not os.path.exists(path):
        print(f"[seed] {path} not found — falling back to a fresh gen 0.")
        return None
    try:
        b = json.load(open(path))
    except Exception as e:
        print(f"[seed] could not read {path}: {e} — fresh gen 0.")
        return None
    run_dir, weights = b.get("run_dir"), b.get("weights")
    if not run_dir or not os.path.isdir(run_dir) or not weights:
        print(f"[seed] {path} missing run_dir/weights (run_dir={run_dir!r}) — fresh gen 0.")
        return None
    fit, comp = E.score(run_dir)                      # re-grade with the CURRENT fitness
    if fit == float("-inf"):
        print(f"[seed] no TB data in {run_dir} — fresh gen 0.")
        return None
    return Candidate(name=b.get("name", "seed"), weights=dict(weights), fitness=fit,
                     components=comp, run_dir=run_dir, stopped_reason="seeded")


def _gen_best_curve(history: list[Candidate], cfg: C.SearchConfig) -> list[float]:
    """Best fitness per generation so far (for the outer plateau detector)."""
    by_gen: dict[int, float] = {}
    for c in history:
        # name is "g<gen>c<idx>:..." — parse the generation
        try:
            g = int(c.name.split("c", 1)[0][1:])
        except Exception:
            g = 0
        by_gen[g] = max(by_gen.get(g, float("-inf")), c.fitness)
    running, out = float("-inf"), []
    for g in sorted(by_gen):
        running = max(running, by_gen[g])
        out.append(running)
    return out


if __name__ == "__main__":
    main()
