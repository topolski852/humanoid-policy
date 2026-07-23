# TD-MPC2 walk — experiment journal

Curated record of reward/config changes → outcome → next step, so we can attribute what each
change did and plan the next one. Metric curves live in TensorBoard (`tensorboard --logdir
logs/tdmpc/tdmpc_biped` overlays all runs); each run also auto-writes `run_config.json` (exact
config + reward weights + git commit) into its run dir. Visuals: `scripts/tdmpc/eval_smoothness.py
--checkpoint <ckpt> --plan --num_envs 4 --cmd_vx 0.3` (GUI). Algorithm is TD-MPC2 throughout — only
the reward (task) and run flags change.

Key signals: **ground_speed_mps** (real locomotion; ~0.01 = standing/fallen, target ≥0.3), episode
return, `ep_len` (500 = non-episodic holding; <500 = collapse-resets firing), pi_loss (>1.5 =
value-overoptimism divergence).

| # | date | reward / key change | run flags | outcome | next |
|---|------|--------------------|-----------|---------|------|
| 1 | 07-19 | gated stand-gate + additive fall pen; episodic (45°+15cm term) | warm-start stand, UTD 4, sq | **crouch** — fwd 0.025, 0 falls, stable 20s; audit found episodic survival + UTD too low | make non-episodic, fix buffer, UTD↑ |
| 2 | 07-20 | + non-episodic (hard-collapse only), buffer terminal fix, seed burst | warm-start, UTD 16, sq, compile | **crouch, calmer** — binary standing gate (neg stand_height) = no gradient in sag | smooth gate + reset sags |
| 3 | 07-20 | + smooth standing gate (margin 0.12), collapse-reset −0.18 | warm-start, UTD 16, sq | **leans, feet planted, 3/4 fall** — no gait incentive; leaning fakes velocity | linear speed reward |
| 4 | 07-20 | linear-speed `move` (unfakeable) + non-episodic (deep collapse) | warm-start, UTD 16, sq, compile+TF32 | **more active (rocking 0.28) but fwd 0.0005** — wobbles in place, no stride; under-trained (1M) | run base full budget |
| 5 | 07-20 | official BASE: latent/mlp 512, horizon 3, **TD-M(PC)² OFF**, from scratch | UTD 32, compile | **stuck in crouch cold-start** — return flat ~−0.1 for 3M steps, fwd ~0.01, no divergence. Never gets upright from scratch (gate ~0 + flat gradient). Confirms warm-start was doing essential cold-start work. Stopped ~6.5M. | HYBRID reward + warm-start |
| 6 | 07-21 | **HYBRID** from scratch (latent512) | UTD 16, compile, no sq | **REWARD-HACKED** — return climbed −36→−6.5 then plateaued; every robot learned to FALL, lift a leg in the air, and rock. `feet_air_time` rewards an airborne foot → a fallen robot games it by waving a leg. Also the stability penalties formed a barrier suppressing the motion needed to stand. Not walking (speed ~0.02). Stopped 2M. | STAND FIRST (no gameable terms) |
| 7 | 07-21 | **STAND v1** — clean stand reward, cmd=0, calm spawn, NON-episodic, heavy penalty stack, no sq | from scratch, latent512, UTD 16, compile | **still slump −18** — pi_loss pinned 1.6 (overoptimism); 3/4 fall, last one hops one-footed. Deviated from the recipe that stood before. | revert to proven recipe |
| 8 | 07-21 | **STAND v2** — EPISODIC (fall→reset) + minimal reward (gate+upright+term_pen+action_rate+dof) + **TD-M(PC)²** | from scratch, latent512, UTD 16, compile | **stands but MEDIOCRE plateau** — return +ve, ep_len ~18→~400, but per-step reward pinned ~0.055 (return≡ep_len perfect match = frozen static posture). Falls when FEET COLLAPSE TOGETHER (narrow base). Neutral foot-sep=0.17 m. | reward WIDER stance |
| 9 | 07-21 | **STAND v3** — + `feet_stance_width` banded [0.23,0.30] (w0.3) | from scratch, latent512, UTD 16, compile, sq | **REGRESSED** — stance widened to ~0.23 (reward worked) but the extra term destabilized the value (pi_loss→−3.2); peaked ep_len 407 @160k then declined to ~220. Worse than v2. Wider base didn't help (maybe dynamically harder w/ actuator latency). | revert stance, tighten termination |
| 10 | 07-21 | **STAND v4** — revert stance; TIGHTEN termination: tilt 45°→30°, crouch 15→12 cm | from scratch, latent512, UTD 16, compile, sq | **SUCCESS (best stand)** — ep_len ~470/500 climbed+held; eval in STAND env = 0 falls/16, calm (rocking 0.084). Solid+calm FROM A CALM START. But NOT robust to harder/randomized spawns (flails in the walk env). Preserved stand_v4_best. | accept → WALK |
| 11 | 07-22 | **WALK phase** — warm-start stand_v4; lean reward (gated core + upright + **feet_air_time UPRIGHTNESS-GATED** + feet_slide + action_rate + dof); non-episodic; dropped heavy stability stack (barbers the gait) | warm-start stand_v4, latent512, UTD 16, compile, sq | pending | — |
| 12 | 07-22 | **WALK v5 / reward-C-pushmove** — `move_weight` 0.75→0.9 (force stepping) | warm stand_v4 (auto) | **won't-move plateau** — ep_len 500, return climbed to 5.3 but fwd speed stuck 0.05–0.10 (<0.15 floor); graded fitness **0.092** (honest fwd 0.012). Survive-in-place farms posture/survival; the weight bump did NOT break the equilibrium. Stopped PLATEAU_NONWALKING @1.47M. | curriculum (force motion via ramp), NOT reward-weight tweaks |
| 13 | 07-22 | **curriculum-A** — survival-gated command ramp (cmd_scale 0.10→full) | warm best(0.092) | **ROBBED by supervisor race (see BUG)** — ran only **26k steps** yet at 16–17k it was the **ONLY run to actually MOVE (speed spiked to 0.39!)** before being killed. Supervisor read the PREVIOUS run's stale plateaued scalars, killed it, and re-graded reward-C (duplicate journal entry, bogus fitness 0.092). The curriculum spine never got a fair run. | re-run curriculum outside the race window |
| 14 | 07-22 | **reward-A-stride** — widen `feet_air_time` band → [0.22, 0.45] | warm best(0.092) | **won't-move plateau (repeat of #12)** — ep_len ~484, return 5.5 plateaued, fwd 0.067 (<0.15). Widening the stride band has no mechanism to *initiate* stepping from a stand. But graded **0.191** under MPPI eval (walk_gate 0.208) → new best_ckpt. Confirms reward-tweaks-from-stand ≠ walking. | curriculum lever, not more reward tweaks |
| 15 | 07-22 | **curriculum-B-slow** — stricter ramp (survive_frac 0.45, ramp_interval 80k) | warm best(0.191) | **ROBBED by the race too** (2nd curriculum run lost) — journaled with reward-A-stride's run_dir + grade (fitness 0.191 @1018880). Confirmed the race hits *every* curriculum run (follows a plateaued reward run). Prompted the operator fix (`_discover_run_dir`) + min_judge_steps→2M + curriculum re-queued to run next. | run curriculum-A2-retry with race fixed |
| 16 | 07-22 | **curriculum-A2-retry** (LIVE) — faithful curriculum-A retry, race fixed | warm **stand_v4** | **PROMISING, still ramping @280k** — race fix holds (own dir 19-27-25); `cmd_scale` ramped 0.10→**0.40**; speed peaks **0.46 m/s** but mean still ~0.05 (below 0.15 floor), ep_len 500, pi_loss −0.13 (healthy). First fair run of the curriculum spine — the one direction that produced real forward motion. Advisor 20:xx: healthy+early, no action. | let it run to ≥2M; watch mean speed clear 0.15 floor |

### SUPERVISOR BUG — `_find_run_dir` robs runs launched near the previous grade (found 2026-07-22)
`_find_run_dir(since)` (supervisor.py:208) picks the newest `tdmpc_biped/*` dir with **directory
mtime ≥ launch−5s**. But grading the *previous* run writes `grade.json`/`eval_metrics.json` INTO the
previous run's dir, bumping its mtime. If a new run launches while that write is within ~5 s of its
t0 (the previous grade's Isaac eval overlaps the new launch), `_find_run_dir` returns the PREVIOUS
dir during the new run's ~60 s Isaac boot → the supervisor reads the previous run's (plateaued)
scalars → `evaluate_run` returns STOP → kills the new run at ~26 k steps → then **re-grades the
previous checkpoint** (producing a duplicate journal entry). This robbed **curriculum-A** (idx 1):
killed at 26 k, journaled with reward-C's run_dir + stop_reason `@1472512` + fitness 0.092.
reward-A-stride survived only because its launch fell minutes after the prior grade (old mtime
outside the −5 s window). Intermittent + timing-dependent → the CURRICULUM runs (which follow long
plateaued reward runs whose grade overlaps the next launch) are the ones most often robbed.
**Fix (operator — outside the advisor's allowed edits):** resolve run_dir from train.py's own
`[tdmpc] logging to <dir>` stdout line (or a PID/sentinel file) and grade THAT dir, and/or exclude
dirs that already contain `grade.json`. Until fixed, curriculum experiments may be silently skipped.

**RESOLVED 2026-07-22:** replaced `_find_run_dir` with `_discover_run_dir` — primary = parse the
`[tdmpc] logging to <dir>` stdout line (exact); fallback = newest dir NOT in a snapshot taken
immediately before launch. Both make a stale/previous dir unreturnable regardless of its mtime.
Verified live: curriculum-A2-retry (idx4) latched its own dir `19-27-25`. Also this run: raised
`min_judge_steps` 800k→2M (runs are under-saturated ~1M; give every idea a real exploration window
before any plateau call) and re-queued the curriculum spine to run NEXT (it was the only direction
to produce real forward motion, 0.39 m/s).

### CURRICULUM GATE FIX (operator, 2026-07-22, commit 4b4cacb)
The command curriculum widened `cmd_scale` when `mean_ep_len > frac*max` — meaningless in the
non-episodic walk env where ep_len is pinned at 500, so the command ran all the way to full while
the robot stood still (real speed ~0.05 at cmd_scale 1.0; see curriculum-A2-retry). Now the ramp
gates on ACHIEVED TRACKING: widen only once body velocity projected onto the command direction
reaches `cmd_track_frac` (0.5) of the current commanded speed, over the moving-commanded envs
(un-fakeable — ~0 for a rocker, correct sign for backward commands). Also removed the
`len_hist.clear()` that produced the ep_len→0 dips at each ramp. Applies to the next curriculum run
(curriculum-C-frombest); the live curriculum-A2-retry finishes on the old logic (already at full).

Note: v5 (warm-start v4 + smoothness) was launched then ABORTED — its premise (calm the jitter)
was based on a WALK-env eval artifact; in the stand env v4 is already calm. Skipped to WALK.
Eval env matters: always pass `--task Walk-Humanoid-Policy-Biped-Tdmpc-Stand-v0` to watch a STAND.

### feet_air_time is GAMEABLE (run 6) — critical
`feet_air_time_positive_biped` rewards the swing foot being airborne; it is NOT gated by
uprightness, so a robot lying on its back with a leg up farms it. Fix for the WALK phase: gate it by
the uprightness (multiply by stand_gate / only reward when upright), or only credit air-time when
base height is near standing. For the STAND phase we simply drop it (nothing to farm by falling).

## OVERNIGHT AUTONOMOUS RUN (2026-07-21 ~24:00 → 2026-07-22 noon) — protocol
User granted autonomy to start/stop runs overnight. Goal: nail the STAND, then start WALK.
Check ~every 0.8–1M steps. **Decision rule per check** (read full trajectory each time):
- HEALTHY (ep_len ↑ or holding high, return ↑, pi_loss <2.5 stable, per-step reward ≥ v2's 0.055):
  → CONTINUE.
- SOLID STAND (ep_len ≥~450 sustained + per-step reward clearly > 0.055 + low speed): → preserve
  ckpt, log, then start a WALK experiment (warm-start from the stand; feet_air_time uprightness-gated;
  non-episodic; linear-speed reward).
- REGRESSION (ep_len peaks then drops ≥25% for ≥300k) or DIVERGENCE (pi_loss >2.5): → STOP, change,
  log, relaunch.
- PLATEAU mediocre (ep_len flat <300, per-step ~0.055) : → STOP, tighten toward STILLNESS/upright, log.
- CAN'T SURVIVE (ep_len <50 for ≥1M): bounds too tight → loosen (30°→35°), log.
**Experiment queue (pick by failure mode):**
1. v4 (running): tight termination 30° / 12 cm.
2. STILLNESS (user's "shouldn't move almost at all"): tighten gated move_stand margin 0.5→~0.2 so
   standing requires near-zero velocity; and/or tilt 30°→25°, height 12→10 cm.
3. If regress/diverge: reduce value inflation — UTD 16→8 and/or trim reward magnitude.
4. If solid stand: WALK phase (feet_air_time gated by uprightness — the run-6 hack fix).
Every change is committed + logged in the table below with its outcome.

## Findings (why each reward change was made)
- **Cold-start problem (run 5):** from scratch the multiplicative `standing×upright` gate is ~0 with a near-flat gradient when not upright → no signal to stand up → crouch. Fix: small *ungated* `upright_posture` term (smooth gradient to vertical) + keep the stand warm-start.
- **No gait incentive (runs 3–4):** pure velocity reward specifies the goal, not the stride. Leaning/wobbling fakes some velocity. Fix: `feet_air_time` (reward foot-lift) + `feet_slide` (punish drag) — the proven PPO terms.
- **Stability / IMU (run 6 rationale):** the deployed PPO walk worked because it prioritized stability; we'd stripped `ang_vel_xy`/`flat_orientation`/`base_accel`/smoothness. Re-added (modest) for low IMU noise + smooth sim-to-real. Caution: keep modest so they don't suppress the gait into standing.
- **Sample budget:** official TD-MPC2 default is 10M steps/task (not the 2M I'd mis-cited); PPO needed 590M. 1M was ~10% — under-trained.

## Run 8/9 — STAND v2/v3
- **v2** (episodic + minimal reward + TD-M(PC)²): WORKED far better than v1 — return positive, ep_len
  climbed ~18→~400. But PLATEAUED at a *mediocre* stand: per-step reward pinned ~0.055 (return and
  ep_len a perfect scalar match = constant per-step = frozen static posture, no active balancing /
  no posture improvement). Visual: some stand, others fall when the FEET COLLAPSE TOGETHER (narrow
  base). Neutral foot sep measured = 0.17 m; successful ones abduct wider.
- **v3** (this run): + `feet_stance_width` reward (w0.3, target 0.25 m > neutral 0.17) to give a
  gradient toward a WIDE base of support. Added `collect/stance_width_m` telemetry. Watch: stance
  climbs toward 0.25 AND ep_len→500 with per-step reward RISING (curves decoupling = real posture
  improvement, not just surviving longer). Tunables if needed: stance target/weight, tilt-termination
  angle (45°→25° to force taller), upright_bonus weight.

## Run 7 — STAND phase (crawl→stand→walk, step 1)
Decision (user): get a solid upright STAND first, then add walking. Clean stand reward
(`StandRewardsCfg`): gated core w1.0 (cmd=0 → rewards upright+still) + upright_bonus w0.4 + the
stability penalties (ang_vel/flat_orientation/lin_vel_z/base_accel/action_rate — for standing these
correctly mean "hold still & level") + safety (dof_pos_limits, undesired_contacts). NO feet terms
(ungameable). Task `walk-biped-tdmpc-stand` (calm StandEventsCfg spawn). From scratch, latent 512.
Milestone: return climbs to +hundreds (upright hold), 0 falls in eval. Then warm-start the WALK
phase from this checkpoint (and gate feet_air_time by uprightness before re-adding it).

## Run 6 — HYBRID reward (commit: see run_config.json)
Reward = gated linear-speed core (w1.0) + ungated upright bonus (w0.15) + feet_air_time (w1.0) +
feet_slide (w−0.07) + ang_vel_xy (w−0.05) + flat_orientation (w−0.5) + lin_vel_z (w−0.1) +
base_accel_xy (w−0.02) + action_rate (w−0.01) + dof_pos_limits (w−0.1) + undesired_contacts (w−0.3).
Warm-start from `_preserved/stand_phase1_550176_verified.pt` (clears the cold-start dead zone).
Watch: does ground_speed climb toward the command (stepping) while staying upright/smooth. Stability
weights are intentionally modest for run 6 — if it walks but is jerky, raise ang_vel/base_accel/
action_rate; if it stands but won't step, raise feet_air_time / lower stability.
