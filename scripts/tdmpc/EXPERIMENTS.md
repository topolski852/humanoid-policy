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
| 16 | 07-22 | **curriculum-A2-retry** — faithful curriculum-A retry, race fixed | warm **stand_v4** | **FAILED non-walking @2.09M** — race fix HELD (own dir 19-27-25, own grade). But `cmd_scale` ramped 0.10→**1.0 (FULL)** while ground_speed mean stayed **0.107 (<0.15)**; graded fitness **0.147**, honest fwd 0.019. First FAIR curriculum run → proves the survival-gated ramp reaches full command on SURVIVAL, not locomotion. **CURRICULUM_GATE alert filed (idx4)** — the ramp is structurally broken for the non-episodic walk env. | STOP feeding curriculum until trainer ramp is speed-gated; use reward levers |
| 17 | 07-23 | **curriculum-C-frombest** — 2nd curriculum shot, survive_frac 0.40 | warm best(0.191) | **REWARD-HACKED (worst run) — flag confirmed** — return climbed to ~18.9 (~4× the ~5 of runs 12–16) while `cmd_scale` stuck 0.10 and honest speed collapsed: graded fitness **0.003**, walk_gate 0.004, fwd **0.0004**. The advisor's watch-flag was right: inflated return, zero locomotion. Warm-starting a curriculum from a non-walking base farms a gameable reward term. cmd_scale never hit 0.5 so NOT a countable CURRICULUM_GATE hit — it's a reward-hacking failure. | curriculum from a stand is counterproductive → reward levers only |
| 18 | 07-23 | **reward-B-noslip** — `feet_slide` −0.12 + `upright_bonus` 0.3→0.4 | warm best(0.191) | **NEW BEST 0.325** (was 0.191) — walk_gate 0.35 (highest yet), honest fwd 0.042 (still <0.25 win bar, not truly walking yet, but the honest grade moved for the FIRST time via a reward lever). Punishing foot-drag + holding more upright is the productive direction. Preserved walk_best_0.325. | push this shape: keep no-slip+upright, ADD a move/stride lever |
| 19 | 07-23 | **reward-D-smooth** — `action_rate_l2` −0.03 (jitter), DEFAULT feet_slide/upright | warm best(0.325) | **collapsed 0.036** (fwd 0.005) — the extra jerk penalty (on the plain default reward, not idx6's shape) broke the stand back to non-walking. | jitter tweak alone is harmful; keep idx6 shape |
| 20 | 07-23 | **reward-E-moveslip** (advisor append) — idx6 shape + `move_weight` 0.75→**0.9** | warm best(0.325) | **TOTAL FAILURE 0.0** (walk_gate 0.0, fwd **−0.0007**) — move_weight 0.9 is POISON even with slip punished (also killed idx0 at 0.09). The hypothesis "no-slip converts the move push into stepping" is FALSE. move_weight bumps are now retired. | never bump move_weight; small single-lever steps around idx6 |
| 21 | 07-23 | **reward-F-stride** (advisor append) — idx6 shape + `feet_air_time` band [0.22,0.45] | warm best(0.325) | **regressed 0.167** (fwd 0.021) — widening the strike band from the best base hurt. 3rd straight idx6-perturbation to regress (idx7/8/9) → idx6 looked like a fragile lucky basin. | try the CADENCE lever instead of the band |
| 22 | 07-23 | **reward-G-airweight** (advisor append) — idx6 shape + `feet_air_time.weight` 1.5→**2.5** | warm best(0.325) | **MIRAGE — graded 0.214** (fwd 0.028), BELOW idx6. Live gs 0.149 did NOT hold under deterministic eval: training-time ground_speed counts rocking/shuffle magnitude; eval measures net displacement (~0.03). **Lesson: collect/ground_speed is NOT a reliable predictor of graded fwd.** | trust the GRADE, not live gs |
| 23 | 07-23 | **reward-I-upright5** (advisor append) — feet_slide −0.12 + `upright` 0.4→0.5 | warm best(0.325) | **graded 0.264** (fwd 0.034), below idx6 → upright peaks at 0.4 (0.3→0.4 helped, 0.4→0.5 overshot). 2nd-best overall but still < idx6. | upright optimized at 0.4 |
| 24 | 07-24 | **reward-J-airweight3** (LIVE, advisor append) — idx6 shape + `feet_air_time.weight` **3.0** | warm best(0.325) | @1.58M: same gs-0.15 mirage, healthy (pi_loss −0.65). Follows the idx10 mirage → expected to grade ~0.21 < idx6. Let it finish (not worth abort). | — |
| — | 07-24 | **advisor wake6 — REWARD SEARCH CONVERGED.** idx6 (0.325) is a hard local optimum; 6 perturbations all tie/regress; EVERY run's honest fwd is 0.02–0.04 (6× short of 0.25 win bar). Fitness spread = posture/gate, not locomotion. **Filed OPERATOR alert** (ADVISOR_ALERTS.md) to prioritize the idx4 CURRICULUM_GATE trainer fix — the only remaining path to real walking. Appended **idx13 reward-K-tracksharp** (`tracking_std` 0.25→0.15) as the FINAL untried reward-param probe. | — | reward tuning cannot bridge 6×; curriculum fix is the unblock | — |

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

## STABILITY broad phase A–D (graded 2026-07-24 → 07-25, 2M each) — TRACKING COLLAPSE
| run | track_lin_vel wt | extra lever | fitness | track_err_lin | rocking_rms | fwd/cmd speed |
|-----|------|------|------|------|------|------|
| A-track | 1.0 | none | 0.0002 | 0.279 | 0.839 | 0.062 / 0.453 |
| B-gyro | 1.0 | ang_vel_xy -0.05 | 0.0001 | 0.292 | 0.821 | 0.049 / 0.453 |
| C-level | 1.0 | flat_orient -0.3 | 0.0001 | 0.344 | 0.949 | 0.031 / 0.453 |
| D-novert | 1.0 | lin_vel_z -0.1 | 0.0001 | 0.284 | 0.865 | 0.043 / 0.453 |

**Finding:** all 4 collapse the warm-started gait. TB shows every run starts at ground_speed ~0.48 m/s
and decays to ~0.09–0.20 by end while training return *rises* (8→15). At track_lin_vel weight 1.0 the
reward optimum is to stand/shuffle (upright/stand terms beat tracking), so the robot barely translates
(~0.05 m/s vs 0.45 commanded) and rocks hard (rocking_rms ~0.85). Quietness levers (B–F) are moot until
the base omni gait tracks. Not a grader bug (grades match TB, raw metrics vary per run; pi_loss ~-0.25).
**Response:** appended G/H (track 4x/6x, 5M), I (4x+level), J (4x, 10M) to restore tracking dominance.
Expect E-smooth/F-fullstack (already queued ahead) to also collapse — extra penalties on weak tracking.

## STABILITY wake 3 (2026-07-25): E/F + G-track4x confirm STANDING-ATTRACTOR collapse
- E-smooth / F-fullstack (2M): fitness 0.0003, forward_speed 0.05–0.06 vs 0.45 cmd — collapsed like A–D.
- G-track4x (live, 5M, track lin 4.0/ang 2.0): STILL collapses. TB starts 0.526 m/s → plateaus ~0.14;
  training return *rises* to 32.5 while speed falls. So 4× tracking authority did NOT hold the gait.
- **Root cause (from reward config read):** stand_walk(w1.0, move_weight 0.75 → 0.25 baseline) + upright_bonus(0.3)
  reliably pay every step for standing upright at height; feet_air_time(1.5) + track_lin_vel only pay for a
  *proper omni gait* the forward-only warm-start can't produce under varied_commands → no gradient to move,
  so standing upright is the safe optimum. Boosting tracking weight just raises return without recovering speed.
- **Response:** appended K/L/M — remove the standing payoff (upright_bonus→0, stand_walk move_weight→0.95/0.9),
  keep 4× tracking, M also boosts feet_air_time→2.5. If standing stops paying, moving-and-tracking is the only
  way to score. H(6×)/I(4×+level)/J(4×,10M) still queued ahead as brackets on tracking-weight & steps (expected
  to also collapse — will confirm the lever is standing, not tracking authority). No machinery issue (grades match
  TB, raw metrics vary per run, pi_loss ~-0.2, disk 14%).

## STABILITY wake 4 (2026-07-25): G-track4x FINAL + H confirm — tracking-weight/steps are the WRONG lever
- G-track4x FINAL (5M): fitness 0.0001, forward_speed 0.039, track_err_lin 0.372 — WORSE than the 2M runs.
  More steps of the same reward drive it DEEPER into the standing basin (5M worse than 2M). Steps don't help.
- H-track6x (live): gs 0.494 -> 0.188 monotonic decay, return 49.5. 6× tracking collapses too, just slower.
- Finding now 7-strong (A–F @2M + G @5M) + H: the collapse is the STANDING ATTRACTOR, independent of
  tracking weight (1×/4×/6×) and steps (2M/5M). NOT a grader bug — grades track the collapse, raw metrics
  vary per run, no duplicate grades, pi_loss ~-0.2, no divergence.
- **Response:** queued N/O = anti-standing configs (K/M overrides) at 10M, for the "learn omni from scratch
  with the right reward" test. Plan: abort J (idx9, 4×track @10M — confirmed-doomed) when it becomes current
  to save ~10h GPU for the K/L/M anti-standing test. K's ground_speed trajectory is the decisive read next.

## STABILITY wake 5 (2026-07-26): H final + command-config recon (no new runs of interest)
- H-track6x FINAL (idx7): fitness 0.0001 — as predicted. 8 collapses now (A–H), all <=0.0003.
- Now running I-track4x (idx8, 4×+level, zero info); J (idx9, 4×@10M) next; K-nostand (idx10) is the real test.
- **Command-config recon:** `--varied_commands` uses env range lin_vel_x=(-0.8,0.8) (incl. BACKWARD) + lateral
  + yaw, rel_standing_envs=0.02. So the forward-only warm-start must learn back/lateral/turn from scratch.
  rel_standing is only 2% -> NOT the driver; the standing attractor is REWARD-side (stand_walk baseline +
  upright_bonus under nonzero commands), which K/L/M already target. `_apply_overrides` runs BEFORE the
  varied_commands block and that block only sets heading_command=False, so `commands.base_velocity.ranges.*`
  and `.rel_standing_envs` ARE overridable -> a forward->omni CURRICULUM is available as a fallback if the
  reward-side anti-standing fix is insufficient. Did NOT queue a rel_standing spec (2% is negligible).
- **Plan:** abort J (10M waste) when it becomes current next wake -> launches K. Watch K's ground_speed.

## STABILITY wake 6 (2026-07-26): I graded, J ABORTED -> launching K (anti-standing test)
- I-track4x FINAL (idx8): fitness 0.0, forward 0.033 — collapsed (worst variant: 4×track + level penalty).
- 9 runs now (A-I) all collapse to standing: tracking weight (1×/4×/6×) and steps (2M/5M) are the WRONG levers.
- J-track4x (idx9, 4×@10M): live TB confirmed same collapse (0.526->0.155 plateaued) at only ~1M/10M steps.
  ABORTED via control_stability.json to redirect ~10h GPU to the anti-standing test. Abort consumed cleanly
  (supervisor killed J, cleared control, grading partial then advancing to K idx10).
- **K-nostand (idx10) is the pivot:** upright_bonus->0, stand_walk move_weight 0.75->0.95, 4× tracking, 5M.
  First run where standing stops paying. Watch its ground_speed next wake — if it holds the warm-start gait
  (~0.4-0.5 m/s) instead of decaying to ~0.15, the collapse is broken and we start narrowing to the win bar.
  If it also freezes, the forward->omni command curriculum (ranges ARE overridable) is the ready fallback.
- Queue after K launches: pending 5 (K,L,M,N,O), above floor. Machinery healthy (abort worked, grades track TB).

## STABILITY wake 7 (2026-07-26): ROOT CAUSE FOUND — omni collapse is a command-curriculum WIRING gap
- K/L-nostand (idx10/11) COLLAPSED too (K: fitness 0.0001, forward 0.048, TB 0.498->~0.12). Removing the
  reward-side standing terms (upright_bonus->0, move_weight->0.95) did NOT break the collapse. Anti-standing
  hypothesis DISPROVEN.
- **Root cause:** the TRACKING REWARD itself rewards standing over wrong-direction motion for any command the
  forward-only warm-start can't execute (standing err=|cmd| < wrong-dir err). Under full omni cmds
  (lin_vel_x=(-0.8,0.8) incl. backward + lateral + yaw) the min-error policy is to FREEZE. The remedy — the
  tracking-gated command curriculum (fixed in commit 4b4cacb) — is OFF: config.py cmd_curriculum=False,
  train.py has --cmd_curriculum but bootstrap.py (the stability entry) does NOT wire it, and --overrides only
  patches env_cfg not agent_cfg. No [curriculum] line in any stab log. 11 runs corroborate.
- **Actions:** (1) ESCALATED to ADVISOR_ALERTS.md (OTHER): wire --cmd_curriculum into bootstrap.py (mirror
  train.py:184-189). (2) Queued P-S width-sweep (idx15-18): 4x-tracking (same reward that collapsed on omni),
  command range narrowed via env_cfg overrides — P forward-only, Q +lateral, R +backward, S near-full. Isolates
  which command dimensions the warm-start can hold. READ VIA TB ground_speed (graded on full omni -> low
  fitness by construction). (3) ABORTED the dead anti-standing tail L/M/N/O (idx11-14, ~28M steps incl. two
  10M runs) to launch P now. P-cur-fwd running.
- **Next wake read:** if P HOLDS the gait (TB ground_speed ~0.4) while G/K collapsed on the same reward, the
  command-width mechanism is PROVEN and the curriculum is the confirmed fix. Q/R/S then locate the threshold.

## STABILITY wake 8 (2026-07-26): P width-sweep supports command-width mechanism; curriculum wiring still pending
- P-cur-fwd (idx15, FORWARD-ONLY range vx 0-0.5): grade 0.0002 BUT that's a full-omni eval (grader always uses
  commanded_speed~0.45, incl. back/lat/yaw P never trained) so the scalar is uninformative for P. Real signals:
  TB ground_speed settles ~0.17-0.24 ~= P's mean forward command (~0.25), and return=74 >> omni runs' 15-49 on
  the SAME 4x tracking reward. A standing policy would bank the same ~15-49 as the omni runs; 74 means P is
  EARNING the tracking reward -> forward tracking WORKS. Supports: collapse is command-WIDTH (omni dirs the
  fwd-only warm-start can't do), not the online refine per se.
- Q-cur-lat (idx16, +lateral) running, trending worse (mean_last10 ~0.09) -> adding lateral degrades, threshold
  effect as expected.
- Engineer has NOT yet wired --cmd_curriculum into bootstrap (grep=0); escalation from wake 7 still pending.
- **Action:** appended T (mild widen: vx0-0.6, lat/yaw +-0.2, 5M) and U (moderate: vx -0.4..0.6, lat/yaw +-0.3,
  5M), both warm from the known-good walker, to map how far the range widens before tracking breaks (robust vs
  chaining from P's uncertain-quality ckpt). Queue 21 specs, pending 5. No abort (Q healthy), escalation stands.
- **Next read:** if T (mild omni) holds tracking from the walker, the curriculum's early stages are survivable
  directly; if only forward holds, the curriculum must COMPOUND -> the --cmd_curriculum wiring is the critical path.

## STABILITY wake 9 (2026-07-27): width-sweep = MONOTONIC degradation with command width
- Q/R graded, S live. Achieved/commanded ratio + return both fall monotonically with width:
  P(fwd) 0.60/ret74 -> Q(+lat) 0.42/58 -> R(+back) 0.28/55 -> S(+yaw) ~/50. Forward tracks; each added
  dimension degrades. Even MILD widening (Q) degrades DIRECTLY from the walker -> curriculum must COMPOUND.
- Engineer still hasn't wired --cmd_curriculum (grep=0). Strengthened the wake-7 escalation with this evidence
  (it is now the SOLE path; 18 runs exhaust reward/weight/step levers).
- **Action:** appended V/W = COMPOUNDING-chain specs (warm from P's forward-tracker ckpt + add lateral (V) /
  lateral+yaw (W), forward vx). KEY TEST: does building on P beat widening from the walker (Q)? If yes, I
  hand-crank the curriculum stage-by-stage; if V~=Q, only the in-run ramp works. Queue 23 specs, pending 5.
  Kept T/U (mild/moderate @5M from walker) — they test the STEPS axis (does more time help mild widths?).

## STABILITY wake 10 (2026-07-27): steps don't rescue direct widening -> prioritized the compounding test
- T-cur (mild widen vx0-0.6/lat+yaw+-0.2 @5M from walker): ratio ~0.44 (gs 0.144 / mean cmd 0.33), return 59
  == Q's 0.42 at 3M. MORE STEPS DO NOT rescue direct widening from the walker. T/U redundant with Q/R.
- Aborted T (idx19) + U (idx20) to launch the decisive COMPOUNDING test V (idx21, chain from P's forward-tracker
  + lateral). Appended X (chain from P + backward-only) and Y (chain from P + near-full @5M). Queue 25 specs,
  pending 4 (V/W/X/Y). All four are stage-2 compounding tests from P across different added dimensions.
- Engineer STILL hasn't wired --cmd_curriculum (grep=0, ~30h post-escalation). Manual compounding is now the
  primary path; V/W/X/Y answer whether it works.
- **Decisive next read:** compare V/W/X (chain from P, single-dim adds) achieved/commanded ratio vs Q/R (same
  widths from the walker). If chaining holds ~0.6 where direct-widening gave ~0.3-0.4, COMPOUNDING WORKS -> I
  hand-crank stage 3+ from the winning ckpt. If V~=Q, only the in-run --cmd_curriculum ramp can solve it and the
  escalation is the sole path.

## STABILITY wake 11 (2026-07-27): COMPOUNDING FAILS -> manual paths exhausted, in-run curriculum is sole path
- V-chain (idx21, warm from P + lateral +-0.2): ratio 0.40 (gs 0.111/mcmd 0.28) == Q's 0.42 (same width from
  walker). Compounding via warm-start chaining does NOT beat direct widening; online refine re-collapses
  regardless of init. W (live, +lat+yaw) ~0.49 (noisy, not conclusive).
- Manual levers now exhausted across ~22 runs: reward weights, steps, standing-terms, direct-widen, compounding.
  Only the IN-RUN --cmd_curriculum ramp (tiny tracking-gated width increments) is untested and plausibly works.
- **Actions:** (1) hardened escalation (UPDATE 2): compounding also fails -> --cmd_curriculum is the definitive
  sole path (still unwired, grep=0, 30h+). (2) appended Z1/Z2 = granularity probe (tiny +-0.1 lateral / -0.15
  backward chained from P) — the one thing distinguishing manual chaining from the in-run ramp. Queue 27, pending 5.
- **Next read:** if Z1/Z2 (tiny increments) hold ~0.6 where V(+-0.2) gave 0.40, fine-grained manual chaining is
  viable and I hand-crank many tiny stages; if they also degrade, manual chaining is definitively dead and the
  search is blocked on the --cmd_curriculum wiring until the engineer/human acts.

## STABILITY wake 12 (2026-07-27): compounding DEAD (3 corroborations) -> testing exploration levers
- V/W/X (chain from P + lateral / lat+yaw / backward): ratios 0.40 / 0.40 / 0.30 == direct-widen Q/R (0.42/0.28).
  Compounding via warm-start chaining is definitively dead across all dimensions; online refine re-collapses
  regardless of init. Y (near-full @5M) running = 4th confirmation.
- SEARCH BLOCKED: grader is always full-omni, so nothing grades >~0 until the policy tracks omni, which needs the
  incremental --cmd_curriculum ramp (still unwired, grep=0, 36h+ post-escalation). Manual levers exhausted:
  reward weights, steps, standing-terms, direct-widen, compounding.
- **Action:** queued 2 EXPLORATION-lever probes (the one untested seeding dim; all prior runs used identical BC):
  AA (minimal BC: seed 50k/pretrain 2k) and AB (doubled plan-exploration bc_plan_std 0.6), both +lateral from
  walker. Tests whether less anchoring / more exploration lets the refine learn lateral (ratio > V's 0.40). Plus
  Z1/Z2 (granularity) still queued. Queue 29, pending 5. No abort (Y is a useful 4th corroboration, keeps GPU busy).
- **Read:** if AA/AB/Z1/Z2 all == ~0.40, EVERY within-advisor lever is exhausted and the search is hard-blocked
  on the --cmd_curriculum wiring (engineer/human action required). If any breaks 0.40, that's the new thread.

## STABILITY wake 13 (2026-07-28): granularity marginally helps (Z1 0.49 > V 0.40); starting fine-grained chain
- Z1 (tiny lat +-0.1 chain from P): ratio ~0.49 vs V (+-0.2) 0.40 -> finer increment helps a little, in the
  predicted direction, but still far below the ~0.6 fwd baseline and nowhere near the win bar (needs ~0.9+).
  Y = 4th compounding confirmation (0.0001). Z2 (tiny back) live, high return 81.7.
- Confirms granularity matters but manual runs can't reach the ~50k-step increments the in-run --cmd_curriculum
  ramp uses. Curriculum still unwired (grep=0, ~42h). AA/AB (exploration levers) still queued.
- **Action:** queued AC/AD = FINE-GRAINED MANUAL CHAIN from Z1 (widen lat to +-0.15 / add yaw +-0.1). Tests
  whether chaining SMALL increments ACCUMULATES where coarse +-0.2 compounding (V/W/X) re-collapsed. If AC/AD hold
  ~0.5 at wider range, hand-cranked fine curriculum is viable and I chain stage 3+ toward omni; if they drop to
  ~0.4, only the in-run ramp is fine enough. Queue 31, pending 5.
- **Reality check:** all manual levers top out ~0.4-0.5 ratio, far from the win bar. Without --cmd_curriculum the
  full-omni goal is likely unreachable this week; the fine-chain is the best-available approximation and the GPU
  stays productive, but the clean fix remains the escalated bootstrap wiring (human/engineer action).

## STABILITY wake 14 (2026-07-28): investigation COMPLETE - ceiling ~0.5 across all levers; final escalation
- AA (minimal BC): ratio 0.49; Z2 (tiny backward): 0.39. BC/exploration is NOT the cause. AB (high plan-std) live.
- CEILING CONFIRMED ~0.5 achieved/commanded across EVERY advisor lever (weights/steps/standing/direct-widen/
  compounding/granularity/exploration). Win bar needs ~0.9+. Full-omni goal is HARD-BLOCKED on --cmd_curriculum
  wiring (grep=0, ~48h, engineer not acting).
- **Actions:** (1) wrote FINAL/HUMAN-ACTION-REQUIRED escalation in ADVISOR_ALERTS.md (full lever table + the exact
  6-line bootstrap.py fix). (2) queued AE (stack the marginal helpers: fine width + low BC + high std, 5M -- last
  shot at 0.5) and AF (FALLBACK DELIVERABLE: warm P + quietness levers on forward+mild range @10M -- first quietness
  tuning on a tracking-capable base; a real stable forward walker if the curriculum never lands). Queue 33, pending 5.
- **Posture going forward:** the strategy question is settled. Remaining wakes = keep GPU on the deliverable/last
  probes, watch for the curriculum getting wired (grep cmd_curriculum in bootstrap.py) -- the moment it is, pivot ALL
  specs to full-omni + --cmd_curriculum runs. Until then, no new manual lever is worth much; avoid churning repeats.

## STABILITY wake 15 (2026-07-28): monitoring - ceiling holds, queue busy ~24h, watching for curriculum wiring
- AB (high plan-std): ratio 0.42, fit 0.0001 -> exploration confirmed not the lever. AC (fine-chain from Z1,
  live) marginally higher ~0.56 -- watch its final grade, but still far from win bar (~0.9+).
- No action: queue 4 pending (AC 3M/AD 3M/AE 5M/AF 10M = ~24h compute, no idle risk); strategy settled (blocked on
  --cmd_curriculum, grep=0, ~54h). Avoided churning more ceiling repeats. FINAL escalation stands (wake 14).
- WATCH each wake: grep cmd_curriculum scripts/tdmpc/bootstrap.py -- if it becomes >0, PIVOT all specs to
  full-omni + --cmd_curriculum (drop the ranges.* overrides) and the search unblocks.

## STABILITY wake 16 (2026-07-28): fine-chain is the best manual thread (~0.55), slowly accumulating
- AC (Z1+lat0.15) 0.51, AD (Z1+yaw0.1) 0.57 -- best manual results yet, vs ~0.40 coarse. Fine granularity DOES
  accumulate a little through chaining (validates the curriculum principle). But ranges still tiny (lat/yaw +-0.1-0.15
  vs full omni vx+-0.8); can't reach win-bar omni manually in the time left. AE (combine) live ~0.54.
- Curriculum still unwired (grep=0, ~60h). Best fitness 0.0003. Machinery healthy, disk 14%.
- **Action:** chained stage 3 from AD (best base): AG (widen lat->0.2/yaw->0.15) + AH (add tiny backward). Grows the
  hand-cranked curriculum; the widest range that still holds ~0.5+ becomes the best fallback deliverable. Queue 35, pending 4.
- Continue watching grep cmd_curriculum -> pivot to full-omni + --cmd_curriculum the moment it's wired.

## STABILITY wake 17 (2026-07-29): ceiling holds (AE 0.46); fine-chain still best (AG ~0.58); deliverable notes
- AE (combine marginal helpers) 0.46 -> they do NOT stack past 0.5. AF (deliverable @10M): vertical_speed 0.032
  (BELOW win bar!) but rocking 0.730 / fwd 0.029 polluted by the full-omni grader (flails on un-trackable cmds);
  true forward-range quietness unmeasurable via harness. AG (fine-chain st3) holds ~0.58 at wider range.
- Curriculum unwired (grep=0, ~66h). Best fitness 0.0003. Machinery healthy.
- **Action:** chained stage 4 @5M from AD: AI (add tiny back + lat/yaw 0.2) and AJ (forward-hemisphere wide: vx0-0.6
  lat0.25 yaw0.2, no backward -- tests if a fwd+turn+strafe walker reaches higher ratio, best achievable deliverable).
  Queue 37, pending 4. Watching grep cmd_curriculum for the unblock.

## STABILITY wake 18 (2026-07-29): fine-chain does NOT accumulate -> consolidate best config as deliverable
- AG (st3, lat0.2/yaw0.15) final 0.44, AH (+back0.15) 0.34 -- DOWN from AD's 0.57 at narrower range. The fine-chain
  degrades with width just like direct widening (0.57->0.44->0.34). CONCLUSION: every manual approach is capped by
  command width; no manual path holds tracking at wide range. Only the in-run --cmd_curriculum ramp can. (grep=0, ~72h)
- **Pivot:** stop churning capped-width probes. Queued AK (10M) + AL (20M) = consolidate AD's best config (narrow
  fwd+lat0.1+yaw0.1, ratio 0.57) + mild quietness (flat_orient -0.3, ang_vel_xy -0.05) into the best-achievable STABLE
  narrow-omni walker (the fallback deliverable). These occupy the GPU ~30h, ending the per-wake churn. Queue 39, pending 4.
- Best config so far = AD (fwd+lat0.1+yaw0.1, ratio 0.57). Watching grep cmd_curriculum -> if wired, ABORT AL and
  pivot ALL specs to full-omni + --cmd_curriculum (the only path to the actual win bar).

## STABILITY wake 19 (2026-07-29): width cap re-confirmed (AI 0.36); holding on deliverable consolidation
- AI (chain4 @5M wider+back) 0.36 -- more steps don't beat the width cap. Consistent with all prior. best fit 0.0003.
- Curriculum unwired (grep=0, ~78h). Queue now AJ(5M)/AK(10M)/AL(20M)/AM(10M) = ~45M compute (~2 days), no idle risk.
- Added AM = deliverable with STRONGER quietness (flat_orient -0.6, ang_vel -0.1) on AD's narrow range, to tune the
  clean-IMU half on the achievable range. Best of AK/AL/AM is the fallback stable-walker deliverable.
- Holding pattern: settled/blocked. Watching grep cmd_curriculum -> pivot to full-omni + --cmd_curriculum when wired.

## BOOTSTRAP-FROM-PPO (07-24) — the win, and the POSTURE hole it hid (measured 2026-08-12)
Belated journal entry: the two bootstrap wins were never written up here (they live in
`bootstrap_queue.jsonl` + `supervisor_journal_bootstrap.jsonl`). Both cleared the win bar —
A-base fitness 0.913 / fwd 0.363, B-strongbc 0.924 / fwd 0.363, 0 falls, held to 3M — and are the
ONLY 2 of 52 graded evals in this project ever to exceed 0.073 m/s. Seeding the replay buffer from
`deploy/walk/policy.pt` genuinely escaped the standing basin that the whole reward search could not.

**Then the operator watched it: it walks bent far forward.** Measured, not eyeballed:

| policy | fwd speed | cmd | torso lean | tilt (mean/max) |
|---|---|---|---|---|
| PPO demo `deploy/walk/policy.pt` | 0.300 | 0.300 | **-3.3 deg** (nose-UP) | — |
| bootstrap winner `walk_win_0.924` | 0.363 | 0.300 | **+27.3 deg** | 29.2 / **98.2** deg |

So the 3M-step online phase did not *refine* the demo — it drifted 30 deg off it and overshot the
command by 21%. The demo was already the better policy on both axes.

**Root cause (mechanical, verified):** nothing in the objective priced torso attitude.
`gated_locomotion` computed `upright = _tolerance(up, lower=upright_min, margin=upright_min)` with
`upright_min=0.8` — margin hardwired to the threshold makes the gate FLAT over every posture a
walker can hold: a 30-deg lean cost **0.0%** of reward, 45 deg cost 3.1%. Meanwhile leaning forward
directly buys the `move` term, and `flat_orientation_l2` (the one term that would have caught it)
sits at weight 0. The policy did exactly what it was asked. Secondary: the 400k seed is the OLDEST
data in a 1M ring, so it was fully evicted by ~1.0M of 3M steps (confirmed: `buffer/size` hits 1e6
at ~620k) and the BC-via-prior anchor died with it — nothing held the gait near the demo after that.
Also note the 98-deg tilt peak: under `NonEpisodicTerminationsCfg` an env can go past horizontal and
recover without ever registering a "fall", so "0 falls" is a weaker claim than it reads as.

**Fixes landed (this commit):**
- `gated_locomotion` gains `upright_margin` (default `None` = legacy flat behaviour, so no earlier
  run changes). `0.97/0.10` = free to 15 deg, 0.81 at 20, 0.24 at 27, 0.005 at 35.
- Posture is now MEASURED everywhere: `torso_lean_fwd_deg` / `torso_tilt_deg` / `_max` in
  eval_smoothness, `collect/torso_lean_deg` in TB + the trainer's step line, and a lean readout in
  bootstrap's phase-0 so we know whether the DEMO leans before trusting a seed.
- `grade_run.py` win bar gains `WIN_LEAN_DEG = 15`. Runs graded before the metric existed are not
  judged on it; **the 07-24 winners would fail the new bar.**
- `--buffer_size` on bootstrap.py = the demo-ANCHOR lever (size > seed + budget -> seed never evicts).
- Two review bugs fixed: the seed loop stepped the env with the UNCLAMPED PPO action while storing
  the clamped one (`mdp.last_action` feeds the raw value back as 12 of 45 obs dims, so seed states
  were unreachable online), and the seed->online seam was never flagged `time_out` so sampled
  windows could straddle `TdmpcTrainer`'s `env.reset()`.

**Caveats on the 07-24 result that stand regardless (from the 08-12 code review):**
- Both "replicates" ran `seed: 0` with `torch.manual_seed` before net construction -> identical
  init. That is n=1 with one perturbation, not two runs. It explains fwd agreeing to 0.1%
  (0.36312 vs 0.36348) while base_accel differed 18%. The lean fleet uses seeds 1-4.
- The bootstrap fleet switched to `--plant baseline` while the entire prior reward search ran on
  `modeled`. The headline 0.363-vs-0.03 contrast is two changes, not one; the matched control
  (scratch + fixed cmd + baseline plant) has still never been run. Partial defence: the 40 later
  varied-command bootstrap runs are all baseline-plant and still collapse, so the easy plant is not
  sufficient on its own.
- `fitness` is near-constant for any non-faller: `walk_gate` saturates at 0.12 m/s, `SURVIVE_TARGET_S`
  (20 s) IS the max episode length, `upright`=1 at 0 falls -> fitness == 0.8 + 0.2*smooth. Quote
  `forward_speed_mean` + `is_win`, never the fitness scalar.

## ANTI-LEAN fleet (launched 2026-08-12) — `lean_queue.jsonl`, `--run_tag lean`
Four arms x 3M, baseline plant, fixed cmd 0.3, otherwise identical to bootstrap-B-strongbc, seeds 1-4.
Crosses the two levers: PRICE THE LEAN (upright gate firm 0.97/0.10 vs soft 0.95/0.15 vs additive
`flat_orientation_l2` -0.5) x ANCHOR THE DEMO (`--buffer_size 3500000` -> seed resident all 3M, ~12%
of the buffer at the end, vs today's full eviction at 1M).
- **A firm** (gate only, seed still evicts) — does pricing lean work at all? Does it drift back after 1M?
- **B firm+anchor** — the main bet; both mechanisms pull toward the upright demo.
- **C soft+anchor** — safety arm against the recurring "extra reward pressure barbers the gait" failure.
- **D flat+anchor** — additive lever instead of multiplicative; C-level tried this weight under VARIED
  commands and collapsed, it deserves one fair shot on a held fixed-command walk.
Watch `collect/torso_lean_deg` next to `collect/ground_speed_mps`: the failure to catch early is
speed holding while lean grows (the 07-24 mode), or lean going to ~0 because it stopped walking.

### lean-A-firm RESULT (power-cut at 2.80M/3.00M, graded manually 2026-08-12)
Mains dropped at ~19:41 (circuit blew), 93% through arm A. Graded `model_2750880.pt` (last intact
checkpoint; `model_2800896.pt` was truncated to 0 bytes mid-write). NOT resumed — lean had plateaued
since ~1.2M and bootstrap.py has no resume, so the last 7% would have cost a full ~4 h re-run.

| | 07-24 winner | lean-A @2.75M |
|---|---|---|
| fwd m/s | 0.363 | 0.332 |
| **torso lean** | **+27.3 deg** | **-3.8 deg** |
| torso tilt (mag) | 29.2 | 19.3 |
| tilt max | 98.2 | 101.8 |
| rocking_rms | 1.449 | 1.230 |
| base_accel_rms | 2.887 | 2.429 |
| falls | 0 | 0 |
| fitness | 0.924 | **0.933** (best recorded) |

**THE GATE WORKS — and it is not enough.** Collection lean went +25.7 -> -1.3 deg within 200k steps.
Speed cost 9% (0.363 -> 0.332), which also improved command tracking (1.21x -> 1.11x of the 0.3 cmd);
rocking, accel and action-rate all improved. But two things the run taught us that the design missed:

1. **The gate has a FLAT INTERIOR.** `_tolerance` returns exactly 1.0 inside its bound, so the gate
   BOUNDS tilt and can never MINIMIZE it — the policy parks at the edge of whatever free zone it is
   given. Residual tilt magnitude is still 19.3 deg with tilt_max ~102 deg (barely moved). The
   lean/tilt gap is the diagnostic: the 07-24 winner had lean ~= tilt (27.3 vs 29.2) = a STEADY
   forward pitch; lean-A has -3.8 vs 19.3 = the mean is gone but it now WOBBLES. Note also that
   `torso_lean_fwd_deg` averages a SIGNED value, so a +-20 deg fore-aft rock reads as ~0, and lateral
   roll is invisible to it entirely — always read it beside `torso_tilt_deg`.
2. **The gate is DIRECTION-AGNOSTIC** (it gates cos(tilt)), so +27 deg and -27 deg pay identically.
   lean-A duly slid from +27 to -15 deg (leaning BACKWARD) at no reward cost. This also exposed a bug
   in the new win bar — `lean <= 15` is one-sided and a -17 deg backward lean would have PASSED. Now
   `abs(lean) <= WIN_LEAN_DEG`.

**Unplanned result — direct evidence for the demo-anchor hypothesis.** The collection lean sat within
a few degrees of the demo's -5.5 for the first 1M steps, then slid hard exactly where the seed is
fully evicted:
```
0.20M -1.3   0.40M -6.4   0.60M -4.3   0.80M -5.7   1.00M -9.6  <-- seed fully evicted
1.20M -18.7  1.40M -11.7  1.80M -16.5  2.60M -15.1
```
This is the F9 drift mechanism showing up unprompted, and it makes lean-B (firm + anchor) a sharp
falsifiable test: lean should hold near -5 past 1M instead of sliding to -15.

**Appended lean-E-firm-flat** (seed 5): firm gate + `flat_orientation_l2` -0.5 + anchor. The L2 term
is minimized only at zero tilt (gradient everywhere, and it sees LATERAL roll), so it attacks exactly
the residual the gate structurally cannot. D isolates the L2 term alone, so E-vs-D attributes any
gain to the combination. Fleet resumed at idx 1 (B, C, D, E), state next_index=1, runs=1, wins=1.

## ANTI-LEAN fleet RESULT (2026-08-13): lean fixed on 3 seeds; the SOFT gate wins
Stopped after 3 graded arms (D killed at 0.82M, E never launched) — the primary question was
answered and the GPU was wanted for visual inspection. Deterministic eval, same protocol throughout:

| arm | fwd m/s | lean deg | tilt deg | rocking | accel | fitness |
|---|---|---|---|---|---|---|
| 07-24 winner (baseline) | 0.363 | **+27.3** | 29.2 | 1.449 | 2.887 | 0.9236 |
| A firm 0.97/0.10, evict, seed 1 | 0.332 | -3.8 | 19.3 | 1.230 | 2.429 | 0.9334 |
| B firm 0.97/0.10, anchor, seed 2 | 0.356 | +4.3 | 18.7 | 1.290 | 2.960 | 0.9221 |
| **C soft 0.95/0.15, anchor, seed 3** | 0.306 | -4.7 | **14.0** | **0.875** | 2.500 | 0.9318 |

**Pricing torso attitude works, and it replicates.** Three seeds, three reward configs, all land
within +-5 deg from a +27.3 baseline. This is the first result in the project with a genuine
multi-seed confirmation.

**The SOFT gate beat the firm one — the opposite of the prediction.** C has the lowest tilt (14.0 vs
18.7-19.3), the lowest rocking (0.875, -40% vs the old winner) and near-exact command tracking
(0.306 achieved vs 0.300 commanded = 1.02x, where the old winner ran 1.21x). The firm gate's sharp
cliff at 0.97/0.10 appears to make the policy fight the boundary; the gentler basin lets it settle.
Follow-ups should explore SOFTER, not firmer. C is preserved as
`_preserved/walk_posture_0.932_lean-C-soft-anchor_2026-08-13.pt`.

**Three corrections to earlier claims in this journal:**
1. **The anchor is NOT established.** The prediction was "B holds near -5 past 1M while A drifts".
   Collection lean: A (evict) +15.8 -> -19.4 -> -17.6 (drifts away); B (anchor) +19.3 -> +7.0 ->
   +2.5 (converges); but **C also has the anchor and still drifted to -10.8**. One seed per config
   means anchor and seed are confounded. No attribution is possible from this fleet.
2. **Collection lean systematically overstates the problem** (exploration noise wrecks posture):
   A -17.6 collect -> -3.8 eval; C -10.8 -> -4.7; B +2.5 -> +4.3. Judge posture on the GRADE.
   The live `OVER BAR` flags in fleet_status are mostly this artifact.
3. **`torso_tilt_deg_max` is not a discriminator** — ~101 deg for all three arms and 98 for the old
   winner. No lever moved it; at 64k samples/eval it is almost certainly reset/DR transients.

**Arm E's premise was undermined before it ran.** E (firm gate + flat_orientation_l2) was appended to
attack the residual ~19 deg tilt using a gradient-everywhere term — but C reached 14 deg with a
SOFTER gate and no flat term at all. The tilt reduction came from gate softness, not from adding a
penalty. D (flat_orientation alone) was stopped at 0.82M; both remain queued at next_index 3.

**Machinery note:** `preserve_if_better` uses fitness, which the code review showed is
~`0.8 + 0.2*smooth` for any non-faller and therefore cannot rank posture — it skipped C (0.9318 <
A's 0.9334) even though C is the better gait on every quality axis. Preserved manually. Worth
changing the preserve rule to consider posture, or preserving every `is_win` run.

## FIDELITY AUDIT vs tdmpc2_official (2026-08-13) — horizon is the lever, warm start is not
Audited our port line-by-line against `/home/nse/humanoid/tdmpc2_reference`. Full findings D1-D11 in
the audit artifact; the three that were tested this session:

**D4 torch.compile + Adam — NOT A BUG.** The reference builds both optimizers with `capturable=True`
(tdmpc2.py:29,31) because it compiles `_update` with `reduce-overhead`; we omit it. Measured
(`check_compile_adam.py`, `check_compile_engaged.py`, no Isaac needed): compiled is BIT-IDENTICAL to
eager over 40 updates and Adam's step counter advances correctly (40/40). Compile really does engage
(23 dynamo frames, 1.81x) rather than silently falling back. Mechanism: 10 graph breaks remain, at
`.backward()` (autograd untraced by default) and at `optim.zero_grad` + a `graph_break()` torch itself
inserts in the optimizer wrapper, so the optimizer step is never cudagraph-captured and the CPU step
counter cannot freeze. `capturable` is moot for us. Also corrected two stale claims in agent.py:
it is not "eager only" (compile on since 07-20) and the speedup is 1.81x, not ~3.6x.

**D2 MPPI warm start — implemented, NULL at inference.** `plan_batch` never carried `_prev_mean`
forward (upstream tdmpc2.py:168) though the unused single-obs `plan()` always did, so every
collection and eval step in this project planned from mean=0/std=max_std. Now behind
`cfg.mppi_warm_start` (default OFF; existing checkpoints re-grade identically) with a `reset` mask
for upstream's `t0`. A/B on lean-C: 38.5% -> 38.0% episodes-with-a-fall = 74 -> 73 of 192, ~1/7 of
one binomial SE. Indistinguishable from nothing. I had ranked this "severe" in the audit; it is not.
Does NOT rule out a training-time benefit (better plans -> better data + better plan_mean/std for the
TD-M(PC)2 prior), only a free inference-time one.

**D5 PLANNING HORIZON — the lever, and it needs no retraining.** The world model is one-step and no
loaded weight depends on horizon, so an existing checkpoint can simply plan further. Replicated on
three independently-trained policies (warm start held ON throughout):

| checkpoint | %eps fall H=3 -> H=6 | mean tilt | fwd speed |
|---|---|---|---|
| 07-24 winner | 30.2% -> **17.2%** | 29.5 -> 22.2 | 0.363 -> 0.425 |
| lean-A firm | 44.8% -> **32.3%** | 19.8 -> 16.8 | 0.320 -> 0.368 |
| lean-C soft | 38.5% -> **19.8%** | 14.5 -> **8.9** | 0.303 -> 0.313 |
| *PPO reference* | *6.2%* | *9.7* | *0.281* |

12.5-18.7 pp off the fall rate on every checkpoint, with tilt DOWN and speed UP. lean-C at H=6 has
mean tilt 8.9 deg, BELOW the PPO reference's 9.7. For lean-C, 38.0 -> 19.8% is 73 -> 38 episodes of
192, ~3.7 SE, p<0.001. H=12 (0.48 s) buys no further fall reduction (21.4%) and costs 15% of the
speed -> the learned dynamics degrade past ~0.25 s. This confirms the audit's mechanism argument:
0.12 s of lookahead against a ~0.7 s gait cycle was the binding constraint.

**Consequences.** (1) All 55 historical grades understate their checkpoints -- every one planned at
H=3. (2) H=6 closes about HALF the gap to PPO (19.8% vs 6.2%); the task-permissiveness finding is
independent and still stands. (3) MPPI cost scales linearly with H, so H=6 doubles planning cost --
a deployment consideration at 25 Hz, and a ~1.5-2x slowdown for online collection during training.
(4) Untested: training AT H=6, where the consistency loss would fit 6-step rollouts and the value
function would match the horizon actually used. Expected to beat plan-at-6-with-an-H3-model, but
that is a hypothesis, not a measurement.

## H=6 FLEET RESULT (2026-08-15) — nothing plateaued; flat_orientation wins; UTD 1.0 does not
All four arms ran to budget (no early stops). Graded at H=6 + warm start, objective walk_v2.

| arm | fitness | falls %eps | tilt | lean | fwd (x cmd) | note |
|---|---|---|---|---|---|---|
| **h6-flat** | **0.0936** | **12.5%** | **6.5** | -3.6 | 0.370 (1.23x) | + flat_orientation_l2 -0.5 |
| h6-base-rep | 0.0620 | 14.8% | 9.5 | +1.6 | 0.376 (1.25x) | seed 12 |
| h6-base | 0.0612 | 17.2% | 8.8 | -5.9 | 0.355 (1.18x) | seed 11 |
| h6-utd1 | 0.0523 | 16.4% | 9.3 | -2.0 | 0.380 (1.27x) | UTD 1.0 |
| lean-C (H3-trained) | 0.0807 | 18.8% | 9.1 | -3.9 | 0.312 (1.04x) | prior best |
| PPO reference | 0.2571 | 6.2% | 9.7 | -3.3 | 0.281 (0.94x) | the target |

**NOTHING PLATEAUED — the operator's read was right.** Speed slope over 2-3M was +0.030..+0.069
m/s per 1M in every arm, and the new in-training fall telemetry declined monotonically to the
budget wall without flattening:
```
falls/min      0.0M  0.5M  1.0M  1.5M  2.0M  2.5M   slope 2-3M
h6-flat        2.51  2.31  2.15  2.03  1.89  1.55     -0.612   <- ACCELERATING downward
h6-base-rep    2.34  2.06  2.02  1.90  1.79  1.71     -0.172
h6-utd1        2.40  2.25  2.17  1.89  1.77  1.70     -0.227
```
Falling is still being actively trained out at the point every run in this project has been stopped.

**flat_orientation_l2 -0.5 is the reward win.** Against its matched-config seed (h6-base-rep):
falls 14.8 -> 12.5%, tilt 9.5 -> 6.5 deg -- better torso attitude than the PPO reference (9.7) --
and by far the steepest fall-decline slope. This is the audit's prediction confirmed: the upright
gate has a FLAT INTERIOR and can only bound tilt, while an L2 term on gravity-xy has a gradient
everywhere and also sees lateral roll.

**Training at H=6 helps falls** (18.8% -> 12.5-17.2% vs the H=3-trained lean-C), so H=6 earns its
~1.4x collection cost. **UTD 1.0 does NOT help**: h6-utd1 was the worst arm (0.0523). Staying at
UTD 0.5 saves ~40% of wall clock at no measured cost. **Replication is tight** (base 0.0612 vs
rep 0.0620; falls 17.2 vs 14.8%, inside the ~3.5 pp binomial SE), so h6-flat's margin is real.

**New deficit, and it is the planner's not the reward's:** every H=6 arm overshoots the command
(1.18-1.27x vs lean-C's 1.04x). Nothing in the reward pays for exceeding it -- `move_go` clamps at
1.0 -- so this is the longer-horizon planner exploiting model error. It drags v2 fitness below
lean-C's despite better safety, and collection speed was still climbing +0.069/1M at the wall.

## LONG RUN LAUNCHED (2026-08-15) — `long_queue.jsonl`, 20M, stoppers off
First run in this project's history to be allowed past ~4.4M. h6-flat's measured config +
`track_lin_vel_xy 0.5` (the one untested element, to price the overshoot; PPO uses 1.787), H=6,
UTD 0.5, buffer 10M (3.3 GB; demo resident for the first ~10M), `--no_early_stop` so only a
pi_loss divergence can cut it. ~46 h at 121 steps/s; checkpoints every 50k so stopping early from
TensorBoard costs nothing. Watch `collect/falls_per_min` and `collect/ground_speed_mps` vs the
0.300 command. If falls drop while speed collapses toward 0.15, the tracking term is too strong --
stop, drop to 0.25, resume.

### LONG RUN take 1 ABORTED at 1.54M (2026-08-15) — I added a farmable reward term
Operator flagged that `mean_episode_len` looked wrong ~3.5 h in. It was: 443.9 and declining
(427 latest) where **every one of the 5 prior runs held exactly 500.0**. ep_len < 500 means
`hard_collapse` (base 0.30 m below standing) is firing -- the robot is genuinely ending up on the
floor, which had never happened in this project.

Cause: `track_lin_vel_xy 0.5`, the one untested element I added to price the overshoot.
`track_lin_vel_xy_yaw_frame_exp` is `exp(-lin_vel_error/std^2)` with **no uprightness or height
factor at all**. Per-step unscaled reward for a robot moving at the commanded 0.3 m/s:

| term | weight | gated | upright walker | COLLAPSED, sliding |
|---|---|---|---|---|
| stand_walk (core) | 1.0 | YES | 0.60 | 0.00 |
| upright_bonus | 0.3 | no (+) | 0.30 | 0.00 |
| feet_air_time | 1.5 | YES | 0.10 | 0.00 |
| flat_orientation_l2 | -0.5 | no (-) | -0.02 | -0.35 |
| **track_lin_vel_xy** | **0.5** | **NO (+)** | **0.50** | **0.50** |
| h6-flat total | | | 0.98 | **-0.35** |
| take-1 total | | | 1.48 | **+0.15** |

A collapsed robot went from -0.35/step (strictly worse than any upright state) to +0.15/step,
because an ungated velocity-tracking term cannot distinguish a walk from a body sliding along the
floor at the right speed. Measured: return ROSE 14.4 -> 17.4 (+21%) while episodes SHORTENED --
earning more per step while collapsing more. That is the run-6 signature.

This violated the env's own stated design rule, which is written at the top of `HybridRewardsCfg`:
"(b) additive NEGATIVE penalties can only subtract, so they can't be farmed while falling;
(c) additive POSITIVE shaping is kept small/moderate so it can't out-earn actually walking while
upright." `flat_orientation_l2 -0.5` obeys it (negative, so unfarmable -- which is why h6-flat was
clean). `track_lin_vel_xy +0.5` broke it: an ungated positive worth half the entire gated core.
The stability fleet's warning about this term was in the journal and I reasoned it away as a
varied-commands artifact; the real mechanism is simply that it is ungated.

**Relaunched as long-20M-v2 with h6-flat's config EXACTLY.** The overshoot (1.23x) is real but
stays unpriced for now -- it must be fixed with an additive-NEGATIVE penalty on speed ABOVE the
command, or a properly gated tracking term, as its own experiment rather than a passenger on a
46 h run.

**Lesson for the watch-list: RETURN IS NOT A HEALTH SIGNAL.** Take 1 had the highest return ever
recorded in this project while collapsing in ~22% of episodes. `collect/mean_episode_len` pinned
at 500 is the canary; it is now first on the queue's watch-list.
