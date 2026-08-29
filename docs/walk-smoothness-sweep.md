# Walk smoothness sweep — 3 fast-profile runs

**Status:** full run complete. ⚠️ **See §10 — the screening method used in §5-§9 is unreliable; conclusions revised.** **Owner:** training PC. **Created:** 2026-08-24.
**Goal:** find reward weights that lower the walk policy's control frequency from **5.1 Hz to
~1.5–2.5 Hz** without collapsing the gait into standing, then pick one for a full run.

---

## 1. Why — the hardware evidence

The deployed walk policy (`deploy/walk/`, run `2026-08-18_20-45-17`, kp=45/kd=1.5) **cannot be
executed by the real robot.** Measured on `humanoid-control` over four instrumented runs on
2026-08-24:

| finding | value |
|---|---|
| Policy gait frequency (ideal closed loop) | **5.12 Hz** |
| Knee swing amplitude | 0.775 rad |
| Peak commanded joint velocity | 8–10 rad/s |
| Peak *demanded* joint torque | up to **47.9 Nm** (motor ceiling ≈ 26.9 Nm) |
| Knee L/R phase correlation on hardware | **+0.91** (should be ≈ −0.85 for stepping) |

A 12.6 kg biped should step at roughly **1–2 Hz**. At 5 Hz the robot never steps: it thrashes,
both knees sag in phase, and the motion cannot be tracked.

Three runtime-side fixes were tried on the robot and are now **exhausted**:

1. **Action clip** (runtime was clipping at ±100 instead of the trainer's ±4) — fixed; real
   correctness win (`prev_action` was being fed to the net at 10.17, far outside training range),
   but it did not change the gait.
2. **Knee torque limit 6 → 11 Nm** — fixed a genuine static droop (knee mean error −0.198 → −0.025
   rad). But with authority restored the joint *followed* the 4.4 Hz command and shook an I2C
   magnetic encoder into `ERROR_ENCODER_FAULT`. The 6 Nm cap had been acting as an accidental
   low-pass filter.
3. **Action low-pass filter** — **proven non-viable.** The policy's gait *is* the high-frequency
   content, so any cutoff low enough to smooth the motion also deletes the gait:

   | filter alpha | cutoff | knee L/R corr | knee swing | verdict |
   |---|---|---|---|---|
   | 1.00 (none) | – | −0.62 | 0.775 | gait intact |
   | 0.70 | 2.8 Hz | −0.92 | 0.372 | gait intact |
   | 0.50 | 2.0 Hz | −0.77 | **0.000** | **gait killed** |
   | 0.30 | 1.2 Hz | −0.99 | **0.004** | **gait killed** |

**Conclusion: the fix must come from training.** The smoothness penalties that would prevent this
already exist in the env cfg and are switched off.

---

## 2. The root cause in this repo

`source/humanoid_policy/humanoid_policy/tasks/locomotion/velocity/config/biped/env_cfg.py`:

```python
action_rate_l2  weight = -0.014      # very weak
action_l2       weight =  0.0        # OFF
dof_vel_l2      weight =  0.0        # OFF
dof_acc_l2      weight = -1.027e-6   # negligible
```

The file's own comment already predicts this failure:

> `action_l2` / `dof_vel_l2`: hardware-safety penalties (`docs/walk-policy-divergence-report.md`
> §4B) that suppress high-frequency, large-amplitude actions. Berkeley/g2c3 leave them OFF
> (they had over-damped the gait into standing); re-introduce with small weights for sim→real.

### Measured term magnitudes (current policy, vx = 0.5, ideal loop)

| term | raw value | × current weight | as % of tracking reward (1.787) |
|---|---|---|---|
| `action_rate_l2` = Σ(Δa²) | 4.35 | 0.061 | **3.4%** |
| `action_l2` = Σ(a²) | 17.90 | 0 (off) | 0% |
| `dof_vel_l2` = Σ(v²) | 124.42 | 0 (off) | 0% |

The smoothness penalty is effectively **switched off**. That is why the gait is at 5 Hz.

> ⚠️ **Known failure mode:** an earlier penalty stack over-damped the policy into standing still.
> That is why this is a *graded* sweep and why the screening metric below is gait frequency, **not
> reward** — a collapsed policy scores well on smoothness.

---

## 3. PREREQUISITE — update `_CONTRACT_EFFORT` before any of these runs

**Do this first. It changes the plant, so it must be identical across all three runs and the
full run.**

`source/humanoid_policy_assets/humanoid_policy_assets/robots/humanoid.py` still has **6.0 Nm**
knees. The robot has since been flashed and verified at:

| joint | was | now (flashed, verified 12/12) |
|---|---|---|
| `leg_left_knee_pitch_joint` | 6.0 | **11.0** |
| `leg_right_knee_pitch_joint` | 6.0 | **11.0** |
| `leg_right_ankle_roll_joint` | 6.0 | **7.0** |

Why the knees were raised: on hardware they showed a persistent *static* droop — mean error
−0.136 / −0.198 rad with |mean|/rms ≈ 0.5, i.e. the knee sat more bent than commanded and 6 Nm
could not extend it. That implied 6.1 / 8.9 Nm of steady demand against a 6.0 cap, saturating
38.7% / 45.1% of policy steps. Motor ceiling is Kt·I·gear ≈ 26.9 Nm and `leg_left_hip_yaw_joint`
already runs 12.0 Nm on the identical actuator. Raising it fixed the droop (knee mean error →
−0.036 / −0.025, a 4–8× improvement).

⚠️ **This raises the weights you will need.** The 6 Nm cap was acting as an *implicit* smoothness
constraint in both sim and hardware — the joint simply could not execute fast, large commands. At
11 Nm that constraint is gone, so the explicit penalties now have to do work the torque ceiling was
doing for free. If run B looks only marginally better, go to run C rather than concluding the
weights are the wrong lever.

---

## 4. The three runs

Edit the four weights in the `RewardsCfg` class of
`source/humanoid_policy/humanoid_policy/tasks/locomotion/velocity/config/biped/env_cfg.py`.
**Change nothing else** — same plant, same gains (`HUMANOID_GAIN_PRESET` default `tuned` = 45/1.5),
same profile, same seed if one is set. Only the weights may differ between runs.

| run | `action_rate_l2` | `action_l2` | `dof_vel_l2` | total penalty | % of tracking |
|---|---|---|---|---|---|
| **A — light** | −0.05 | −0.002 | −1e-4 | ~0.23 | ~13% |
| **B — moderate** ⭐ | −0.10 | −0.005 | −3e-4 | ~0.48 | ~27% |
| **C — strong** | −0.20 | −0.010 | −1e-3 | ~1.08 | ~60% |

`dof_acc_l2` stays at −1.027e-6 in all three.

**Run B first** — it is the most likely to land in the useful band. Then A and C to bracket it.
Run C is where the collapse-to-standing risk lives; expect it as a plausible outcome, not a bug.

### Commands

```bash
# ~1.6 h each on the RTX 5080 (4096 envs x 24 steps x 6000 iters = 590M samples)
python scripts/rsl_rl/train.py --variant walk-biped --profile fast --headless

# export the trained policy for screening (writes exported/ + configs/)
python scripts/rsl_rl/play.py --variant walk-biped --load_run <run_dir> --checkpoint model_5999.pt
```

Record for each run: run directory, final `mean_reward`, final `mean_episode_length`.

---

## 5. Screening — run this on each exported `policy.onnx`

**Do not judge these runs on reward.** A policy that has collapsed into standing scores *well* on
smoothness and badly on nothing obvious. Judge on gait frequency and whether it still steps.

Save as `scripts/screen_gait.py` and run `python scripts/screen_gait.py <path/to/policy.onnx>`.
Needs only `numpy` + `onnxruntime`.

```python
"""Screen an exported walk policy for a hardware-executable gait. No robot required."""
import sys
import numpy as np, onnxruntime as ort

ONNX = sys.argv[1] if len(sys.argv) > 1 else "exported/policy.onnx"
DT, ASCALE = 0.04, 0.25
# canonical joint order (L then R): hip_roll, hip_yaw, hip_pitch, knee_pitch, ankle_pitch, ankle_roll
KL, KR, HL, HR = 3, 9, 2, 8
DEFAULT = np.array([0.11, 0.0, -0.24, 0.83, -0.56, -0.07,
                    -0.11, -0.0, -0.24, 0.83, -0.56, 0.07], np.float32)
LO = np.array([-0.1745, -0.9817, -1.8980, 0.0, -0.7854, -0.2618,
               -1.5708, -0.5890, -1.8980, 0.0, -0.7854, -0.2618], np.float32)
HI = np.array([1.5708, 0.5890, 0.9817, 2.4435, 0.7854, 0.2618,
               0.1745, 0.9817, 0.9817, 2.4435, 0.7854, 0.2618], np.float32)
SIGN = np.ones(12, np.float32); SIGN[[6, 7, 11]] = -1.0   # URDF-mirrored right roll/yaw joints

s = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"]); inp = s.get_inputs()[0].name

def rollout(vx, n=500, track=0.9):
    pos, prev, P, A, V = DEFAULT.copy(), np.zeros(12, np.float32), [], [], []
    for k in range(n):
        vel = np.zeros(12, np.float32) if k == 0 else (pos - P[-1]) / DT
        obs = np.concatenate([[vx, 0, 0], [0, 0, 0], [0, 0, -1],
                              SIGN * (pos - DEFAULT), SIGN * vel, prev]).astype(np.float32)
        a = s.run(None, {inp: obs.reshape(1, 45)})[0].ravel()
        P.append(pos.copy()); A.append(a.copy()); V.append(vel.copy())
        a = np.clip(a, -4.0, 4.0)                       # trainer action_limit
        tgt = np.clip(SIGN * (a * ASCALE) + DEFAULT, LO, HI)
        pos = pos + track * (tgt - pos); prev = a
    sl = slice(100, None)
    return np.array(P)[sl], np.array(A)[sl], np.array(V)[sl]

def dom_hz(x):
    x = x - x.mean(); f = np.fft.rfftfreq(len(x), DT); Pw = np.abs(np.fft.rfft(x)) ** 2
    return float(f[1:][np.argmax(Pw[1:])])

def corr(x, y):
    x, y = x - x.mean(), y - y.mean()
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-9))

print(f"{ONNX}\n{'vx':>5s} {'gait Hz':>8s} {'knee corr':>10s} {'hip corr':>9s} "
      f"{'knee swing':>11s} {'max|v|':>7s} {'S(da^2)':>8s}  verdict")
for vx in (0.3, 0.4, 0.5, 0.6):
    P, A, V = rollout(vx)
    hz, kc, hc = dom_hz(P[:, KL]), corr(P[:, KL], P[:, KR]), corr(P[:, HL], P[:, HR])
    sw, mv = np.ptp(P[:, KL]), np.abs(V).max()
    ar = (np.diff(A, axis=0) ** 2).sum(axis=1).mean()
    if sw < 0.05:                       v = "COLLAPSED (standing)"
    elif hz > 3.5:                      v = "still too fast"
    elif kc > -0.3 and hc > -0.3:       v = "no gait phase"
    elif 1.0 <= hz <= 3.0:              v = "*** GOOD ***"
    else:                               v = "marginal"
    print(f"{vx:5.2f} {hz:8.2f} {kc:+10.2f} {hc:+9.2f} {sw:11.3f} {mv:7.2f} {ar:8.2f}  {v}")
```

### Pass / fail

| metric | current (fail) | target |
|---|---|---|
| Dominant knee frequency | 5.12 Hz | **1.0–3.0 Hz** |
| Knee L/R correlation | +0.91 on hw | **≤ −0.3** (antiphase = stepping) |
| Knee swing amplitude | 0.775 rad | **0.15–0.5 rad** (>0.05 = not collapsed) |
| Peak joint velocity | 8–10 rad/s | **< ~5 rad/s** |
| Σ(Δa²) | 4.35 | **< ~1.5** |

A run **fails** if knee swing < 0.05 rad (collapsed into standing) *or* gait frequency > 3.5 Hz
(still unexecutable). Note the policy commands **no stepping below ~0.4 m/s** — a deadband, not a
bug — so screen at vx ≥ 0.4 and ignore a flat result at 0.3.

---

## 6. What to report back

For each of A, B, C:

1. Run directory + final `mean_reward` + final `mean_episode_length`.
2. The full `screen_gait.py` table.
3. One-line verdict: GOOD / too fast / collapsed.

Then recommend one config for the **full** profile. If B is good and C collapsed, the useful band
is between them — consider a fourth fast run at the midpoint before spending 30 h.

If **all three** collapse or all three stay above 3.5 Hz, stop and report: the weights are not the
right lever and the reward structure needs revisiting (e.g. add an explicit gait/air-time term
rather than only penalising motion).

---

## 7. Before the full run

- Confirm `_CONTRACT_EFFORT` (§3) was applied to **all three** sweep runs. If it was not, the
  screening results were produced on a different plant than the robot and do not transfer — rerun.
- Consider exporting `action_limit_lower/upper` into `leg_policy_contract.json`. They currently
  live only in `policy_latest.yaml`, so the runtime contract had to be updated by hand.


---

## 8. Sweep results — 2026-08-25

All three runs used `--profile fast` (4096 envs x 24 steps x 6000 iters), gains `tuned` 45/1.5,
and the §3 effort limits (knees 11.0 Nm), applied before run A. Weights were selected via
`HUMANOID_SMOOTH_PRESET` rather than hand-edited, so each run's weights are printed in its log.

| | baseline (deployed) | **A** light | **B** moderate | **C** strong | target |
|---|---|---|---|---|---|
| run dir | `2026-08-18_20-45-17` | `2026-08-24_19-52-36_smoothA` | `2026-08-24_17-33-47_smoothB` | `2026-08-24_21-57-18_smoothC` | |
| gait Hz | 5.00-5.31 | 2.50-3.12 | 0.62-1.69 | collapsed | 1.0-3.0 |
| knee L/R corr | -0.58/-0.67 | **-0.89/-0.98** | -0.29/-0.93 | n/a | <= -0.3 |
| knee swing (rad) | 0.61-0.87 | 0.159-0.670 | 0.092-0.590 | **0.000** | 0.15-0.5 |
| peak abs(v) | 15.2-17.8 | 2.48-7.00 | 1.01-7.98 | 0.00 | < 5 |
| S(da^2) | 4.18-5.49 | 1.87-2.24 @speed | **0.35-0.77** | 0.00 | < 1.5 |
| mean_episode_length | 441.9 | 411.6 | 402.3 | 356.7 | higher better |
| mean_reward | 13.50 | 10.56 | 9.60 | 6.94 | *not a criterion* |
| verdict | FAIL (unexecutable) | in-band | in-band | **COLLAPSED** | |

**C collapsed outright** — swing 0.000 rad at every speed, while scoring perfectly on every
smoothness metric. This is precisely the failure mode §2 warns about and the reason the screening
metric is gait frequency, not reward.

**Neither A nor B is cleanly correct.** Each clears the §5 pass/fail bar but fails a different
secondary target:
- **A** fails action rate at speed (S(da^2) 2.24 vs <1.5) — the quantity tied most directly to the
  torque demand and the encoder fault that stopped the hardware run.
- **B** fails gait phase at vx=0.40 (knee corr -0.29, swing 0.092 rad) — thin exactly inside the
  range the robot gets commanded through.

Both are 2-4x better than the baseline on frequency and cut peak velocity by more than half, so the
training lever works; the §3 warning also held — the weights that were needed are well above the
pre-flash intuition.

**Action:** per §6, a fourth fast run **D** at the A/B midpoint (`-0.075 / -0.0035 / -2e-4`) before
committing ~30 h to a full run. The cliff is between B and C, so the A-B interval is the safe place
to search.


---

## 9. Run D + full velocity sweep — the decision (2026-08-25)

Run D (`2026-08-25_00-06-03_smoothD`, midpoint -0.075 / -0.0035 / -2e-4) did **not** behave as an
interpolation between A and B. Its training metrics land exactly between them (reward 10.24,
ep_len 406.5) yet it does not step at all below 0.7 m/s. **The weight -> gait mapping is not
monotonic, and reward/episode-length do not predict it.** Judge only on the screener.

Re-screening every policy across vx = 0.3 ... 1.0 (rather than the original 0.3-0.6) is what
actually settled the choice:

**Knee swing (rad)** — >0.15 stepping properly, <0.05 not stepping

| run | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 1.0 |
|---|---|---|---|---|---|---|---|
| baseline | 0.000 | 0.606 | 0.871 | 0.848 | 1.071 | 1.319 | 1.623 |
| **A** | **0.159** | **0.419** | **0.621** | **0.670** | **0.707** | **0.843** | **0.868** |
| B | 0.139 | 0.092 | 0.390 | 0.590 | 0.661 | 0.736 | 0.699 |
| C | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.059 |
| D | 0.000 | 0.000 | 0.033 | 0.040 | 0.747 | 0.802 | 0.847 |

**S(da^2)** — target < 1.5

| run | 0.4 | 0.5 | 0.6 | 1.0 |
|---|---|---|---|---|
| baseline | 4.18 | 4.68 | 5.49 | 13.31 |
| A | 1.37 | 1.87 | 2.24 | 3.15 |
| B | 0.07 | 0.35 | 0.77 | 0.57 |

- **C** is genuinely dead everywhere.
- **D**'s deadband moved out to 0.7 m/s — for a robot commanded at 0.4-0.6 that is worse than a
  collapse, because it looks healthy on every training metric.
- **B** dips to 0.092 rad at vx=0.40, *below* its own 0.3 value. Non-monotonic, i.e. unstable
  exactly inside the commanded band.
- **A** is monotonic and clean at every speed, knee corr -0.84 to -0.98 throughout, and holds the
  highest episode length of the penalised runs (411.6).

**Selected: A.** It is the only policy that steps reliably across the whole commanded range. Its
one miss is action rate above vx=0.4 (1.87 / 2.24 vs <1.5) — still a 2.5x reduction on the
baseline that failed on hardware, alongside a 2x frequency reduction. B's smoothness advantage is
bought by not reliably stepping at 0.4, and a policy that will not step in the commanded range
cannot be rescued by being smooth. The midpoint gamble already failed once (D), so no further
interpolation was attempted.

### Full run launched

    logs/rsl_rl/biped/2026-08-25_02-13-41_smoothA-full
    HUMANOID_SMOOTH_PRESET=A   -0.05 / -0.002 / -1e-4   (verified in params/env.yaml)
    HUMANOID_GAIN_PRESET=tuned  kp=45.0 / kd=1.5        (verified)
    knee effort_limit 11.0                              (verified)
    --profile full: 24576 envs x 64 steps x 6000 iters, NO --plateau

No early stop: it matches how all four sweep runs were scored, and `--plateau` keys on reward,
which sec 2 explicitly rules out as a criterion here.

⚠️ **The full run may not reproduce A's gait.** It uses 6x the batch (24576 vs 4096 envs), so
gradient noise differs; D already showed this reward landscape is not smooth. Screen the result
with `screen_gait.py` across vx = 0.3-1.0 before exporting anything to `deploy/walk`.


---

## 10. ⚠️ Correction — the ideal-closed-loop screener is not trustworthy (2026-08-26)

The full run (`2026-08-25_02-13-41_smoothA-full`, preset A) completed 6000 iterations with the
best training metrics of any walk run to date (reward 17.27, ep_len 476.9 vs the baseline's
13.50 / 441.9). `screen_gait.py` then reported it **COLLAPSED at every speed** — zero motion on
all 12 joints.

That verdict was **wrong**. Measured in Isaac with `eval_plant_compare.py`:

| | fwd speed (cmd 0.5) | joint_vel_rms | action_rate_rms | ep_len_s | falls/min |
|---|---|---|---|---|---|
| baseline | 0.500 | 2.145 | 0.480 | 19.69 | 0.234 |
| A (fast) | 0.489 | 1.880 | 0.379 | 19.25 | 0.586 |
| B | 0.480 | 1.675 | 0.332 | 18.82 | 0.586 |
| C | 0.466 | 1.542 | 0.280 | 18.55 | 1.031 |
| D | 0.483 | 1.760 | 0.346 | 18.55 | 0.914 |
| **A-FULL** | **0.485** | 1.840 | 0.357 | **19.54** | **0.141** |

Every policy walks, including C and D, which the screener also called collapsed. In the simulator
the weights behave **monotonically** (smoothness rises with penalty strength, stability falls) —
the non-monotonic mess in §9 was an artifact of the screener, not a property of the reward.

### Why the screener fails

It drives the policy with synthetic observations: `base_ang_vel` pinned to `[0,0,0]`,
`projected_gravity` pinned to `[0,0,-1]`, and joint states from its own first-order integrator
(`track=0.9`). A policy that relies on base-motion feedback to generate gait phase never starts
stepping under those inputs. Fast-A happened to be self-oscillating enough to run open-loop;
the full-profile policy is not. **False "collapsed" verdicts are the expected failure mode.**

### The premise of this document is also affected

§1 lists the policy gait as **5.12 Hz (ideal closed loop)**. Measured in Isaac, the same baseline
policy walks at **1.62 Hz** with knee correlation **-0.83** and 100% of envs stepping — a healthy
antiphase gait. The 5.12 Hz figure comes from the same class of idealised model as the screener.

What remains solid is the **hardware** evidence, which was measured on the robot, not modelled:
knees in phase at **+0.91**, demanded torque up to **47.9 Nm** against a ~26.9 Nm ceiling, and the
encoder fault once the knee had authority to follow. Sim says the gait is fine; hardware says it
is not. **That gap is the actual problem**, and no amount of sim-side reward tuning can be
validated against it from sim alone.

### Standing conclusions

1. `scripts/rsl_rl/eval_plant_compare.py` is now instrumented with `EVAL_GAIT=1` to report
   `gait_hz_median`, `knee_corr_median`, `knee_swing_median` and `envs_stepping_pct` from real
   simulator rollouts. **Use it, not `screen_gait.py`, to judge a gait.**
2. `scripts/screen_gait.py` is retained only as a cheap smoothness probe. Its "COLLAPSED" verdict
   means "did not self-oscillate under synthetic inputs" — NOT that the policy cannot walk.
3. **A-FULL is a real improvement** on the baseline in sim: 26% lower action rate, 14% lower joint
   velocity, and a **40% lower fall rate** (0.141 vs 0.234 /min) at the same tracked speed. It is
   the best candidate to deploy, but whether it fixes the hardware jitter is **unverified and not
   verifiable from sim**.


---

## 11. Hardware result for A-full, and the B-full deployment (2026-08-29)

**A-full was run on the robot. The data was poor and the encoders faulted across the whole leg** —
worse than the earlier single-joint `ERROR_ENCODER_FAULT` seen with the kp45 baseline.

This is the most important measurement in this document, because A-full was **smoother than that
baseline on every simulator metric** and the hardware symptom still got worse:

| | action_rate_rms | joint_vel_rms | falls/min | gait Hz | hardware |
|---|---|---|---|---|---|
| kp45 baseline | 0.480 | 2.145 | 0.234 | 1.62 | jitter, 1 encoder fault |
| A-full | 0.357 | 1.840 | 0.141 | 1.56 | **encoders faulted across the whole leg** |
| **B-full** (deployed) | **0.316** | **1.696** | 0.188 | 1.50 | untested |

Sim-side smoothness is **not** predicting the hardware failure. Two policies now, each smoother
than the last in simulation, each no better or worse on the robot.

### B-full — now in `deploy/walk`

    run        logs/rsl_rl/biped/2026-08-28_00-09-42_smoothB-full-resumed
    checkpoint model_5999.pt
    rewards    HUMANOID_SMOOTH_PRESET=B  -0.1 / -0.005 / -3e-4
    gains      kp=45.0 / kd=1.5;  knees 11.0 Nm
    sim        fwd 0.480 m/s, gait 1.50 Hz, knee corr -0.765, 100% envs stepping,
               action_rate_rms 0.316, joint_vel_rms 1.696, falls 0.188/min

⚠️ Note this run was **resumed from iteration 900** after two machine crashes (source:
`2026-08-26_11-35-01_smoothB-full-CRASHED-iter900/model_900.pt`), whereas A-full ran continuously.
The optimizer state was restored and reward resumed at 9.394 vs 10.303 pre-crash, so the effect
should be small — but it is a real difference between the two runs.

### If B-full also faults the encoders

Then the sim-side smoothness lever is **exhausted** and the cause is not policy smoothness. Three
runs would have shown monotonically better sim smoothness with no hardware improvement. The next
place to look is hardware/contract, not reward weights:

- encoder mounting and I2C signal integrity under load (the fault is an *encoder* fault, not a
  tracking failure);
- whether raising the knee cap 6 -> 11 Nm removed the accidental filter that was protecting the
  encoders — the fault first appeared *after* that flash (sec 3);
- the unresolved **hip_pitch sim<->hardware sign inversion** from
  `docs/walk-policy-divergence-report.md`, still open;
- the sim-to-real contract itself, given sim reports a healthy antiphase gait while hardware shows
  knees in phase at +0.91.
