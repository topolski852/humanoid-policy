# Walk smoothness sweep — 3 fast-profile runs

**Status:** ready to run. **Owner:** training PC. **Created:** 2026-08-24.
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
