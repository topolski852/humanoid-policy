# Walk Policy Divergence — Root-Cause Report & Retraining Plan

**Task:** `Walk-Humanoid-Policy-Biped-v0` (deploy: `deploy/walk/`)
**Date:** 2026-07-14
**Author:** hardware bring-up + control-stack analysis (humanoid-control) → this report is written for review/iteration by an LLM before retraining.

---

## 0. TL;DR — priority-ordered actions

The control/deployment stack is **verified end-to-end** (joint mapping, ONNX inference, observation assembly all bit-exact on hardware — see §2). The walk policy nonetheless **diverges within ~0.2 s** on the real robot: actions grow unboundedly, joint targets saturate the position limits, joints thrash at 12 rad/s. **The dominant cause is a sim↔real actuator (PD-gain) mismatch, not the reward.** Fix in this order:

1. **[HIGHEST — decided & config-applied] Train walk on the real firmware gains** (§3-H1, §4A). Walk trained at uniform `kp=20, kd=2`; the real ESCs run **asymmetric** gains up to `kp=68.4, kd=9.8` (hip_pitch), confirmed live. This alone is sufficient to explain the runaway. **Decision: match the policy to the (commissioned) plant, not vice-versa** — forcing the motors to 20/2 risks a hip that can't hold the trunk. The config change is applied in this commit (walk now uses `HUMANOID_BIPED_WALK_CFG` with the contract gains; one stale kd corrected). **Retraining + deploy re-export still required.**
2. **[HIGH] Verify / resolve the hip_pitch sim↔hardware sign inversion** flagged unresolved in code (§3-H2).
3. **[HIGH] Bound the action** — add action clipping/squashing (train + deploy) and a real action-magnitude penalty; the policy currently emits raw actions of ~10 → **149° target offsets** (§3-H3, §4B). Deploy `action_limit = ±10000` (unbounded).
4. **[MED] Add actuator delay/latency + in-episode push randomization** for closed-loop robustness margin (§4A/§4C).
5. **[MED] Reward shaping** for smoothness (raise `action_rate_l2`, add joint-velocity penalty), and tighten termination (§4B).

**Ordering rationale:** reward shaping tunes behavior *within a given plant*. If the training plant (kp=20) ≠ deploy plant (kp=68, asymmetric), no reward weighting will make the sim-trained policy stable on hardware. Fix plant fidelity first; then shape.

---

## 1. Symptom & measured evidence (hardware)

Data was captured with a per-tick recorder added to the control runtime (`humanoid-control/humanoid_control/recorder.py`), logging **every 25 Hz policy tick**: the exact 45-dim observation fed to the ONNX, the raw action out, the mapped joint targets, measured joint pos/vel, and the IMU base state. One walk run was captured — **84 frames / 3.36 s at a clean 25.0 Hz** — before the left CAN bus browned out (see §5, note on hardware).

### 1.1 Inference path is exact (rules out wiring/mapping)
Replaying the **recorded observations** back through `deploy/walk/policy.onnx` offline reproduces the **recorded actions exactly**:

```
max | onnx(recorded_obs) − recorded_action |  =  0.00e+00   (all 84 frames)
```

⇒ observation assembly, joint ordering, sign convention, and onnxruntime inference on the robot are all correct. The problem is **not** in deployment plumbing.

### 1.2 The failure is a runaway divergence
Measured trajectory (subsampled; `|action|` = max-abs over 12 joints; `prev_action` = max-abs of the last-action obs block; `jvel` = max-abs joint velocity, rad/s; `pinned` = # joints whose target hit a position limit):

| t (s from 1st tick) | prev_action | → \|action\| | joint-dev (rad) | \|jvel\| (rad/s) | pinned /12 |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.00 | 1.63 | 0.37 | 5.8 | 0 |
| 0.24 | 4.82 | 2.90 | 0.42 | 7.9 | 2 |
| 0.96 | 3.36 | 4.06 | 0.41 | 2.8 | 1 |
| 1.44 | 6.53 | 4.41 | 0.56 | 7.3 | 2 |
| 1.68 | 6.21 | **8.92** | 0.68 | **12.0** | 3 |
| 2.16 | 3.93 | 3.73 | **1.45 (pinned)** | 8.8 | 0 |
| 2.88 | 6.57 | 6.27 | 1.45 | 8.8 | 2 |

Summary statistics over the run:
- `|action|`: start **1.63** → peak **10.42**. With `action_scale = 0.25`, peak action → **2.60 rad = 149° target offset** from default pose (hard-clamped by joint limits downstream).
- Time to `|action| > 3`: **0.20 s** after the first policy tick (near-instant divergence).
- Joint velocity: mean **6.6**, peak **12.0 rad/s**.
- Joints pinned at a position limit: mean **1.8/12**, peak **6/12**.
- **`corr(|base_ang_vel|, |action|) = −0.05`** (negligible) — divergence is *not* driven by body rotation rate.
- **`corr(prev_action, action) = +0.79`** (strong) — the action explosion is driven by the policy's **own action-history feedback** (the `last_action` obs term).

### 1.3 Handoff is not at rest
On the **first** policy tick the joint deviation from default is only 0.37 rad (the ramp placed it near `default_pose`), **but joint velocity is already 5.8 rad/s.** The policy takes over a moving plant and is out-of-distribution almost immediately.

### 1.4 Static-pose corroboration (separate capture)
Under a zero-policy hold (no balancer) at `default_pose`, the IMU reads **~+10° forward pitch** at the clean instant and the robot topples to **+34°** — i.e. `default_pose` is CoM-forward / not statically stable, and the nominal-input policy probe (obs = default pose, upright, zero vel/cmd/prev) already commands **~+12° ankle** offset. So `default_pose` is not the policy's equilibrium (see §3-H6). Secondary to the divergence, but relevant to reward/pose design.

---

## 2. Verified — do NOT re-debug these

- **Joint order / frame mapping** (control canonical order ≡ ONNX order): proven at the file level — `default_joint_positions[action_indices]` equals the canonical-order `default_pose` exactly, and confirmed behaviorally by the exact replay (§1.1).
- **ONNX inference on-robot**: bit-exact replay (§1.1).
- **Observation assembly** (45-dim, the layout in §3 obs table): exact.
- **Calibration / encoder zeros**: joint tracking is faithful (hip/knee hold targets in the zero-policy hold; only the whole-body topple moves them).

The remaining problem space is entirely **policy stability / sim-to-real dynamics**.

---

## 3. Root-cause analysis (ranked)

### H1 — Actuator PD-gain mismatch, sim vs. real  ★ primary
The walk task loads `HUMANOID_BIPED_CFG` (`source/humanoid_policy_assets/.../robots/humanoid.py:41-103`), which uses **`ImplicitActuatorCfg` with uniform `stiffness=20, damping=2, effort_limit=6`** on all 12 leg joints. The real ESC firmware gains (device-truth, pulled from hardware) are **asymmetric and much stiffer**:

| joint | sim kp/kd/eff | **real kp/kd/eff** |
|---|---|---|
| hip_pitch (L/R) | 20 / 2 / 6 | **68.4 / 9.8 / 9.5** |
| knee_pitch (L) | 20 / 2 / 6 | 27.0 / 2.45 / 6 |
| hip_yaw | 20 / 2 / 6 | **10.5 (L) vs 20 (R)** / 0.5 vs 1.0 |
| ankle_pitch | 20 / 2 / 6 | 18 (L) vs 20 (R) / **2.0 (L) vs 0.5 (R)** |

**Mechanism:** a position-target policy learns targets calibrated to the training plant's closed-loop response `τ = kp·(target − q) − kd·q̇`. Deployed on a hip_pitch that is **3.4× stiffer (kp 20→68) with 4.9× the damping**, the same target error produces ~3.4× the torque. The effective loop gain the policy sees at runtime is far higher than trained → overshoot → the measured runaway (§1.2), amplified by the strong `prev_action` self-feedback (§1.2, corr 0.79). The L/R **asymmetry** additionally breaks the mirror symmetry the policy assumes.

Note the repo already contains the correct gains: `HUMANOID_BIPED_SQUAT_CFG` (`humanoid.py:269-297`, used by the standup/squat task) applies the per-joint contract gains `_CONTRACT_KP/KD/EFFORT` (`humanoid.py:238-255`). **The walk task simply does not use them.**

**Confirmed (2026-07-14):** three independent sources agree the ESCs run the asymmetric ~68 set, not uniform 20/2 — (a) `humanoid-studio/configs/humanoid_lite.json` (`HUMANOID_CONFIG`, what the daemon writes to the motors: `config_loader.cpp:78` → `robot.cpp:923 write_gains`), (b) the device-truth pull in `humanoid-control/configs/leg_policy_params.json`, (c) the web service's live `read_device_config` on connect (`service.py:307`). The deploy yaml's `joint_kp: 20` is the *sim/policy* assumption; the physical plant is the commissioned asymmetric set. That train(20)/deploy(68) gap is the driver.

### H2 — hip_pitch sign inversion sim↔hardware  ★ verify
`humanoid.py:210-217` explicitly documents hip_pitch as **inverted between the USD and the hardware**, flagged as "an open reconciliation item" affecting walk. A wrong hip_pitch sign would command the pitch DOF backwards and diverge immediately. *Counter-evidence:* the ramp to `default_pose` produced a sane pose (10° lean, not a gross fold), and replay is exact — so a full command-level inversion is unlikely to be unhandled. **But** given it is flagged unresolved, verify directly: command +Δ hip_pitch on the real robot and confirm the joint moves the same direction as sim.

### H3 — Unbounded actions + weak smoothness regularization
- **No action clipping/squashing.** `JointPositionActionCfg` (`env_cfg.py:97-107`) has no `clip`; deploy `action_limit = ±10000` (`deploy/walk/policy_latest.yaml`). The Gaussian policy head (`init_std=1.0`) can and does emit raw actions ~10 (§1.2) → 149° target offsets.
- **`action_rate_l2 = −0.01`** (`env_cfg.py:150`) is weak; there is **no action-magnitude penalty** and **no joint-velocity penalty** in the reward (§4B). Nothing strongly discourages the high-frequency, large-amplitude action sequences seen on hardware.

This is not the root cause (H1 is), but it removes the guardrails that would have kept a mismatched policy from slamming the joints to their limits at 12 rad/s.

### H4 — No latency / actuator-delay modeling
`ImplicitActuatorCfg` is instantaneous PD; there is **no actuator delay, no first-order lag, no control/observation latency** anywhere (§ config map). Real actuation + CAN + firmware introduce delay; a policy trained delay-free is prone to overshoot on a delayed plant — compounding H1.

### H5 — No in-episode disturbance robustness
`push_robot` is **commented out** (`env_cfg.py:310-315`); only a one-time ±2 N reset force exists. The policy never learned to reject in-episode disturbances, so the handoff transient (§1.3, 5.8 rad/s initial joint velocity) and any hand contact push it straight out of distribution.

### H6 — `default_pose` is CoM-forward / not the policy's equilibrium
Measured: `default_pose` sits ~10° forward and is statically unstable; the nominal policy probe wants +12° ankle (§1.4). Not a divergence driver on its own, but it means the ramp hands the policy a pose it immediately wants to leave, and it interacts with H1 (the correction is applied through a mismatched actuator).

### H7 — Observation robustness
`joint_vel` obs noise is **±2.0 rad/s** (`env_cfg.py:74`) — large; `obs_normalization = False` (raw obs); no obs history; no obs clipping. Real velocity estimates that differ in *distribution* (not just magnitude) from uniform training noise degrade differently than trained. Lower priority than H1–H3.

---

## 4. Recommended changes

### 4A. Sim-fidelity (highest impact — do first)
1. **Use the real per-joint gains for walk — APPLIED in this commit.** Added `HUMANOID_BIPED_WALK_CFG` (`humanoid.py`): same contract-gain actuators as `HUMANOID_BIPED_SQUAT_CFG` (`_CONTRACT_KP/KD/EFFORT`, incl. L/R asymmetry) but keeping the walk standing init pose; the walk task (`velocity/config/biped/env_cfg.py:359`) now loads it instead of `HUMANOID_BIPED_CFG` (uniform 20/2). **Also fixed one stale value:** `_CONTRACT_KD` right_ankle_pitch was `0.5`, corrected to **`0.25`** to match the current `humanoid_lite.json` device truth (all other 11 joints already matched exactly). **Action required:** retrain the walk policy, then re-export `deploy/walk` (the exported `joint_kp`/`default_joint_positions`/contract will then carry the real gains, making sim↔deploy↔hardware consistent).
2. **Randomize actuator gains around the real values**, not around 20. Change `scale_all_actuator_torque_constant` (`env_cfg.py:262`) so the *center* is the contract gains and the range spans real uncertainty (e.g. ±30–50%). This is the single most important robustness lever.
3. **Add actuator/control delay.** Model a 1–3 tick (40–120 ms) action delay and/or a first-order actuator lag; randomize it. (Isaac Lab `DelayedPDActuator` or an action-delay event.)
4. **Resolve H2** (hip_pitch sign) at the asset level so sim and hardware agree; re-export the contract after.

### 4B. Reward changes (the requested focus)
Current `RewardsCfg` (`env_cfg.py:110-211`) and proposed edits:

| term | current weight | proposed | rationale |
|---|---:|---:|---|
| `action_rate_l2` | −0.01 | **−0.05 → −0.1** (sweep) | penalize jerk/high-freq oscillation (§1.2); primary reward lever |
| **`action_l2` (NEW)** | — | **−0.005 → −0.02** | direct penalty on raw action magnitude; keeps output near 0, complements clipping (H3) |
| **`dof_vel_l2` (NEW)** | — | **−1e-4 → −5e-4** | penalize the 12 rad/s thrashing (§1.2); currently only `dof_acc_l2=−1e-6` exists |
| `dof_torques_l2` | −2e-3 | −3e-3 → −5e-3 | modest increase; discourages saturating the (now-correct) effort limits |
| `flat_orientation_l2` | −2.0 | keep (maybe −2.5) | already meaningful; ties to the CoM-forward pose (H6) |
| `termination_penalty` | −10.0 | keep | fine once terminations fire earlier (below) |

Also:
- **Action clipping/squash (H3):** add `clip=(-1.0, 1.0)` to `JointPositionActionCfg` (or a tanh-squashed action head) at train time, and set the deploy `action_limit` to the same bound (currently ±10000). This caps target offsets at `±0.25 rad ≈ ±14°` regardless of policy output — a hard safety guardrail and a training regularizer. Retrain with it (do not just clip at deploy — that shifts the deployed distribution).
- **Tighten termination:** `bad_orientation limit_angle = 0.78 rad (~45°)` (`env_cfg.py:222`) is the *only* fall condition. Add a **base-height** termination and/or reduce the angle so bad episodes end before the policy learns to thrash; consider an early-termination shaping so divergence is penalized, not explored.

**Caveat (state this in review):** these reward edits improve smoothness/robustness but will **not** stabilize the policy on hardware while H1 (gain mismatch) stands. Sequence 4A before 4B.

### 4C. Domain randomization additions (`EventsCfg`, `env_cfg.py:228-315`)
- **Re-enable `push_robot`** (currently commented, :310-315) with a modest interval (e.g. every 5–10 s, ±0.3–0.5 m/s velocity kick).
- Add **CoM / per-link mass randomization** (currently only base mass ±(−1,+2) kg; no CoM). Ties to H6.
- Add **observation-latency** and **action-latency** randomization (pairs with 4A-3).
- Consider raising `base_external_force_torque` from ±2 to ±3 N·m (a ±3 variant is already noted in-code).

### 4D. Observation / deploy
- Consider enabling **`obs_normalization`** (currently `False`, `rsl_rl_ppo_cfg.py`) or hard obs clipping, so the recorded 149°-equivalent excursions can't feed unbounded values back through the `last_action` term.
- Revisit `joint_vel` obs noise (±2.0) — validate it matches the real velocity-estimate error distribution; if hardware velocity is filtered/quantized differently, model that.
- **Fix the handoff transient (§1.3):** ensure the ramp→policy handoff starts from rest (seed `prev_action=0` — already done — and gate policy start on low joint velocity). Reduces immediate OOD.

---

## 5. Retraining & validation plan

**Metrics to track (add to eval):** per-episode max `|action|`, action-rate RMS, max `|joint_vel|`, fraction of ticks with any joint pinned at a limit, and time-to-divergence under a scripted handoff-with-initial-velocity. These are exactly the quantities that exposed the failure (§1.2).

**Staged validation before hardware:**
1. **Sim2sim replay/gain-sweep:** after 4A, run the trained policy in sim while sweeping actuator kp/kd across the real asymmetric values ±uncertainty. The policy must stay bounded across the whole sweep, not just at the nominal training gains. This is the acceptance gate for H1.
2. **Handoff test:** initialize the episode at `default_pose` with a 3–6 rad/s joint-velocity perturbation (matching the measured 5.8 rad/s handoff) and confirm the policy recovers rather than diverging.
3. **Replay parity:** re-run the on-hardware replay test (feed recorded obs → ONNX → compare to recorded action) after every re-export; it must stay ≈0 (it does today).
4. **Hardware, supported, short:** re-run with the per-tick recorder on; confirm `|action|` stays bounded (target: peak `|action| < 2`, no joints pinned, `|jvel|` < velocity limit). Only then test unsupported.

**Hardware note (not a policy bug):** the capture ended when all six **left-leg** joints dropped offline simultaneously (err=0 = bus/power event, not a motor fault) during the max-velocity thrash. The violent, limit-pinned motion (peak current) likely browned out the left CAN bus/adapter. A bounded policy (post-fix) will reduce this stress, but the bus's tolerance to peak current is a separate reliability item worth addressing (adapter power, bus voltage under load).

---

## Appendix A — measured data provenance
- Recording: `humanoid-control/recordings/run_1784071302112924227_28540.jsonl` (84 frames, 25.0 Hz, walk policy, robot hand-supported).
- Recorder: `humanoid-control/humanoid_control/recorder.py` (opt-in via `HUMANOID_RECORD_DIR`; logs `{t, base_ang_vel, projected_gravity, joint_pos, joint_vel, obs(45), action(12), targets(12), command}` per `PolicyRunner.step()`).
- Replay/analysis used `deploy/walk/policy.onnx` via onnxruntime; `default_pose`, `action_scale=0.25`, and joint position limits from `humanoid-control/configs/leg_policy_params.json` (device-truth ESC pull).

## Appendix B — offline policy probes (nominal, no hardware)
Feeding controlled synthetic observations to `deploy/walk/policy.onnx`:
- **Nominal** (default pose, upright, zero vel/cmd/prev): `|action| = 0.85`, biased toward **+12° ankle** (equilibrium ≠ default pose; H6).
- **prev_action loop, state frozen at nominal:** converges to a fixed point (|action| ~0.82) — the policy is stable when the *state* is held still; it only diverges when the plant moves (supports H1: plant-driven instability, not intrinsic self-oscillation).
- **Feedback gains** (Δ|action| per unit perturbation): pitch **~2.8 /rad**, base ang-vel **~0.5 /(rad/s)**, joint-vel **~0.05 /(rad/s)**. Note the real-run action magnitudes (4–10) far exceed what these near-nominal gains predict → the run operates far outside the trained distribution, consistent with H1/H3.

## Appendix C — key config references
- Rewards: `source/humanoid_policy/humanoid_policy/tasks/locomotion/velocity/config/biped/env_cfg.py:110-211`
- Actions: `env_cfg.py:97-107`  ·  Observations: `env_cfg.py:52-82`  ·  Events/DR: `env_cfg.py:228-315`  ·  Terminations: `env_cfg.py:214-225`  ·  Timing (decimation=8 → 25 Hz): `env_cfg.py:355`
- Actuators: `source/humanoid_policy_assets/humanoid_policy_assets/robots/humanoid.py` — walk `HUMANOID_BIPED_CFG:41-103` (kp=20/kd=2), contract gains `_CONTRACT_KP/KD/EFFORT:238-255`, squat cfg using them `:269-297`, hip_pitch inversion note `:210-217`.
- PPO: `.../config/biped/agents/rsl_rl_ppo_cfg.py` (MLP [256,128,128] elu, lr 1e-3 adaptive, entropy 0.008, `obs_normalization=False`, `init_std=1.0`, 750 iters, 16384 envs).
- Deploy contract: `deploy/walk/policy_latest.yaml` (`joint_kp:20`, `action_limit:±10000`, `policy_dt:0.04`).
