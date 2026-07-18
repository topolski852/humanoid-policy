# Proactive World-Model Walk Policy — Design

**Branch:** `proactive-world-model`  ·  **Status:** design / phase 0

## Motivation

The current walk policy is **reactive** (model-free PPO: obs → action). Observed failure:
when the robot is held or perturbed it *overreacts* — "kicks and lashes out" — because a pure
feedback net has no forward model; it slams a correction without knowing whether the
correction achieves anything. Humans walk smoothly because we carry an **internal model of
our body** (learn how the limbs move first, then walk). We want the robot to do the same:
**predict, then act** — anticipatory instead of reactive → smoother, more human, more robust.

Dual goal: (a) get the robot walking well; (b) be on the forefront of R&D (investor-facing).

## Approach: TD-MPC2 (chosen)

Model-based RL. Chosen over the alternatives:
- **PlaNet** (encoder + latent dynamics + reward + CEM planner — the literal 4-component list):
  no value function → short-horizon **myopic** planning, unstable for balance. Rejected.
- **DreamerV3**: powerful but heavier than we need, and deploys a **reactive** actor only.
- **TD-MPC2**: the 4 components **+ a terminal value function + a policy prior**, with an MPPI
  planner. The value function supplies return-to-go beyond the planning horizon (what keeps a
  walker from stepping off a cliff at the horizon edge). Lighter than Dreamer, SOTA on
  high-DOF continuous control, released PyTorch code. Critically, it supports **either**
  deploy mode (plan online **or** ship the fast policy prior), so we can defer that decision.

### Components (map to the user's spec)
| user term | TD-MPC2 | role |
|---|---|---|
| latent encoder | `z = h(obs)` | encode 45-dim proprio → latent |
| transition model | `z' = d(z, a)` | **body dynamics** — "learn how the limbs move" |
| reward predictor | `r = R(z, a)` | predicted reward |
| simple planner | MPPI over the model | anticipatory action selection |
| (added) value | `Q(z, a)` | return beyond the horizon — anti-myopia |
| (added) policy prior | `π(z)` | warm-starts the planner / fast reactive deploy |

**Train the world model in the accurate sim we just built** (bench actuator models + corrected
12.61 kg mass + DR + pushes). The transition model then learns the *real* actuator dynamics
(friction, latency) — the proactive model literally anticipates what the reactive policy tripped on.

## Integration with this repo

- Reuse the Isaac Lab biped walk env (45 obs / 12 act, modeled plant, DR + harder commands).
- **Adopt-and-adapt** the official TD-MPC2 (PyTorch), not from-scratch. Main work = wrap the
  vectorized Isaac Lab env for data collection (off-policy replay vs Isaac's massive
  parallelism is the key design point).
- Keep the 45-obs/12-act contract so a distilled reactive policy can drop into the existing
  deploy/export path unchanged.

## Success metric (already have the tooling)

`scripts/rsl_rl/eval_plant_compare.py` already measures the exact overreaction signals:
**`action_rate_rms`, `joint_vel_rms`, `base_accel_rms`** (+ fall rate, tracking). The win
condition is: **smoother (lower action_rate / joint_vel) than the PPO policy at equal or
better fall rate + tracking** — a quantified, before/after "less wobble" story.

## Phased roadmap (investor-visible milestones)

- **P0 — scaffold**: adopt TD-MPC2, build the Isaac Lab env adapter, smoke-train on the walk
  task. → *the pipeline runs end-to-end.*
- **P1 — world model**: encoder + dynamics + reward + value learn walking; low transition
  prediction error; policy prior walks. → *"the robot has an internal body model."*
- **P2 — planning + smoothness**: enable MPPI; tune anticipatory smooth gait; beat PPO on
  action_rate / joint_vel / falls. → *side-by-side "reactive vs proactive" demo clip.*
- **P3 — deploy**: decide plan-on-robot vs distilled reactive policy (needs robot compute
  spec); export; hardware test.

## Open items (gate later phases)

- **Robot compute spec** (Jetson? Pi-class? MCU only?) — hard-gates whether online planning at
  25 Hz is feasible, which decides P3.
- Off-policy replay vs 24k-env parallel collection — data-pipeline design in P0.
- Optional: a quick scan of late-2024/2025 MBRL-for-humanoid work to confirm TD-MPC2 is still
  the frontier before committing the build.
