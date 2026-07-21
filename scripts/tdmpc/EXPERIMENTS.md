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
| 7 | 07-21 | **STAND phase** — clean stand reward, cmd=0, calm spawn | from scratch, latent512, UTD 16, compile | pending | warm-start walk from it |

### feet_air_time is GAMEABLE (run 6) — critical
`feet_air_time_positive_biped` rewards the swing foot being airborne; it is NOT gated by
uprightness, so a robot lying on its back with a leg up farms it. Fix for the WALK phase: gate it by
the uprightness (multiply by stand_gate / only reward when upright), or only credit air-time when
base height is near standing. For the STAND phase we simply drop it (nothing to farm by falling).

## Findings (why each reward change was made)
- **Cold-start problem (run 5):** from scratch the multiplicative `standing×upright` gate is ~0 with a near-flat gradient when not upright → no signal to stand up → crouch. Fix: small *ungated* `upright_posture` term (smooth gradient to vertical) + keep the stand warm-start.
- **No gait incentive (runs 3–4):** pure velocity reward specifies the goal, not the stride. Leaning/wobbling fakes some velocity. Fix: `feet_air_time` (reward foot-lift) + `feet_slide` (punish drag) — the proven PPO terms.
- **Stability / IMU (run 6 rationale):** the deployed PPO walk worked because it prioritized stability; we'd stripped `ang_vel_xy`/`flat_orientation`/`base_accel`/smoothness. Re-added (modest) for low IMU noise + smooth sim-to-real. Caution: keep modest so they don't suppress the gait into standing.
- **Sample budget:** official TD-MPC2 default is 10M steps/task (not the 2M I'd mis-cited); PPO needed 590M. 1M was ~10% — under-trained.

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
