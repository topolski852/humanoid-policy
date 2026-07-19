"""Config for the TD-MPC2 agent/trainer.

Defaults are the verified upstream TD-MPC2 hypers (config.yaml) EXCEPT the model size, which
starts small (latent/mlp/enc = 256) per the approved plan — 12-DOF proprioceptive walking is
almost certainly enough capacity there; scale to 512 only if it underfits.

An isaaclab ``@configclass`` (not an rsl_rl base) so it loads via the same
``gym.register(kwargs={... "tdmpc_cfg_entry_point"})`` seam as the PPO cfg.
"""

from __future__ import annotations

from isaaclab.utils.configclass import configclass


@configclass
class TdmpcAgentCfg:
    """Hyperparameters for the TD-MPC2 world-model agent."""

    # --- experiment / logging -------------------------------------------------
    experiment_name: str = "tdmpc_biped"
    seed: int = 0

    # --- data collection (LOW vectorization: TD-MPC2 degrades under heavy parallelism) ---
    num_envs: int = 32
    buffer_size: int = 1_000_000        # total transitions (spread across num_envs rings)
    seed_steps: int = 5_000             # env-steps of random actions before learning
    updates_per_step: int = 1           # gradient updates per env-step
    max_env_steps: int = 3_000_000      # total env-steps budget (per-env count × num_envs)
    plan_collection: bool = False       # collect with the MPPI planner (proper TD-MPC2) vs the fast prior

    # --- model size (START SMALL) ---------------------------------------------
    latent_dim: int = 256
    enc_dim: int = 256
    mlp_dim: int = 256
    num_q: int = 5                       # Q-ensemble size
    horizon: int = 3                     # model rollout / planning horizon
    num_enc_layers: int = 2              # encoder MLP depth
    simnorm_dim: int = 8                 # SimNorm group size (latent_dim must be divisible)
    dropout: float = 0.01                # Q-net dropout
    log_std_min: float = -10.0
    log_std_max: float = 2.0
    entropy_coef: float = 1.0e-4

    # --- value distribution (two-hot) -----------------------------------------
    num_bins: int = 101
    vmin: float = -10.0
    vmax: float = 10.0

    # TD-MPC2 acts in normalized [-1,1] action space; the env expects raw JointPosition actions
    # up to ±_ACTION_RAW_LIMIT (4.0 -> ±1 rad offset via action scale 0.25). Map agent→env by ×this.
    act_env_scale: float = 4.0

    # --- MPPI planner ----------------------------------------------------------
    mpc: bool = True
    mppi_iterations: int = 6
    num_samples: int = 512
    num_elites: int = 64
    num_pi_trajs: int = 24
    temperature: float = 0.5
    min_std: float = 0.05
    max_std: float = 2.0

    # --- optimization ----------------------------------------------------------
    batch_size: int = 256
    lr: float = 3.0e-4
    enc_lr_scale: float = 0.3
    grad_clip_norm: float = 20.0
    tau: float = 0.01                    # target-net Polyak
    discount: float = 0.99
    consistency_coef: float = 20.0
    reward_coef: float = 0.1
    value_coef: float = 0.1
    rho: float = 0.5                     # per-step loss discount within the horizon

    # --- design toggles (approved plan) ---------------------------------------
    # Asymmetric privileged critic is DEFERRED: TD-MPC2's value/Q operate on the shared latent z
    # (encode(policy_obs)) and are evaluated on IMAGINED latents inside the MPPI planner, where the
    # 48-dim privileged obs does not exist — so a privileged Q can't feed the planner. First cut is
    # symmetric (45-dim latent for everything). Kept as a flag for a future separate value pathway.
    use_privileged_critic: bool = False
    # TD-M(PC)² (github.com/DarthUtopian/tdmpc_square_public, MIT): regularize the policy prior
    # toward the PLANNER's action distribution (mu,std) that generated the data — fixes the
    # value-overoptimism / policy-collapse that TD-MPC2 shows on higher-DOF bodies. Requires
    # plan_collection so mu/std are available. prior_coef/scale_threshold from their config.yaml.
    use_tdmpc2_square: bool = False
    prior_coef: float = 1.0
    scale_threshold: float = 2.0
    prior_dof_ref: float = 61.0          # their coef is scaled by action_dim/61 (61-DOF HumanoidBench)

    # --- command curriculum (stand -> slow walk -> full walk) -----------------
    # Ramp the velocity command from ~0 up to full as the robot SURVIVES at each level (mean
    # episode length > cmd_survive_frac of the max). Warm-start from a stand so level 0 already
    # holds. Command-magnitude, not step-schedule, so it only speeds up when it's succeeding.
    cmd_curriculum: bool = False
    cmd_scale_start: float = 0.1
    cmd_ramp_step: float = 0.1
    # widen when mean TRAINING ep length > this * max_episode_length. Training survival is
    # noise-limited (planning exploration + DR) below the deterministic policy's, so this is
    # deliberately lenient (0.35 * 500 = 175 steps = 7 s of holding at the current command).
    cmd_survive_frac: float = 0.35
    cmd_ramp_interval: int = 50_000      # env-steps between ramp checks

    # --- checkpoint / eval cadence --------------------------------------------
    save_interval_steps: int = 50_000
    log_interval_steps: int = 1_000
