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

    # --- model size (START SMALL) ---------------------------------------------
    latent_dim: int = 256
    enc_dim: int = 256
    mlp_dim: int = 256
    num_q: int = 5                       # Q-ensemble size
    horizon: int = 3                     # model rollout / planning horizon

    # --- value distribution (two-hot) -----------------------------------------
    num_bins: int = 101
    vmin: float = -10.0
    vmax: float = 10.0

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
    use_privileged_critic: bool = True   # feed clean 48-dim critic obs into Q during training
    use_tdmpc2_square: bool = False      # TD-M(PC)² policy-regularization (buffer stores plan_mean/std regardless)

    # --- checkpoint / eval cadence --------------------------------------------
    save_interval_steps: int = 50_000
    log_interval_steps: int = 1_000
