import gymnasium as gym

from . import env_cfg, env_cfg_tdmpc, agents

##
# Register Gym environments.
##

gym.register(
    id="Walk-Humanoid-Policy-Biped-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg.HumanoidBipedEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.HumanoidBipedPPORunnerCfg,
        "tdmpc_cfg_entry_point": agents.tdmpc_cfg.HumanoidBipedTdmpcCfg,
    },
)

# TD-MPC2-only variant with the stability-GATED reward (env_cfg_tdmpc). The PPO task above is
# unchanged; this one carries a different reward and is what the TD-MPC2 trainer targets.
gym.register(
    id="Walk-Humanoid-Policy-Biped-Tdmpc-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg_tdmpc.HumanoidBipedTdmpcEnvCfg,
        "tdmpc_cfg_entry_point": agents.tdmpc_cfg.HumanoidBipedTdmpcCfg,
    },
)

# Curriculum phase 1 — STAND (zero command, calm spawn). Warm-start the walk task from its ckpt.
gym.register(
    id="Walk-Humanoid-Policy-Biped-Tdmpc-Stand-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg_tdmpc.HumanoidBipedTdmpcStandEnvCfg,
        "tdmpc_cfg_entry_point": agents.tdmpc_cfg.HumanoidBipedTdmpcCfg,
    },
)
