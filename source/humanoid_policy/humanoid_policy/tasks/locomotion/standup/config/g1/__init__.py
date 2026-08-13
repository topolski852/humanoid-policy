import gymnasium as gym

from . import env_cfg, squat_env_cfg, agents

##
# Register Gym environments.
##

# Squat -> stand: rise from the captured safety squat to a standing pose.
gym.register(
    id="Standup-Humanoid-Policy-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg.G1StandupEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.G1StandupPPORunnerCfg,
    },
)

# Stand -> squat (controlled descent): the safety framework's safe-stop motion.
gym.register(
    id="Squat-Humanoid-Policy-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": squat_env_cfg.G1SquatEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.G1SquatPPORunnerCfg,
    },
)
