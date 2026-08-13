from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class G1StandupPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO for the G1 squat -> stand transition.

    Same architecture and hyper-parameters as the biped standup runner -- the task shape is the
    same (short transient, terminal-pose reward) and there is no evidence yet that the wider
    13-DoF action space needs a different setup. Change one thing at a time from this baseline.
    """

    num_steps_per_env = 48
    max_iterations = 3500
    save_interval = 100
    experiment_name = "standup_g1"
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 128],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class G1SquatPPORunnerCfg(G1StandupPPORunnerCfg):
    """Stand -> squat descent: same PPO setup, separate log dir."""

    experiment_name = "squat_g1"
