"""TD-MPC2 agent cfg for the biped walk task.

Registered via ``tdmpc_cfg_entry_point`` on the same gym env as the PPO cfg (additive).
"""

from isaaclab.utils.configclass import configclass

from humanoid_policy.tdmpc.config import TdmpcAgentCfg


@configclass
class HumanoidBipedTdmpcCfg(TdmpcAgentCfg):
    experiment_name = "tdmpc_biped"
    horizon = 5   # longer balance lookahead (was 3) — helps the planner see falls coming
