"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip
import variants  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append training-variant argument (--variant resolves to a gym task id, overriding --task)
variants.add_variant_arg(parser)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# resolve --variant into args_cli.task
variants.resolve_variant(args_cli)
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import itertools
import os
import sys
import threading
import time
import torch

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

from importlib.metadata import version as _pkg_version

from rsl_rl.runners import OnPolicyRunner
from omegaconf import OmegaConf

import isaaclab.utils.string as string_utils
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

# Import extensions to set up environment tasks
import humanoid_policy.tasks  # noqa: F401


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, _pkg_version("rsl-rl-lib"))
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    ppo_runner.alg.actor = torch.compile(ppo_runner.alg.actor, mode="default")
    ppo_runner.alg.critic = torch.compile(ppo_runner.alg.critic, mode="default")
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    ppo_runner.export_policy_to_jit(export_model_dir, filename="policy.pt")
    ppo_runner.export_policy_to_onnx(export_model_dir, filename="policy.onnx")

    # === export the yaml config for deployment ===
    num_joints = len(env_cfg.scene.robot.init_state.joint_pos)

    # we take the joint order defined from the init joint state entry
    joint_names = [name for name in env_cfg.scene.robot.init_state.joint_pos.keys()]
    init_joint_pos = [v for v in env_cfg.scene.robot.init_state.joint_pos.values()]

    joint_kp = torch.zeros(num_joints, device=env.unwrapped.device)
    joint_kd = torch.zeros(num_joints, device=env.unwrapped.device)
    effort_limits = torch.zeros(num_joints, device=env.unwrapped.device)

    def _assign_group_param(target, group, value, joint_names):
        """Write an actuator-group param into ``target`` (indexed by joint), handling both a
        scalar (applies to every joint the group matches) and a per-joint dict (contract gains)."""
        if isinstance(value, dict):
            # dict keys are joint-name expressions -> per-joint values
            indices, _, values = string_utils.resolve_matching_names_values(
                value, joint_names, preserve_order=True
            )
            target[indices] = torch.tensor(values, dtype=target.dtype, device=target.device)
        else:
            match_expr_dict = {expr: None for expr in group.joint_names_expr}
            indices, _, _ = string_utils.resolve_matching_names_values(
                match_expr_dict, joint_names, preserve_order=True
            )
            target[indices] = value

    # extract the configurations from the actuator groups (scalar or per-joint dict gains)
    for group in env_cfg.scene.robot.actuators.values():
        _assign_group_param(joint_kp, group, group.stiffness, joint_names)
        _assign_group_param(joint_kd, group, group.damping, joint_names)
        _assign_group_param(effort_limits, group, group.effort_limit, joint_names)

    # extract the indices of the actuated joints
    match_expr_list = {expr: None for expr in env_cfg.actions.joint_pos.joint_names}
    action_indices, _, _ = string_utils.resolve_matching_names_values(match_expr_list, joint_names, preserve_order=True)

    deploy_config = {
        # === Policy configurations ===
        "policy_checkpoint_path": f"{export_model_dir}/policy.onnx",

        # === Networking configurations ===
        "ip_robot_addr": "127.0.0.1",
        "ip_policy_obs_port": 10000,
        "ip_host_addr": "127.0.0.1",
        "ip_policy_acs_port": 10001,

        # === Physics configurations ===
        "control_dt": 0.004,   # 250 Hz
        "policy_dt": env_cfg.sim.dt * env_cfg.decimation,      # 25 Hz
        "physics_dt": 0.0005,    # 2000 Hz
        "cutoff_freq": 1000,

        # === Articulation configurations ===
        "num_joints": num_joints,
        "joints": joint_names,
        "joint_kp": joint_kp.tolist(),
        "joint_kd": joint_kd.tolist(),
        "effort_limits": effort_limits.tolist(),
        "default_base_position": env_cfg.scene.robot.init_state.pos,
        "default_joint_positions": init_joint_pos,

        # === Observation configurations ===
        "num_observations": env.observation_space["policy"].shape[-1],
        "history_length": env_cfg.observations.policy.actions.history_length,

        # === Command configurations ===
        # sample a command
        "command_velocity": env_cfg.observations.policy.velocity_commands.func(
            env.unwrapped, env_cfg.observations.policy.velocity_commands.params["command_name"]
            )[0].tolist(),

        # === Action configurations ===
        "num_actions": env.action_space.shape[-1],
        "action_scale": env_cfg.actions.joint_pos.scale,
        "action_indices": action_indices,
        "action_limit_lower": -10000,
        "action_limit_upper": 10000,
    }
    if not os.path.exists("configs"):
        os.makedirs("configs")
    OmegaConf.save(deploy_config, "configs/policy_latest.yaml")

    # === export a humanoid-control-compatible policy contract (legs-only) ===
    # Mirrors humanoid-control/configs/leg_policy_params.json so the runtime can load
    # joint_order / obs layout / default_pose / per-joint gains straight from the trainer.
    # Only emitted for the 12-DoF legs policy (the current control target).
    try:
        if int(env.action_space.shape[-1]) == 12:
            import json
            from humanoid_policy_assets.robots.berkeley_humanoid_lite import HUMANOID_LITE_LEG_JOINTS

            # per-joint position limits from the live articulation (sim order -> by name)
            robot = env.unwrapped.scene["robot"]
            sim_joint_names = list(robot.data.joint_names)
            limits_t = getattr(robot.data, "joint_pos_limits", None)
            if limits_t is None:
                limits_t = robot.data.soft_joint_pos_limits
            pos_limits = limits_t[0].detach().cpu().tolist()  # [n, 2] lower/upper
            limit_by_name = {n: pos_limits[i] for i, n in enumerate(sim_joint_names)}

            kp_by_name = {n: joint_kp[i].item() for i, n in enumerate(joint_names)}
            kd_by_name = {n: joint_kd[i].item() for i, n in enumerate(joint_names)}
            eff_by_name = {n: effort_limits[i].item() for i, n in enumerate(joint_names)}
            default_by_name = {n: init_joint_pos[i] for i, n in enumerate(joint_names)}

            # strip the sim "leg_" prefix to match humanoid-control joint names
            def _contract_name(sim_name):
                return sim_name[len("leg_"):] if sim_name.startswith("leg_") else sim_name

            contract_joints = []
            for idx, sim_name in enumerate(HUMANOID_LITE_LEG_JOINTS):
                lo, hi = limit_by_name.get(sim_name, [None, None])
                contract_joints.append({
                    "index": idx,
                    "joint_name": _contract_name(sim_name),
                    "kp": kp_by_name.get(sim_name),
                    "kd": kd_by_name.get(sim_name),
                    "effort_limit": eff_by_name.get(sim_name),
                    "position_limit_lower": lo,
                    "position_limit_upper": hi,
                    "default_pose": default_by_name.get(sim_name),
                })

            contract = {
                "_meta": {
                    "source": "humanoid-policy trainer export (scripts/rsl_rl/play.py)",
                    "task": args_cli.task,
                    "note": "Sim-side contract for humanoid-control; joint_order/obs/action match leg_policy_params.json.",
                },
                "canonical_joint_order": [_contract_name(n) for n in HUMANOID_LITE_LEG_JOINTS],
                "control": {
                    "policy_dt": float(env_cfg.sim.dt * env_cfg.decimation),
                    "control_dt": 0.004,
                    "action_scale": float(env_cfg.actions.joint_pos.scale),
                },
                "observation": {
                    "num_observations": int(env.observation_space["policy"].shape[-1]),
                    "layout": [
                        "command(3)", "base_ang_vel(3)", "projected_gravity(3)",
                        "joint_pos_minus_default(12)", "joint_vel(12)", "prev_action(12)",
                    ],
                },
                "action": {
                    "num_actions": int(env.action_space.shape[-1]),
                    "formula": "target = clip(action)*action_scale + default_pose, then clamp to position limits",
                },
                "joints": contract_joints,
            }
            with open("configs/leg_policy_contract.json", "w") as f:
                json.dump(contract, f, indent=2)
            print("[INFO]: Wrote humanoid-control contract to configs/leg_policy_contract.json")
    except Exception as exc:  # never let contract export break play
        print(f"[WARN]: contract export skipped: {exc}")

    # reset environment
    obs = env.get_observations()

    # warm up torch.compile — actual JIT compilation happens on the first forward pass
    _done = threading.Event()
    def _spinner():
        for ch in itertools.cycle(r"/-\|"):
            if _done.is_set():
                break
            sys.stdout.write(f"\r[INFO]: Compiling policy (torch.compile warm-up)... {ch}  ")
            sys.stdout.flush()
            time.sleep(0.1)
        sys.stdout.write("\r[INFO]: Policy compiled.                                \n")
        sys.stdout.flush()
    t = threading.Thread(target=_spinner, daemon=True)
    t.start()
    with torch.inference_mode():
        policy(obs)
    _done.set()
    t.join()

    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
        timestep += 1
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
