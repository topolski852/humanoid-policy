"""TD-MPC2 training loop over Isaac Lab's fixed-N parallel envs.

Rewrite of upstream single-env online_trainer for vectorized collection: each iteration steps all
N envs once, stores the batch in the SequenceReplayBuffer, then runs `updates_per_step` gradient
updates. Agent acts in [-1,1]; env actions are ×act_env_scale. Tracks completed-episode returns for
best-checkpoint saving and TensorBoard logging.
"""

from __future__ import annotations

import os
import time
from collections import deque

import torch
from torch.utils.tensorboard import SummaryWriter


class TdmpcTrainer:
    def __init__(self, cfg, env, agent, buffer, log_dir):
        self.cfg = cfg
        self.env = env
        self.agent = agent
        self.buffer = buffer
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        self.N = env.num_envs
        self.act_scale = float(cfg.act_env_scale)
        self.best_return = -float("inf")
        self.ret_hist = deque(maxlen=100)

    def _save(self, name):
        self.agent.save(os.path.join(self.log_dir, name))

    def train(self):
        cfg, env, agent, buf = self.cfg, self.env, self.agent, self.buffer
        obs_p, obs_c = env.reset()
        ep_return = torch.zeros(self.N, device=env.device)
        ep_len = torch.zeros(self.N, device=env.device)
        total = 0                 # total env-steps (transitions) collected
        last_log = 0
        last_save = 0
        t_start = time.time()
        last_info = {}

        while total < cfg.max_env_steps:
            # --- act ---
            plan_mean = plan_std = None
            if total < cfg.seed_steps:
                agent_action = 2.0 * torch.rand(self.N, env.num_actions, device=env.device) - 1.0
            elif cfg.plan_collection:
                agent_action, plan_mean, plan_std = agent.plan_batch(obs_p, eval_mode=False)  # proper TD-MPC2 (MPPI)
            else:
                agent_action = agent.act_pi(obs_p, eval_mode=False)       # fast policy-prior collection
            # TD-M(PC)²: every stored transition needs a planner (mean,std); non-planner steps
            # (seed / prior) get a wide prior (max_std) so their regularization is ~inert.
            if cfg.use_tdmpc2_square:
                if plan_mean is None:
                    plan_mean = agent_action
                    plan_std = torch.full_like(agent_action, cfg.max_std)
            else:
                plan_mean = plan_std = None
            env_action = agent_action * self.act_scale

            # --- step + store (store the NORMALIZED action + pre-step policy obs) ---
            nobs_p, nobs_c, reward, terminated, time_out, _ = env.step(env_action)
            priv = obs_c if buf.use_priv else None
            buf.add(obs_p, agent_action, reward, terminated, time_out, priv=priv,
                    plan_mean=plan_mean, plan_std=plan_std)

            ep_return += reward
            ep_len += 1
            done = terminated | time_out
            if done.any():
                for r in ep_return[done].tolist():
                    self.ret_hist.append(r)
                ep_return = torch.where(done, torch.zeros_like(ep_return), ep_return)
                ep_len = torch.where(done, torch.zeros_like(ep_len), ep_len)

            obs_p, obs_c = nobs_p, nobs_c
            total += self.N

            # --- learn ---
            if total >= cfg.seed_steps:
                for _ in range(cfg.updates_per_step):
                    batch = buf.sample(cfg.batch_size)
                    if batch is not None:
                        last_info = agent.update(batch)

            # --- log ---
            if total - last_log >= cfg.log_interval_steps:
                last_log = total
                sps = total / (time.time() - t_start)
                mean_ret = sum(self.ret_hist) / len(self.ret_hist) if self.ret_hist else float("nan")
                self.writer.add_scalar("collect/mean_episode_return", mean_ret, total)
                self.writer.add_scalar("collect/env_steps_per_sec", sps, total)
                self.writer.add_scalar("buffer/size", len(buf), total)
                for k, v in last_info.items():
                    self.writer.add_scalar(f"loss/{k}", float(v), total)
                print(f"[tdmpc] steps={total} sps={sps:.0f} buf={len(buf)} "
                      f"ep_return={mean_ret:.2f} "
                      + " ".join(f"{k}={float(v):.3f}" for k, v in last_info.items() if 'loss' in k))
                # best-checkpoint on smoothed return
                if self.ret_hist and mean_ret > self.best_return:
                    self.best_return = mean_ret
                    self._save("model_best.pt")

            if total - last_save >= cfg.save_interval_steps:
                last_save = total
                self._save(f"model_{total}.pt")

        self._save(f"model_{total}.pt")
        self.writer.flush()
        self.writer.close()
        print(f"[tdmpc] training done: {total} env-steps, best_ep_return={self.best_return:.2f}")
