#!/usr/bin/env python3
import copy
import math
import os
import pickle as pkl
import sys
import time

import dmc2gym
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from logger import Logger
from per_buffer import PERReplayMemory
from pytorch_model_summary import summary
from replay_buffer import ReplayBuffer
from tqdm import tqdm
from video import VideoRecorder

os.environ["MUJOCO_GL"] = "egl"


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f"workspace: {self.work_dir}")

        self.cfg = cfg

        self.logger = Logger(
            self.work_dir,
            save_tb=cfg.log_save_tb,
            log_frequency=cfg.log_frequency,
            agent=cfg.agent.name,
        )

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.env = utils.make_env(cfg)

        cfg.agent.params.obs_dim = self.env.observation_space.shape[0]
        cfg.agent.params.action_dim = self.env.action_space.shape[0]
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max()),
        ]
        self.agent = hydra.utils.instantiate(cfg.agent)

        if self.cfg.use_per_buffer:
            assert "PER" in cfg.agent["class"] or "RELO" in cfg.agent["class"]
            self.replay_buffer = PERReplayMemory(
                self.env.observation_space.shape,
                self.env.action_space.shape,
                cfg.priority_beta,
                cfg.priority_alpha,
                int(cfg.replay_buffer_capacity),
                self.device,
            )
        else:
            self.replay_buffer = ReplayBuffer(
                self.env.observation_space.shape,
                self.env.action_space.shape,
                int(cfg.replay_buffer_capacity),
                self.device,
            )

        self.video_recorder = VideoRecorder(self.work_dir if cfg.save_video else None)
        self.step = 0

        self.agent.reset()

        print("Actor Summary")
        temp_obs = torch.zeros((1, cfg.agent.params.obs_dim)).to(cfg.device)
        temp_act = torch.zeros((1, cfg.agent.params.action_dim)).to(cfg.device)
        print(summary(self.agent.actor, temp_obs))
        print("Critic Summary")
        print(summary(self.agent.critic, temp_obs, temp_act))

    def evaluate(self):
        average_episode_reward = 0
        average_episode_len = 0
        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            self.agent.reset()
            self.video_recorder.init(enabled=(episode == 0))
            done = False
            episode_reward = 0
            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, _ = self.env.step(action)
                self.video_recorder.record(self.env)
                episode_reward += reward
                average_episode_len += 1

            average_episode_reward += episode_reward
            try:
                self.video_recorder.save("video.mp4")
            except Exception as e:
                tqdm.write(e)
                tqdm.write("Video saving failed.")
        average_episode_reward /= self.cfg.num_eval_episodes
        average_episode_len /= self.cfg.num_eval_episodes
        self.logger.log("eval/episode_reward", average_episode_reward, self.step)
        self.logger.log("eval/episode_len", average_episode_len, self.step)
        self.logger.dump(self.step)

    def run(self):
        episode, episode_reward, done = 0, 0, True
        start_time = time.time()
        pbar = tqdm(
            total=int(self.cfg.num_train_steps),
            initial=int(self.step),
            miniters=int(1e3),
        )
        while self.step < self.cfg.num_train_steps:
            if done:
                if self.step > 0:
                    self.logger.log(
                        "train/duration", time.time() - start_time, self.step
                    )
                    start_time = time.time()
                    if self.step % self.cfg.log_frequency == 0:
                        self.logger.dump(
                            self.step, save=(self.step > self.cfg.num_seed_steps)
                        )

                # evaluate agent periodically
                if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                    self.logger.log("eval/episode", episode, self.step)
                    self.evaluate()

                self.logger.log("train/episode_reward", episode_reward, self.step)

                obs = self.env.reset()
                self.agent.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1

                self.logger.log("train/episode", episode, self.step)

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            if self.cfg.use_per_buffer:
                fraction = min(self.step / self.cfg.num_train_steps, 1.0)
                beta = self.replay_buffer.priority_beta
                beta = beta + fraction * (1.0 - beta)
                self.logger.log(
                    "train/beta", self.replay_buffer.priority_beta, self.step
                )
                self.replay_buffer.priority_beta = min(beta, 1)

            # run training update
            if self.step >= self.cfg.num_seed_steps:
                self.agent.update(self.replay_buffer, self.logger, self.step)

            next_obs, reward, done, _ = self.env.step(action)

            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward

            self.replay_buffer.add(obs, action, reward, next_obs, done, done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1
            pbar.update(1)
        pbar.close()

        self.logger.log("eval/episode", episode, self.step)
        self.evaluate()

        checkpoint = {
            "actor": self.agent.actor,
            "critic": self.agent.critic,
            "critic_target": self.agent.critic_target,
            "log_alpha": self.agent.log_alpha,
        }
        fname = os.path.join(self.work_dir, "checkpoint.pth")
        torch.save(checkpoint, fname)


@hydra.main(config_path="config/train.yaml", strict=True)
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
