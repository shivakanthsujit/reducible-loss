import time

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from agent import Agent
from per_buffer import PERReplayMemory
from tqdm import tqdm


class SACAgent(Agent):
    """SAC algorithm."""

    def __init__(
        self,
        obs_dim,
        action_dim,
        action_range,
        device,
        critic_cfg,
        actor_cfg,
        discount,
        init_temperature,
        alpha_lr,
        alpha_betas,
        actor_lr,
        actor_betas,
        actor_update_frequency,
        critic_lr,
        critic_betas,
        critic_tau,
        critic_target_update_frequency,
        batch_size,
        learnable_temperature,
    ):
        super().__init__()

        self.action_range = action_range
        self.device = torch.device(device)
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size
        self.learnable_temperature = learnable_temperature

        self.critic = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic_target = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor = hydra.utils.instantiate(actor_cfg).to(self.device)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -action_dim

        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=actor_betas
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=critic_betas
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=alpha_betas
        )

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return utils.to_np(action[0])

    def update_critic(self, obs, action, reward, next_obs, not_done, logger, step):
        dist = self.actor(next_obs)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
        target_Q = reward + (not_done * self.discount * target_V)
        target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        loss1 = F.mse_loss(current_Q1, target_Q, reduction="none")
        loss2 = F.mse_loss(current_Q2, target_Q, reduction="none")
        ele_critic_loss = loss1 + loss2
        critic_loss = torch.mean(ele_critic_loss)

        logger.log(f"train/pure_td_error", ele_critic_loss.mean(), step)

        logger.log("train_critic/loss", critic_loss, step)

        # * ReLo Loss
        with torch.no_grad():
            orig_loss = ele_critic_loss / 2
            target_current_Q1, target_current_Q2 = self.critic_target(obs, action)

            irr_loss1 = F.mse_loss(target_current_Q1, target_Q, reduction="none")
            irr_loss2 = F.mse_loss(target_current_Q2, target_Q, reduction="none")
            irr_loss = (irr_loss1 + irr_loss2) / 2
            relo_loss = orig_loss - irr_loss
            if not torch.isnan(irr_loss).any():
                irr_metrics = {
                    "irr_min": irr_loss.min(),
                    "irr_max": irr_loss.max(),
                    "irr_mean": irr_loss.mean(),
                    "irr_std": irr_loss.std(),
                }
                [logger.log(f"train/{k}", v, step) for k, v in irr_metrics.items()]
            else:
                tqdm.write("Nan encountered in irr loss")

            if not torch.isnan(relo_loss).any():
                relo_metrics = {
                    "relo_min": relo_loss.min(),
                    "relo_max": relo_loss.max(),
                    "relo_mean": relo_loss.mean(),
                    "relo_std": relo_loss.std(),
                    "relo_exp": relo_loss.mean().exp(),
                }
                [logger.log(f"train/{k}", v, step) for k, v in relo_metrics.items()]
            else:
                tqdm.write("Nan encountered in relo loss")

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.log(logger, step)

    def update_actor_and_alpha(self, obs, logger, step):
        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(obs, action)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        logger.log("train_actor/loss", actor_loss, step)
        logger.log("train_actor/target_entropy", self.target_entropy, step)
        logger.log("train_actor/entropy", -log_prob.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(logger, step)

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (
                self.alpha * (-log_prob - self.target_entropy).detach()
            ).mean()
            logger.log("train_alpha/loss", alpha_loss, step)
            logger.log("train_alpha/value", self.alpha, step)
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

    def update(self, replay_buffer, logger, step):
        obs, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.sample(
            self.batch_size
        )

        logger.log("train/batch_reward", reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done_no_max, logger, step)

        if step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(obs, logger, step)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target, self.critic_tau)


class PERSACAgent(Agent):
    """SAC algorithm."""

    def __init__(
        self,
        obs_dim,
        action_dim,
        action_range,
        device,
        critic_cfg,
        actor_cfg,
        discount,
        init_temperature,
        alpha_lr,
        alpha_betas,
        actor_lr,
        actor_betas,
        actor_update_frequency,
        critic_lr,
        critic_betas,
        critic_tau,
        critic_target_update_frequency,
        batch_size,
        learnable_temperature,
    ):
        super().__init__()

        self.action_range = action_range
        self.device = torch.device(device)
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size
        self.learnable_temperature = learnable_temperature

        self.critic = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic_target = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor = hydra.utils.instantiate(actor_cfg).to(self.device)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -action_dim

        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=actor_betas
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=critic_betas
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=alpha_betas
        )

        self.train()
        self.critic_target.train()

        print(f"Using {self} as SAC Algo")

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return utils.to_np(action[0])

    def update_critic(
        self, obs, action, reward, next_obs, not_done, weights, logger, step
    ):
        dist = self.actor(next_obs)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
        target_Q = reward + (not_done * self.discount * target_V)
        target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        assert (
            target_Q.shape[0] == obs.shape[0] and target_Q.shape[1] == 1
        ), f"Target Q shape is wrong. Got {target_Q.shape}"
        assert (
            current_Q1.shape == target_Q.shape
        ), f"{current_Q1.shape} != {target_Q.shape}"
        assert (
            current_Q2.shape == target_Q.shape
        ), f"{current_Q2.shape} != {target_Q.shape}"

        loss1 = F.mse_loss(current_Q1, target_Q, reduction="none")
        loss2 = F.mse_loss(current_Q2, target_Q, reduction="none")
        ele_critic_loss = loss1 + loss2

        # * With weights
        critic_loss = torch.mean(ele_critic_loss * weights)

        logger.log(f"train/pure_td_error", ele_critic_loss.mean(), step)

        # * ReLo Loss
        with torch.no_grad():
            orig_loss = ele_critic_loss / 2
            target_current_Q1, target_current_Q2 = self.critic_target(obs, action)

            irr_loss1 = F.mse_loss(target_current_Q1, target_Q, reduction="none")
            irr_loss2 = F.mse_loss(target_current_Q2, target_Q, reduction="none")
            irr_loss = (irr_loss1 + irr_loss2) / 2
            relo_loss = orig_loss - irr_loss
            if not torch.isnan(irr_loss).any():
                irr_metrics = {
                    "irr_min": irr_loss.min(),
                    "irr_max": irr_loss.max(),
                    "irr_mean": irr_loss.mean(),
                    "irr_std": irr_loss.std(),
                }
                [logger.log(f"train/{k}", v, step) for k, v in irr_metrics.items()]
            else:
                tqdm.write("Nan encountered in irr loss")

            if not torch.isnan(relo_loss).any():
                relo_metrics = {
                    "relo_min": relo_loss.min(),
                    "relo_max": relo_loss.max(),
                    "relo_mean": relo_loss.mean(),
                    "relo_std": relo_loss.std(),
                }
                [logger.log(f"train/{k}", v, step) for k, v in relo_metrics.items()]
            else:
                tqdm.write("Nan encountered in relo loss")

        # * PER Loss
        with torch.no_grad():
            td_error = ele_critic_loss / 2
            td_error = td_error.cpu().numpy().squeeze(1)

        logger.log("train_critic/loss", critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.log(logger, step)

        assert td_error.shape[0] == obs.shape[0]

        return td_error

    def update_actor_and_alpha(self, obs, weights, logger, step):
        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(obs, action)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = self.alpha.detach() * log_prob - actor_Q

        # * No weights
        # actor_loss = actor_loss.mean()

        # * With weights
        actor_loss = (actor_loss * weights).mean()

        logger.log("train_actor/loss", actor_loss, step)
        logger.log("train_actor/target_entropy", self.target_entropy, step)
        logger.log("train_actor/entropy", -log_prob.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(logger, step)

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (
                self.alpha * (-log_prob - self.target_entropy).detach()
            ).mean()
            logger.log("train_alpha/loss", alpha_loss, step)
            logger.log("train_alpha/value", self.alpha, step)
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

    def update(self, replay_buffer: PERReplayMemory, logger, step):
        ttime = {}
        ttime["mem"] = time.time()
        (
            idxs,
            obs,
            action,
            reward,
            next_obs,
            not_done,
            not_done_no_max,
            weights,
        ) = replay_buffer.sample(self.batch_size)
        ttime["mem"] = time.time() - ttime["mem"]

        logger.log("train/batch_reward", reward.mean(), step)

        ttime["update_critic"] = time.time()
        td_error = self.update_critic(
            obs, action, reward, next_obs, not_done_no_max, weights, logger, step
        )
        ttime["update_critic"] = time.time() - ttime["update_critic"]

        if step % self.actor_update_frequency == 0:
            ttime["update_actor_and_alpha"] = time.time()
            self.update_actor_and_alpha(obs, weights, logger, step)
            ttime["update_actor_and_alpha"] = (
                time.time() - ttime["update_actor_and_alpha"]
            )

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target, self.critic_tau)

        ttime["update_priorities"] = time.time()
        replay_buffer.update_priorities(idxs, td_error + 1e-5)
        ttime["update_priorities"] = time.time() - ttime["update_priorities"]

        [logger.log(f"train/{k}", v, step) for k, v in ttime.items()]
        td_metrics = {
            "td_min": td_error.min(),
            "td_max": td_error.max(),
            "td_mean": td_error.mean(),
            "td_std": td_error.std(),
        }
        [logger.log(f"train/{k}", v, step) for k, v in td_metrics.items()]


class RELOSACAgent(PERSACAgent):
    """SAC algorithm."""

    def __init__(
        self,
        obs_dim,
        action_dim,
        action_range,
        device,
        critic_cfg,
        actor_cfg,
        discount,
        init_temperature,
        alpha_lr,
        alpha_betas,
        actor_lr,
        actor_betas,
        actor_update_frequency,
        critic_lr,
        critic_betas,
        critic_tau,
        critic_target_update_frequency,
        batch_size,
        learnable_temperature,
    ):
        super().__init__(
            obs_dim,
            action_dim,
            action_range,
            device,
            critic_cfg,
            actor_cfg,
            discount,
            init_temperature,
            alpha_lr,
            alpha_betas,
            actor_lr,
            actor_betas,
            actor_update_frequency,
            critic_lr,
            critic_betas,
            critic_tau,
            critic_target_update_frequency,
            batch_size,
            learnable_temperature,
        )

        print(f"Using {self} as SAC Algo")

    def update_critic(
        self, obs, action, reward, next_obs, not_done, weights, logger, step
    ):
        dist = self.actor(next_obs)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
        target_Q = reward + (not_done * self.discount * target_V)
        target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        assert (
            target_Q.shape[0] == obs.shape[0] and target_Q.shape[1] == 1
        ), f"Target Q shape is wrong. Got {target_Q.shape}"
        assert (
            current_Q1.shape == target_Q.shape
        ), f"{current_Q1.shape} != {target_Q.shape}"
        assert (
            current_Q2.shape == target_Q.shape
        ), f"{current_Q2.shape} != {target_Q.shape}"

        loss1 = F.mse_loss(current_Q1, target_Q, reduction="none")
        loss2 = F.mse_loss(current_Q2, target_Q, reduction="none")
        ele_critic_loss = loss1 + loss2

        # * With weights
        critic_loss = torch.mean(ele_critic_loss * weights)

        logger.log(f"train/pure_td_error", ele_critic_loss.mean(), step)

        # * ReLo Loss
        with torch.no_grad():
            orig_loss = ele_critic_loss / 2
            target_current_Q1, target_current_Q2 = self.critic_target(obs, action)

            irr_loss1 = F.mse_loss(target_current_Q1, target_Q, reduction="none")
            irr_loss2 = F.mse_loss(target_current_Q2, target_Q, reduction="none")
            irr_loss = (irr_loss1 + irr_loss2) / 2
            relo_loss = orig_loss - irr_loss
            if not torch.isnan(irr_loss).any():
                irr_metrics = utils.get_stats("irr", irr_loss)
                [logger.log(f"train/{k}", v, step) for k, v in irr_metrics.items()]
            else:
                tqdm.write("Nan encountered in irr loss")

            if not torch.isnan(relo_loss).any():
                relo_metrics = utils.get_stats("relo", relo_loss)
                [logger.log(f"train/{k}", v, step) for k, v in relo_metrics.items()]
            else:
                tqdm.write("Nan encountered in relo loss")

        td_error = F.relu(relo_loss).cpu().numpy().squeeze(1)

        logger.log("train_critic/loss", critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.log(logger, step)

        assert td_error.shape[0] == obs.shape[0]

        return td_error
