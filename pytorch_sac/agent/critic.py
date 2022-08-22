import numpy as np
import torch
import torch.nn.functional as F
import utils
from torch import nn


class DoubleQCritic(nn.Module):
    """Critic network, employes double Q-learning."""

    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth, untrainable=0):
        super().__init__()

        self.Q1 = utils.mlp(
            obs_dim + action_dim, hidden_dim, 1, hidden_depth, untrainable=untrainable
        )
        self.Q2 = utils.mlp(
            obs_dim + action_dim, hidden_dim, 1, hidden_depth, untrainable=untrainable
        )

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        self.outputs["q1"] = q1
        self.outputs["q2"] = q2

        return q1, q2

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f"train_critic/{k}_hist", v, step, log_frequency=50000)

        assert len(self.Q1) == len(self.Q2)
        for i, (m1, m2) in enumerate(zip(self.Q1, self.Q2)):
            assert type(m1) == type(m2)
            if type(m1) is nn.Linear:
                logger.log_param(
                    f"train_critic/q1_fc{i}", m1, step, log_frequency=200000
                )
                logger.log_param(
                    f"train_critic/q2_fc{i}", m2, step, log_frequency=200000
                )
