"""
Author:
Time:
"""
import os
import gym
import torch
import torch.nn as nn
import math
import matplotlib
import numpy as np

eps = 1e-13


class ActorCritic(nn.Module):
    def __init__(self, cluster_state_dimension, density_state_dimension, cell_state_dimension, num_outputs,
                 layer_norm=True):
        super(ActorCritic, self).__init__()

        self.density_state_dimension = density_state_dimension
        # encoder
        # cluster net
        self.cluster_conv1 = nn.Conv2d(
            in_channels=5, out_channels=16, kernel_size=(1, 3), stride=1, padding=(0, 1))

        self.cluster_conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=(1, 4), stride=1, padding=(0, 1))

        self.cluster_conv3 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=(1, 1), stride=1, padding=(0, 1))

        # cell net
        self.cell_conv1 = nn.Conv1d(
            in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.cell_conv2 = nn.Conv1d(
            in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.cell_conv3 = nn.Conv1d(
            in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)

        # decoder
        # actor
        self.actor_fc1 = nn.Linear(164, 128)
        self.actor_fc2 = nn.Linear(128, 64)
        self.actor_fc3 = nn.Linear(64, 32)
        self.actor_fc4 = nn.Linear(32, 4)

        # critic
        self.critic_fc1 = nn.Linear(164, 128)
        self.critic_fc2 = nn.Linear(128, 64)
        self.critic_fc3 = nn.Linear(64, 32)
        self.critic_fc4 = nn.Linear(32, 1)

        if layer_norm:
            self.layer_norm(self.cluster_conv1, std=1.0)
            self.layer_norm(self.cluster_conv2, std=1.0)
            self.layer_norm(self.cluster_conv3, std=1.0)
            self.layer_norm(self.cell_conv1, std=1.0)
            self.layer_norm(self.cell_conv2, std=1.0)
            self.layer_norm(self.cell_conv3, std=1.0)

            self.layer_norm(self.actor_fc1, std=1.0)
            self.layer_norm(self.actor_fc2, std=1.0)
            self.layer_norm(self.actor_fc3, std=1.0)
            # self.layer_norm(self.actor_fc1, std=1.0)
            self.layer_norm(self.critic_fc1, std=1.0)
            self.layer_norm(self.critic_fc2, std=1.0)
            # self.layer_norm(self.critic_fc1, std=1.0)

    @staticmethod
    def layer_norm(layer, std=1.0, bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)

    def forward(self, states):
        """
        run policy network (actor) as well as value network (critic)
        :param states: a Tensor2 represents states
        :return: 3 Tensor2
        """
        row_cluster_state = states[0]  # 10 x 10 x 5
        row_density_state = states[1]  # 10
        post_cell_state = states[2]   # 30 x 5
        cluster_x = self.cluster_conv1(row_cluster_state)
        cluster_x = torch.tanh(torch.max_pool2d(
            cluster_x, (1, 3)))
        # density_cat = row_density_state.reshape()
        cluster_x = torch.tanh(torch.max_pool2d(
            self.cluster_conv2(cluster_x), (1, 2)))
        cluster_x = torch.tanh(torch.max_pool2d(
            self.cluster_conv3(cluster_x), (1, 3)))

        cell_x = torch.tanh(torch.max_pool1d(
            self.cell_conv1(post_cell_state), 3))
        cell_x = torch.tanh(torch.max_pool1d(
            self.cell_conv2(cell_x), 3))
        cell_x = torch.tanh(torch.max_pool1d(
            self.cell_conv3(cell_x), 3))

        fusion_x = torch.cat((cluster_x.flatten(1, 3), cell_x.flatten(
            1, 2), row_density_state), dim=1)

        actor_x = torch.tanh(self.actor_fc1(fusion_x))
        actor_x = torch.tanh(self.actor_fc2(actor_x))
        actor_x = torch.tanh(self.actor_fc3(actor_x))
        actor_x = torch.tanh(self.actor_fc4(actor_x))
        soft_prob = torch.softmax(actor_x, dim=1)

        row_density_state[row_density_state > 0] = 1
        action_prob = soft_prob * row_density_state

        prob_sum = (action_prob.sum(dim=1).reshape(action_prob.shape[0], 1))
        action_prob = action_prob / prob_sum

        critic_value = self._forward_critic(fusion_x)

        return action_prob, critic_value

    # def _forward_actor(self, states):
    #     x = torch.relu(self.actor_fc1(states))
    #     x = torch.relu(self.actor_fc2(x))
    #     x = torch.relu(self.actor_fc3(x))
    #     action_prob = torch.softmax(x, dim=1)
    #     action_logstd = self.actor_logstd.expand_as(action_prob)
    #     return action_prob, action_logstd
    #

    def _forward_critic(self, x):

        x = torch.tanh(self.critic_fc1(x))
        x = torch.tanh(self.critic_fc2(x))
        x = torch.tanh(self.critic_fc3(x))
        x = self.critic_fc4(x)
        return x

    def select_action(self, action_prob, return_logproba=True):
        """
        given mean and std, sample an action from normal(mean, std)
        also returns probability of the given chosen
        """
        elements = np.asarray([i for i in range((action_prob.shape[1]))])
        probabilities = action_prob.cpu().detach().numpy().flatten()
        # print(probabilities)
        action = torch.tensor(np.random.choice(
            elements, 1, p=probabilities))  # 正态分布随机

        if return_logproba:
            # logproba = self._normal_logproba(action, action_prob, action_logstd, action_std)
            prob = action_prob[torch.arange(
                action_prob.shape[0]), action.long()]
            logproba = prob
        return action, prob

    '''在输出中加入了了正态分布'''

    @ staticmethod
    def _normal_logproba(x, mean, logstd, std=None):
        if std is None:
            std = torch.exp(logstd)

        std_sq = std.pow(2)
        logproba = - 0.5 * math.log(2 * math.pi) - \
            logstd - (x - mean).pow(2) / (2 * std_sq)
        return logproba.sum(1)

    def get_prob(self, states, actions):
        """
        return probability of chosen the given actions under corresponding states of current network
        :param states: Tensor
        :param actions: Tensor
        """
        action_prob, values = self.forward(states)
        # s1 = torch.arange(action_prob.shape[0])
        # s2 = actions.long()
        # s3 = torch.cat((s1.reshape(s1.shape[0],1), s2.reshape(s1.shape[0],1)), dim=1)
        prob = action_prob[torch.arange(action_prob.shape[0]), actions.long()]
        # logproba = torch.log(prob)
        # logproba = self._normal_logproba(actions, action_prob, action_logstd)
        return prob, values
