"""
Implementation of PPO
ref: Schulman, John, et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).
ref: https://github.com/Jiankai-Sun/Proximal-Policy-Optimization-in-Pytorch/blob/master/ppo.py
ref: https://github.com/openai/baselines/tree/master/baselines/ppo2

NOTICE:
    `Tensor2` means 2D-Tensor (num_samples, num_dims)
"""
import math
import datetime
import argparse
import numpy as np
import pandas as pd
from os import makedirs as mkdir
from os.path import join as joindir
import matplotlib.pyplot as plt
import gym
import torch
import torch.nn as nn
import torch.optim as opt
from torch import Tensor
from torch.autograd import Variable
from collections import namedtuple
from itertools import count
import matplotlib
from .actor_critic import ActorCritic

matplotlib.use('agg')

Transition = namedtuple('Transition', (
    'row_cluster_state', 'row_density_state', 'post_cell_state', 'value', 'action', 'prob', 'mask',
    'next_row_cluster_state', 'next_row_density_state', 'next_post_cell_state', 'reward'))
EPS = 1e-10

'''
正在运行的state
'''
device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
# device_name = "cpu"
device = torch.device(device=device_name)


class RunningStat(object):
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape


class ZFilter:
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """

    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip

        self.rs = RunningStat(shape)

    '''
    调用 norm
    '''

    def __call__(self, x, update=True):
        if update:
            self.rs.push(x)
        if self.demean:
            x = x - self.rs.mean
        if self.destd:
            x = x / (self.rs.std + 1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

    def output_shape(self, input_space):
        return input_space.shape


class Memory(object):
    def __init__(self):
        self.memory = []

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self):
        return Transition(*zip(*self.memory))

    def __len__(self):
        return len(self.memory)


def abacus_episode(args, env, network, norm_cluster_state, norm_cell_state):
    memory = Memory()
    num_steps = 0
    reward_list = []
    state = env.reset()
    if args.state_norm:
        row_cluster_state, row_density_state, post_cell_state = state
        row_cluster_state = norm_cluster_state(row_cluster_state)
        row_density_state = row_density_state
        post_cell_state = norm_cell_state(post_cell_state)
        state = row_cluster_state, row_density_state, post_cell_state
    reward_sum = 0
    tu = []
    for t in range(args.max_step_per_round):
        prob = 0.9
        action, next_state, reward, done, _ = env.abacus_step()
        reward_sum += reward

        if not done and args.state_norm:
            row_cluster_state, row_density_state, post_cell_state = next_state
            row_cluster_state = norm_cluster_state(row_cluster_state)
            row_density_state = row_density_state
            post_cell_state = norm_cell_state(post_cell_state)
            next_state = row_cluster_state, row_density_state, post_cell_state
        mask = 0 if done else 1
        row_cluster_state, row_density_state, post_cell_state = state
        # print(row_cluster_state, reward, action)
        next_row_cluster_state, next_row_density_state, next_post_cell_state = next_state
        tu.append((row_cluster_state, row_density_state, post_cell_state, action, prob, mask,
                   next_row_cluster_state, next_row_density_state, next_post_cell_state, reward))
        if done:
            break

        state = next_state
    value_list = []
    pre_val = 0
    for i in reversed(range(len(tu))):
        row_cluster_state, row_density_state, post_cell_state, action, prob, mask, next_row_cluster_state, next_row_density_state, next_post_cell_state, reward = tu[
            i]
        value_list.append(reward + pre_val)
        pre_val = reward
    value_list.reverse()
    while num_steps < 10:
        for (i, tup) in enumerate(tu):
            row_cluster_state, row_density_state, post_cell_state, action, prob, mask, next_row_cluster_state, next_row_density_state, next_post_cell_state, reward = tup
            value = value_list[i]
            memory.push(row_cluster_state, row_density_state, post_cell_state, value, action, prob, mask,
                        next_row_cluster_state, next_row_density_state, next_post_cell_state, reward)
        num_steps += (t + 1)
        reward_list.append(reward_sum)
    return memory, np.mean(reward_list)


def network_episode(args, env, network, norm_cluster_state, norm_cell_state):
    memory = Memory()
    num_steps = 0
    reward_list = []
    len_list = []
    while num_steps < args.batch_size:
        state = env.reset()
        if args.state_norm:
            row_cluster_state, row_density_state, post_cell_state = state
            row_cluster_state = norm_cluster_state(row_cluster_state)
            row_density_state = row_density_state
            post_cell_state = norm_cell_state(post_cell_state)
            state = row_cluster_state, row_density_state, post_cell_state
        reward_sum = 0
        for t in range(args.max_step_per_round):
            # x = Tensor(state)
            row_cluster_state, row_density_state, post_cell_state = state
            tensor_state = Tensor(row_cluster_state).unsqueeze(0).to(device), Tensor(row_density_state).unsqueeze(0).to(device), Tensor(
                post_cell_state).unsqueeze(0).to(device)
            action_prob, value = network(tensor_state)
            # print(action_prob)
            action, prob = network.select_action(action_prob)
            action = action.data.numpy()[0]
            prob = prob.cpu().data.numpy()[0]
            next_state, reward, done, _ = env.step(action)
            reward_sum += reward

            if not done and args.state_norm:
                row_cluster_state, row_density_state, post_cell_state = next_state
                row_cluster_state = norm_cluster_state(row_cluster_state)
                row_density_state = row_density_state
                post_cell_state = norm_cell_state(post_cell_state)
                next_state = row_cluster_state, row_density_state, post_cell_state
            mask = 0 if done else 1
            next_row_cluster_state, next_row_density_state, next_post_cell_state = next_state
            memory.push(row_cluster_state, row_density_state, post_cell_state, value, action, prob, mask,
                        next_row_cluster_state, next_row_density_state, next_post_cell_state, reward)
            if done:
                break

            state = next_state

        num_steps += (t + 1)
        reward_list.append(reward_sum)
    return memory, np.mean(reward_list)


def test_net(args, env, network, norm_cluster_state, norm_cell_state):
    memory = Memory()
    num_steps = 0
    reward_list = []
    len_list = []
    while num_steps < args.batch_size:
        state = env.reset()
        if args.state_norm:
            row_cluster_state, row_density_state, post_cell_state = state
            row_cluster_state = norm_cluster_state(row_cluster_state)
            row_density_state = row_density_state
            post_cell_state = norm_cell_state(post_cell_state)
            state = row_cluster_state, row_density_state, post_cell_state
        reward_sum = 0
        for t in range(args.max_step_per_round):
            # x = Tensor(state)
            row_cluster_state, row_density_state, post_cell_state = state
            tensor_state = Tensor(row_cluster_state).unsqueeze(0).to(device), Tensor(row_density_state).unsqueeze(0).to(device), Tensor(
                post_cell_state).unsqueeze(0).to(device)
            action_prob, value = network(tensor_state)
            # print(action_prob)
            action, prob = network.select_action(action_prob)
            action = action.data.numpy()[0]
            prob = prob.cpu().data.numpy()[0]
            next_state, reward, done, _ = env.step(action)
            reward_sum += reward

            if not done and args.state_norm:
                row_cluster_state, row_density_state, post_cell_state = next_state
                row_cluster_state = norm_cluster_state(row_cluster_state)
                row_density_state = row_density_state
                post_cell_state = norm_cell_state(post_cell_state)
                next_state = row_cluster_state, row_density_state, post_cell_state
            mask = 0 if done else 1
            next_row_cluster_state, next_row_density_state, next_post_cell_state = next_state
            memory.push(row_cluster_state, row_density_state, post_cell_state, value, action, prob, mask,
                        next_row_cluster_state, next_row_density_state, next_post_cell_state, reward)
            if done:
                break

            state = next_state

        num_steps += (t + 1)
        reward_list.append(reward_sum)


def ppo_train(args, network, optimizer, env, state_dimension, memory_method):
    torch.manual_seed(args.seed)
    norm_cluster_state = ZFilter(state_dimension[0], clip=5.0)
    norm_density_state = ZFilter(state_dimension[1], clip=5.0)
    norm_cell_state = ZFilter(state_dimension[2], clip=5.0)

    # record average 1-round cumulative reward in every episode
    reward_record = []
    global_steps = 0

    lr_now = args.lr
    clip_now = args.clip

    for i_episode in range(args.num_episode):
        # step1: perform current policy to collect trajectories
        # this is an on-policy method!
        memory, mean_reward = memory_method(
            args, env, network, norm_cluster_state, norm_cell_state)
        batch = memory.sample()
        batch_size = len(memory)
        reward_record.append({
            'episode': i_episode,
            'steps': global_steps,
            'meanepreward': mean_reward,
        })
        # step2: extract variables from trajectories
        rewards = Tensor(batch.reward).to(device)
        values = Tensor(batch.value).to(device)
        masks = Tensor(batch.mask).to(device)
        actions = Tensor(batch.action).to(device)
        row_cluster_state = Tensor(batch.row_cluster_state).to(device)
        row_density_state = Tensor(batch.row_density_state).to(device)
        post_cell_state = Tensor(batch.post_cell_state).to(device)
        # states = ()
        oldprob = Tensor(batch.prob).to(device)

        returns = Tensor(batch_size).to(device)
        deltas = Tensor(batch_size).to(device)
        advantages = Tensor(batch_size).to(device)

        prev_return = 0
        prev_value = 0
        prev_advantage = 0

        # generalization advantage estimate
        for i in reversed(range(batch_size)):
            # value : MC
            returns[i] = rewards[i] + args.gamma * prev_return * masks[i]

            # ref: https://arxiv.org/pdf/1506.02438.pdf (generalization advantage estimate)
            # GAE
            deltas[i] = rewards[i] + args.gamma * \
                prev_value * masks[i] - values[i]
            advantages[i] = deltas[i] + args.gamma * \
                args.lamda * prev_advantage * masks[i]

            prev_return = returns[i]
            prev_value = values[i]
            prev_advantage = advantages[i]
        if args.advantage_norm:
            minibatch_return_6std = 6 * returns.std()
            returns = returns / \
                (minibatch_return_6std + EPS)
            advantages = (advantages - advantages.mean()) / \
                (advantages.std() + EPS)

        for i_epoch in range(int(args.num_epoch * batch_size / args.minibatch_size)):
            # sample from current batch
            minibatch_ind = np.random.choice(
                batch_size, args.minibatch_size, replace=False)
            minibatch_row_cluster_state = row_cluster_state[minibatch_ind]
            minibatch_row_density_state = row_density_state[minibatch_ind]
            minibatch_post_cell_state = post_cell_state[minibatch_ind]
            minibatch_states = (minibatch_row_cluster_state,
                                minibatch_row_density_state, minibatch_post_cell_state)
            minibatch_actions = actions[minibatch_ind]
            minibatch_oldprob = oldprob[minibatch_ind]
            minibatch_newprob, minibatch_newvalues = network.get_prob(
                minibatch_states, minibatch_actions)
            # minibatch_newvalues = network._forward_critic(minibatch_states).flatten()
            minibatch_advantages = advantages[minibatch_ind]
            minibatch_returns = returns[minibatch_ind]

            # importance sampling
            ratio = minibatch_newprob / (minibatch_oldprob + EPS)
            surr1 = ratio * minibatch_advantages

            surr2 = ratio.clamp(1 - clip_now, 1 + clip_now) * \
                minibatch_advantages
            # PPO2 policy net objective
            loss_surr = - torch.mean(torch.min(surr1, surr2))
            # not sure the value loss should be clipped as well
            # clip example: https://github.com/Jiankai-Sun/Proximal-Policy-Optimization-in-Pytorch/blob/master/ppo.py
            # however, it does not make sense to clip score-like value by a dimensionless clipping parameter
            # moreover, original paper does not mention clipped value

            if args.lossvalue_norm:
                loss_value = torch.mean(
                    (minibatch_newvalues - minibatch_returns).pow(2))
            else:
                loss_value = torch.mean(
                    (minibatch_newvalues - minibatch_returns).pow(2))

            loss_entropy = torch.mean(
                torch.log(minibatch_newprob + EPS) * minibatch_newprob)

            total_loss = loss_surr + args.loss_coeff_value * \
                loss_value + args.loss_coeff_entropy * loss_entropy
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        if args.schedule_clip == 'linear':
            ep_ratio = 1 - (i_episode / args.num_episode)
            clip_now = args.clip * ep_ratio

        if args.schedule_adam == 'linear':
            ep_ratio = 1 - (i_episode / args.num_episode)
            lr_now = args.lr * ep_ratio
            # set learning rate
            # ref: https://stackoverflow.com/questions/48324152/
            for g in optimizer.param_groups:
                g['lr'] = lr_now

        if i_episode % args.log_num_episode == 0:
            print(minibatch_newprob[0])
            print('Finished episode: {} Reward: {:.4f} total_loss = {:.4f} = {:.4f} + {} * {:.4f}'  # + {} * {:.4f}'
                  .format(i_episode, reward_record[-1]['meanepreward'], total_loss.data, loss_surr.data,
                          args.loss_coeff_value, loss_value.data))
            #  args.loss_coeff_entropy)
            # , loss_entropy.data))
            print('-----------------')

    return reward_record


def ppo_batch_train(args):
    cell_metadata_box, cell_num, rows_num, site_num = args.data
    row_height = 8
    site_width = 1
    cell_metadata_box.sort(key=lambda t: (
        math.floor(t[1] / row_height + 0.5), t[0] + t[2] // 2))

    ROW_MAX = 2
    WINDOW = 10
    POST_CELL_NUM = 30

    num_inputs = ROW_MAX * WINDOW * 5
    num_actions = ROW_MAX
    cluster_state_dimension = ROW_MAX * WINDOW * 5
    density_state_dimension = ROW_MAX * 1
    cell_state_dimension = POST_CELL_NUM * 5
    network = ActorCritic(cluster_state_dimension, density_state_dimension, cell_state_dimension, num_actions,
                          layer_norm=args.layer_norm)
    network = network.to(device)
    optimizer = opt.Adam(network.parameters(), lr=args.lr)
    reward_record = []
    env = gym.make(args.env_name, cell_box=cell_metadata_box, row_num=rows_num, site_num=site_num, row_height=8,
                   site_width=1)
    env.seed(args.seed)
    # record = ppo_train(args, network, optimizer, env,
    #                    ((5, 2, 10), (10,), (3, 30),), abacus_episode)
    # path_model = "./checkpoint/pretrained-model.pkl"
    # torch.save(network, path_model)
    record = ppo_train(args, network, optimizer, env,
                       ((5, 4, 10), (10,), (3, 30),), network_episode)
    path_model = "./checkpoint/model.pkl"
    torch.save(network, path_model)
    reward_record.append(record)

    return reward_record
