import torch
import torch.nn as nn
import torch.optim as optim
from torch import autograd
from torch.distributions import Categorical
import numpy as np
import math

from aimp.aifp.operation.macro_placer.rl_placer.agent.policy_based_agent import PolicyBasedAgent
from torch.utils.tensorboard import SummaryWriter

class PPO:
    def __init__(
        self,
        agent:PolicyBasedAgent,
        clip_param=0.1,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        initial_lr=2.5e-4,
        eps=1e-5,
        max_grad_norm=0.5, #0.5,
        use_clipped_value_loss=True,
        norm_adv=True,
        continues_action=False,
        optimizer='Adam',
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):

        self._clip_param = clip_param
        self._value_loss_coef = value_loss_coef
        self._entropy_coef = entropy_coef
        self._max_grad_norm = max_grad_norm
        self._use_clipped_value_loss = use_clipped_value_loss
        self._norm_adv = norm_adv
        self._continous_action = continues_action

        self._device = device
        self._agent = agent
        self._agent.to_device(device)
        if self._agent.is_shared:
            self._optimizer = self._get_optimizer(optimizer, initial_lr, eps)
        else:
            self._actor_optimizer, self._critic_optimizer = self._get_two_optimizer(optimizer, initial_lr, eps)

    def learn(self, rollout, minibatch_size, batch_size, update_epochs):
    # def learn(self, rollout, num_minibatches, batch_size, update_epochs):
        value_loss_epoch = 0
        action_loss_epoch = 0
        entropy_loss_epoch = 0
        # minibatch_size = int(batch_size // num_minibatches)

        indexes = np.arange(batch_size)
        for epoch in range(update_epochs):
            print('epoch ', epoch)
            np.random.shuffle(indexes)
            for start in range(0, batch_size, minibatch_size):
                print('ppo learn minibatch...')
                end = start + minibatch_size
                sample_idx = indexes[start:end]

                datas = rollout.sample_batch(sample_idx)
                # batch_obs_list = [torch.from_numpy(datas['obs']).squeeze(-1).to(self._device)] # atari test
                batch_obs_list = [
                    torch.from_numpy(datas['node_obs']).to(self._device),
                    torch.from_numpy(datas['macro_idx_to_place']).to(self._device),
                    torch.from_numpy(datas['sparse_adj_i']).to(self._device),
                    torch.from_numpy(datas['sparse_adj_j']).to(self._device),
                    torch.from_numpy(datas['sparse_adj_weight']).to(self._device),
                    torch.from_numpy(datas['action_mask']).to(self._device) ]

                batch_action = torch.from_numpy(datas['action']).squeeze(-1).to(self._device)
                batch_logprob = torch.from_numpy(datas['logprob']).squeeze(-1).to(self._device)
                batch_value = torch.from_numpy(datas['value']).squeeze(-1).to(self._device)
                batch_adv = torch.from_numpy(datas['adv']).squeeze(-1).to(self._device)
                batch_return = torch.from_numpy(datas['return']).squeeze(-1).to(self._device)

                # print('=====================================')
                # print('node_obs: ', batch_obs_list[0].shape, batch_obs_list[0].dtype)
                # print('macro_idx_to_place', batch_obs_list[1].shape, batch_obs_list[1].dtype)
                # print('sparse_adj_i: ', batch_obs_list[2].shape, batch_obs_list[2].dtype)
                # print('sparse_adj_j: ', batch_obs_list[3].shape, batch_obs_list[3].dtype)

                # print('batch_action: ', batch_action.shape, batch_action.dtype)
                # print('batch_logprob: ', batch_logprob.shape, batch_logprob.dtype)
                # print('batch_value: ', batch_value.shape, batch_logprob.dtype)
                # print('batch_adv: ', batch_adv.shape, batch_logprob.dtype)
                # print('batch_return: ', batch_return.shape, batch_return.dtype)

                # batch_obs, batch_action, batch_logprob, batch_adv, batch_return, batch_value = rollout.sample_batch(sample_idx)
                # batch_obs = torch.from_numpy(batch_obs).to(self._device)
                # batch_action = torch.from_numpy(batch_action).to(self._device)
                # batch_logprob = torch.from_numpy(batch_logprob).to(self._device)
                # batch_value = torch.from_numpy(batch_value).to(self._device)
                # batch_adv = torch.from_numpy(batch_adv).to(self._device)
                # batch_return  = torch.from_numpy(batch_return ).to(self._device)

                value_loss, action_loss, entropy_loss = self._learn_batch(batch_obs_list, batch_action, batch_value, batch_return,
                    batch_logprob, batch_adv)
                value_loss_epoch += value_loss
                action_loss_epoch += action_loss
                entropy_loss_epoch += entropy_loss

        update_steps = update_epochs * math.ceil(batch_size / minibatch_size)
        value_loss_epoch /= update_steps
        action_loss_epoch /= update_steps
        entropy_loss_epoch /= update_steps
        return value_loss_epoch, action_loss_epoch, entropy_loss_epoch

    def get_agent(self):
        return self._agent

    def _learn_batch(self, batch_obs_list, batch_action, batch_value, batch_return, batch_old_logprob, batch_adv, lr=None):

        if self._continous_action:
            raise NotImplementedError
        else:
            # action_logits = self._agent._model.policy(*batch_obs_list)
            # values = self._agent._model.value(*batch_obs_list)
            action_logits, values = self._agent.policy_value_forward(*batch_obs_list)
            dist = Categorical(logits=action_logits)
            action_log_probs = dist.log_prob(batch_action)
            dist_entropy = dist.entropy()
        entropy_loss = dist_entropy.mean()

        if self._norm_adv:
            batch_adv = (batch_adv - batch_adv.mean()) / (batch_adv.std() + 1e-8)
        ratio = torch.exp(action_log_probs - batch_old_logprob)
        surr1 = ratio * batch_adv
        surr2 = torch.clamp(ratio, 1.0 - self._clip_param, 1.0 + self._clip_param) * batch_adv
        action_loss = - torch.min(surr1, surr2).mean()

        values = values.view(-1)
        if self._use_clipped_value_loss:
            value_pred_clipped = batch_value + torch.clamp(values - batch_value, -self._clip_param, self._clip_param)
            value_losses = (values - batch_return).pow(2)
            value_losses_clipped = (value_pred_clipped - batch_return).pow(2)
            value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = 0.5 * (batch_return - values).pow(2).mean()

        # update actor and critic together
        if self._agent.is_shared:
            loss = value_loss * self._value_loss_coef + action_loss - entropy_loss * self._entropy_coef

            if lr:
                for param_group in self._optimizer.param_groups:
                    param_group['lr'] = lr
            
            self._optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self._agent.get_params(), self._max_grad_norm)
            self._optimizer.step()
    
        else:
            # update actor
            policy_loss = action_loss - entropy_loss * self._entropy_coef
            self._actor_optimizer.zero_grad()
            policy_loss.backward()

            # update critic
            self._critic_optimizer.zero_grad()
            value_loss.backward()

            policy_params, value_params = self._agent.get_params()
            nn.utils.clip_grad_norm_(policy_params, self._max_grad_norm)
            nn.utils.clip_grad_norm_(value_params, self._max_grad_norm)
            self._actor_optimizer.step()
            self._critic_optimizer.step()

        return value_loss.item(), action_loss.item(), self._entropy_coef * entropy_loss.item()


    def _get_optimizer(self, optimizer_name, initial_lr, eps):
        if optimizer_name == 'Adam':
            return optim.Adam(self._agent.get_params(), lr=initial_lr, eps=eps)
        else:
            return NotImplementedError

    def _get_two_optimizer(self, optimizer_name, initial_lr, eps):
        if optimizer_name == 'Adam':
            policy_params, value_params = self._agent.get_params()
            return optim.Adam(policy_params, lr=initial_lr, eps=eps),\
                   optim.Adam(value_params, lr=initial_lr, eps=eps),
        else:
            return NotImplementedError
