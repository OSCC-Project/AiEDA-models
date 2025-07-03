import numpy as np
from copy import deepcopy
from torch.optim import Adam
import torch
import models.DDPG.core as core
from torchviz import make_dot
import torch.nn.functional as F


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(
            size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(
            size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(
            size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}


class DDPG:
    def __init__(self, obs_dim, act_dim, act_bound, actor_critic=core.MLPActorCritic, seed=0,
                 replay_size=int(1e6), gamma=0.99, polyak=0.995, pi_lr=1e-3, q_lr=1e-3, act_noise=0.1):

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_bound = act_bound
        self.gamma = gamma
        self.polyak = polyak
        self.act_noise = act_noise

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.ac = actor_critic(obs_dim, act_dim)
        print("actor_critic\n", self.ac.q, self.ac.pi)
        # self.ac = actor_critic(obs_dim, act_dim, act_limit=2.0)
        self.ac_targ = deepcopy(self.ac)

        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.q_optimizer = Adam(self.ac.q.parameters(), lr=q_lr)

        for para in self.ac_targ.parameters():
            para.requires_grad = False

        self.replay_buffer = ReplayBuffer(
            obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    def compute_loss_q(self, data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q = self.ac.q(o, a)

        MyConvNetVis = make_dot(q)
        MyConvNetVis.format = "png"
        MyConvNetVis.directory = "AC_Q"
        # MyConvNetVis.view()

        # Bellman backup for Q function
        with torch.no_grad():
            q_pi_targ = self.ac_targ.q(o2, self.ac_targ.pi(o2))
            backup = r + self.gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        td_error = ((q - backup)**2).mean()

        return td_error

    def compute_loss_pi(self, data):
        o = data['obs']
        q_pi = self.ac.q(o, self.ac.pi(o))
        MyConvNetVis = make_dot(q_pi)
        MyConvNetVis.format = "png"
        MyConvNetVis.directory = "AC_Q_PI"
        return -q_pi.mean()
        # return -q_pi.min()
        # return -q_pi.max()

    def update(self, data):
        # First run one gradient descent step for Q.
        self.q_optimizer.zero_grad()
        loss_q = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        # Freeze Q-network so you don't waste computational effort
        # computing gradients for it during the policy learning step.
        for p in self.ac.q.parameters():
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for p in self.ac.q.parameters():
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def get_action(self, state, noise_scale):
        act = self.ac.act(torch.as_tensor(state, dtype=torch.float32))
        act += noise_scale * np.random.randn(self.act_dim)
        # return act
        return np.clip(act, self.act_bound[0], self.act_bound[1])

        # a = np.argmax(act, axis=1)

        # return a
        # return np.clip(act, self.act_bound[0], self.act_bound[1])
