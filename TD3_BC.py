import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


import config
from normalizer import Normalizer
q_norm = Normalizer(1)
bc_norm = Normalizer(3)
l2_norm = Normalizer(3)

from state import RunningNorm
adv_norm = RunningNorm(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim - config.GOAL0_SIZE + config.GOAL1_SIZE, 256)
        self.l2 = nn.Linear(256, 256)
        self.l22 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        
        self.max_action = max_action
        

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = F.relu(self.l22(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l22 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l55 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)


    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = F.relu(self.l22(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = F.relu(self.l55(q2))
        q2 = self.l6(q2)
        return q1, q2


    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = F.relu(self.l22(q1))
        q1 = self.l3(q1)
        return q1

    def Q2(self, state, action):
        sa = torch.cat([state, action], 1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = F.relu(self.l55(q2))
        q2 = self.l6(q2)
        return q2


class TD3_BC(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        alpha=2.5,
    ):

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)#copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)#copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.alpha = alpha

        self.total_it = 0


    def select_action(self, state, train_mode=True):
        x = torch.FloatTensor(state.reshape(1, -1)).to(device)
        with torch.no_grad():
            action = self.actor(x).cpu().data.numpy().flatten()

        if train_mode:
            action += 0.2 * np.random.randn(len(action))
            action = action.clip(-1, 1)

            random_actions = np.random.uniform(low=-1, high=1, size=len(action))
            action += np.random.binomial(1, 0.3, 1)[0] * (random_actions - action)

        return action.clip(-1, 1)


    def train_critic(self, replay_buffer, get_goal, batch_size, use_norm, do_update):
        self.total_it += 1

        # Sample replay buffer 
        state, action, next_state, reward, discount, goal_, lp_, return_ = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            
#            assert all((a-b).abs().sum() < 1e-5 for a,b in zip(
#                    next_state[:, -config.GOAL0_SIZE:], get_goal(os)[:, -config.GOAL0_SIZE:])), "nope {} vs {}".format(
#                    next_state[0, -config.GOAL0_SIZE:], get_goal(os)[0, -config.GOAL0_SIZE:]
#                )
            next_action = (
                #self.actor_target(torch.cat([next_state[:, :-config.GOAL0_SIZE], goal], 1)) + noise
                self.actor_target(torch.cat([next_state[:, :-config.GOAL0_SIZE], get_goal(next_state)], 1)) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
#            target_Q = reward + not_done * self.discount * target_Q
            target_Q = reward + discount * target_Q
            if config.CLIP_Q: target_Q = target_Q.clamp(-1. / (1.-self.discount), 0)
                        

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)
#        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        l1 = current_Q1 - target_Q
        l2 = current_Q2 - target_Q

        if do_update:
            with torch.no_grad():
                adv_norm(torch.cat([l1.detach(), l2.detach()]), update=True)
            
        if use_norm:
            l1 = adv_norm(l1, update=False)
            l2 = adv_norm(l2, update=False)

        critic_loss = l1.pow(2).mean() + l2.pow(2).mean()
        # Compute critic loss

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        return critic_loss.item()

    def train_actor(self, replay_buffer, use_bc, get_goal, batch_size=256):
        # Delayed policy updates
        if self.total_it % self.policy_freq:
            return 0.

        # Sample replay buffer 
        #state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        state, action, next_state_, reward_, discount_, goal_, lp_, return_ = replay_buffer.sample(batch_size)

        # Compute actor loss
        pi = self.actor(torch.cat([ state[:, :-config.GOAL0_SIZE], get_goal(state) ], 1))
        Q = self.critic.Q1(state, pi)
        
        lmbda = self.alpha/Q.abs().mean().detach()
        if use_bc:  actor_loss = -Q.mean() * lmbda + F.mse_loss(pi, action)
        else: actor_loss = -Q.mean()
        actor_loss += pi.pow(2).mean()
            
        # Optimize the actor 
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        return actor_loss.item()

    def polyak(self):
# Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)


