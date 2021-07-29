import numpy as np
import torch

import config
import state
state_norm = state.GlobalNormalizerWithTime(config.STATE_SIZE)
state_norm.share_memory()

class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))

		self.discount = np.zeros((max_size, 1))

		self.goals = np.zeros((max_size, config.GOAL1_SIZE))
		self.lp = np.zeros((max_size, config.GOAL1_SIZE))
		self.ret = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, action, next_state, reward, discount, goal, lp, ret):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward

		self.discount[self.ptr] = discount

		self.goals[self.ptr] = goal
		self.lp[self.ptr] = lp
		self.ret[self.ptr] = ret

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	def sample_full(self):
		return self._sample(
            np.arange(self.size))

	def sample(self, batch_size):
		return self._sample(
            np.random.randint(0, self.size, size=batch_size))

	def _sample(self, ind):
		return (
			self.normalize_state(self.state[ind]),
			torch.FloatTensor(self.action[ind]).to(self.device),
			self.normalize_state(self.next_state[ind]),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.discount[ind]).to(self.device),
			torch.FloatTensor(self.goals[ind]).to(self.device),
			torch.FloatTensor(self.lp[ind]).to(self.device),
			torch.FloatTensor(self.ret[ind]).to(self.device),
		)

	def normalize_state(self, states, update=False):
		if not config.NORMALIZE:
			return states
		return state_norm(torch.from_numpy(states).to(self.device).float(), update)


