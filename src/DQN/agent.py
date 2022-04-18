from dqn_model import DQN
from replay_memory import ReplayMemory
import torch
from torch import optim
import random
import math


class Agent:

    def __init__(self):

        self.batch_size = 128
        self.gamma = 0.999
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 200
        self.target_update = 10
        self.mem_size = 100
        self.k = 10
        self.device = 'cpu'

        # for this setup step = round
        self.step_count = 0

        self.train_network = DQN(10100, 100)
        self.target_network = DQN(10100, 100)
        self.target_network.load_state_dict(self.train_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.train_network.parameters())
        self.memory = ReplayMemory(self.mem_size)

    def train_select_action(self, state):

        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                        math.exp(-1. * self.step_count / self.eps_decay)
        self.step_count += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.

                # we need top-k q-values to select k devices so we chose top k
                return self.train_network(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(1)]], device=self.device, dtype=torch.long)

    def select_k_agents(self, state):

        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                        math.exp(-1. * self.step_count / self.eps_decay)
        self.step_count += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # we need top-k indices to select k devices so we chose top k
                return torch.topk(self.train_network(state), k=self.k, dim=1)[0].item()
        else:
            return torch.tensor([[random.randrange(self.k)]], device=self.device, dtype=torch.long)
