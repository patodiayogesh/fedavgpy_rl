from dqn_model import DQN
from replay_memory import ReplayMemory
import torch
from torch import optim
from torch.autograd import Variable
import random
import math


class Agent:

    def __init__(self, k=10):

        self.batch_size = 32
        self.gamma = 0.999
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 200
        self.target_update = 10
        self.mem_size = 100
        self.k = k
        self.device = 'cpu'
        self.num_clients = None
        self.target_accuracy = 0.92

        # for this setup step = round
        self.step_count = 0

        self.train_network = DQN(10100, 100)
        self.target_network = DQN(10100, 100)
        self.target_network.load_state_dict(self.train_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.train_network.parameters())
        self.memory = ReplayMemory(self.mem_size)

        self.last_state = None
        self.last_action = None
        self.last_reward = None

    def update_Q_network(self):

        self.train_network.eval()
        self.train_network.eval()

        with torch.no_grad():
            pass

    def select_k_agents(self, state):

        # Set to Evaluation Phase
        self.train_network.eval()
        self.train_network.eval()

        # Push to Memory
        if self.last_state:
            self.memory.push(self.last_state, self.last_action, state, self.last_reward)

        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                        math.exp(-1. * self.step_count / self.eps_decay)
        self.step_count += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # we need top-k indices to select k devices, so we chose top k
                action_values = self.train_network(state)
                k_actions = torch.topk(action_values, k=self.k, dim=1).indices().item()
                top_action = k_actions[0]

        else:
            k_actions = torch.tensor([[random.choices(state.shape[0], k=self.k)]],
                                     device=self.device)
            top_action = k_actions[0]

        self.last_state = state
        self.last_action = top_action
        if self.step_count % self.target_update:
            self.update_Q_network()

        return k_actions


