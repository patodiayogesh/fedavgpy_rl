from src.DQN.dqn_model import DQN
from src.DQN.replay_memory import ReplayMemory, Transition
from torch import nn
import torch
from torch import optim
from torch.autograd import Variable
import random
import numpy as np


class Agent:

    def __init__(self, k=10):

        self.batch_size = 16
        self.gamma = 0.999
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = -0.005
        self.optimize_steps = 10
        self.target_update = 10
        self.mem_size = 100
        self.k = k
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_clients = None
        self.target_accuracy = 0.99

        # for this setup step = round
        self.step_count = 0

        self.train_network = DQN(10100, 100)
        self.target_network = DQN(10100, 100)
        self.target_network.load_state_dict(self.train_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.train_network.parameters())
        self.memory = ReplayMemory(self.mem_size)
        self.criterion = nn.SmoothL1Loss()

        self.last_state = None
        self.last_action = None
        self.last_reward = None

    def optimize_model(self):

        # self.train_network.eval()
        # self.train_network.eval()
        print('optimize')
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.stack(batch.state).to(self.device)
        action_batch = torch.stack(batch.action).to(self.device)
        next_state_batch = torch.stack(batch.next_state).to(self.device)
        reward_batch = torch.stack(batch.reward).to(self.device)

        state_action_vals = self.train_network(state_batch).gather(1, action_batch)
        next_state_action_vals = self.target_network(next_state_batch).max(1)[0].detach()
        next_state_action_vals = next_state_action_vals.view(next_state_action_vals.shape[0],1)
        expected_state_action_values = (next_state_action_vals*self.gamma) + reward_batch

        loss = self.criterion(state_action_vals, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.train_network.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        del state_batch
        del action_batch
        del next_state_batch
        del reward_batch

    def select_action(self, state):

        sample = random.random()
        print('step count', self.step_count)
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                        np.exp(self.step_count * self.eps_decay)

        print(eps_threshold)
        if sample > eps_threshold:
            with torch.no_grad():
                # we need top-k indices to select k devices, so we chose top k
                action_values = self.train_network(state.view(1,
                                                              state.shape[0],
                                                              state.shape[1])
                                                   .to(self.device))
                k_actions = torch.topk(action_values, k=self.k, dim=1).indices.cpu().detach().view(-1)
                del action_values

        else:
            k_actions = torch.Tensor(np.array(
                [random.choices(range(state.shape[0]-1), k=self.k)]
            )).view(-1,)

        top_action = k_actions[0].view(-1)
        top_action = top_action.type(torch.int64)
        k_actions = k_actions.numpy()
        return k_actions, top_action

    def select_k_agents(self, state):

        self.step_count += 1
        # self.train_network.eval()
        self.target_network.eval()
        # Push to Memory
        if self.last_state != None:
            self.memory.push(self.last_state, self.last_action, state, self.last_reward)

        k_client_ids, top_action = self.select_action(state)

        self.last_state = state
        self.last_action = top_action
        self.optimize_model()

        return k_client_ids

    def update_Q_network(self):
        self.target_network.load_state_dict(self.train_network.state_dict())

    def reward_function(self, accuracy):

        reward = 64**(accuracy - self.target_accuracy)-1
        self.last_reward = torch.Tensor(np.array([reward])).view(1)


