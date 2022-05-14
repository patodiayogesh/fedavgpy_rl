# DQN Agent

from src.DQN.dqn_model import DQN
from src.DQN.replay_memory import ReplayMemory, Transition
from torch import nn
import torch
from torch import optim
import random
import numpy as np


class Agent:

    def __init__(self, k=10):
        """
        Initialize agent variables

        :param k: No of clients to choose
        """
        self.batch_size = 32  # DQN batch size
        self.gamma = 0.999  # Q-Learning Gamma Value
        self.eps_start = 0.9
        self.eps_end = 0.01
        self.eps_decay = -0.05  # Greedy Epsilon decay rate
        self.target_update = 10  # Target DQN update rate
        self.mem_size = 100  # Memory Buffer Size
        self.k = k
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_clients = None
        self.target_accuracy = 0.99

        # for this setup step = round
        self.step_count = 0

        # Train And Target DQN
        self.train_network = DQN(10100, 100).to(self.device)
        self.target_network = DQN(10100, 100).to(self.device)
        self.target_network.load_state_dict(self.train_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.train_network.parameters())
        self.memory = ReplayMemory(self.mem_size)
        self.criterion = nn.SmoothL1Loss()

        self.last_state = None
        self.last_action = None
        self.last_reward = None

    def optimize_model(self):
        """
        Update Target DQN
        """

        if len(self.memory) < self.batch_size:
            return

        # Random sample from memory
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.stack(batch.state).to(self.device)
        action_batch = torch.stack(batch.action).to(self.device)
        next_state_batch = torch.stack(batch.next_state).to(self.device)
        reward_batch = torch.stack(batch.reward).to(self.device)

        state_action_vals = self.train_network(state_batch).gather(1, action_batch)
        next_state_action_vals = self.target_network(next_state_batch).max(1)[0].detach()
        next_state_action_vals = next_state_action_vals.view(next_state_action_vals.shape[0], 1)
        expected_state_action_values = (next_state_action_vals * self.gamma) + reward_batch

        # Minimize loss between obtained and expected state values
        loss = self.criterion(state_action_vals.unsqueeze(1), expected_state_action_values.unsqueeze(1))
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
        """
        Get Q-values of clients to choose k clients

        :param state: Client and Server model weights
        :return: K Clients and top client
        """
        sample = random.random()
        # Greedy Epsilon method
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
                [random.choices(range(state.shape[0] - 1), k=self.k)]
            )).view(-1, )

        top_action = k_actions[0].view(-1)
        top_action = top_action.type(torch.int64)
        k_actions = k_actions.numpy()
        return k_actions, top_action

    def select_k_agents(self, state):
        """
        Choose K clients from N clients

        :param state: Client and Server model weights
        :return: index of clients chosen
        """

        # Increment Round Number
        self.step_count += 1
        # self.train_network.eval()
        self.target_network.eval()
        # Push to Memory
        if self.last_state != None:
            self.memory.push(self.last_state, self.last_action, state, self.last_reward)

        k_client_ids, top_action = self.select_action(state)
        # Save to be pushed afterwards
        self.last_state = state
        self.last_action = top_action
        self.optimize_model()

        return k_client_ids

    def update_Q_network(self):
        # Set Train Network weights into Target Network
        self.target_network.load_state_dict(self.train_network.state_dict())

    def reward_function(self, accuracy):
        """
        Calculate Reward obtained

        :param accuracy: test accuracy obtained
        """
        # Reward Function
        reward = 64 ** (accuracy - self.target_accuracy) - 1
        self.last_reward = torch.Tensor(np.array([reward])).view(1)
