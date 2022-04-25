from src.trainers.base import BaseTrainer
from src.models.model import choose_model
from src.models.worker import LrdWorker
from src.optimizers.gd import GD
from src.DQN.agent import Agent

import numpy as np
import torch

from sklearn.decomposition import PCA


criterion = torch.nn.CrossEntropyLoss()


class FedAvgDQNTrainer(BaseTrainer):
    """
    Original Scheme
    """
    def __init__(self, options, dataset):
        model = choose_model(options)
        self.move_model_to_gpu(model, options)

        self.optimizer = GD(model.parameters(), lr=options['lr'], weight_decay=options['wd'])
        self.num_epoch = options['num_epoch']
        self.agent = Agent()
        self.pca_model = None
        worker = LrdWorker(model, self.optimizer, options)
        super(FedAvgDQNTrainer, self).__init__(options, dataset, worker=worker)

    def train(self):
        print('>>> Select {} clients per round \n'.format(self.clients_per_round))
        self.agent.num_clients = len(self.clients)

        # Fetch latest flat model parameter
        self.latest_model = self.worker.get_flat_model_params().detach()

        """
        Step 1,2 : 
        Device downloads the initial random model
        Trains for 1 epoch. Report to server
        """
        for client in self.clients:
            client.set_flat_model_params(self.latest_model)
            client.local_train(n_epoch=1)

        for round_i in range(self.num_round):

            print(round_i)
            """
            Step 3,4:
            Q Values for all clients
            Choose K clients
            Train client for 1 epoch
            """
            selected_clients = self.select_clients(round_i)
            solns, stats = self.local_train(round_i, selected_clients, n_epoch=1)

            # Track communication cost
            self.metrics.extend_commu_stats(round_i, stats)

            # Update latest model
            self.latest_model = self.aggregate(solns)
            # self.optimizer.inverse_prop_decay_learning_rate(round_i)

            # Calculate reward obtained for current action
            server_model_accuracy = self.local_test()['acc']
            self.agent.reward_function(server_model_accuracy)

            # Update all clients with new weights and train them
            for client in self.clients:
                client.set_flat_model_params(self.latest_model)
                client.local_train()

            # Update Target DQN
            if round_i % self.agent.target_update == 0:
                self.update_network()

            self.test_latest_model_on_traindata(round_i)
            self.test_latest_model_on_evaldata(round_i)

        # Test final model on train data
        self.test_latest_model_on_traindata(self.num_round)
        self.test_latest_model_on_evaldata(self.num_round)

        # Save tracked information
        self.metrics.write()

    def update_network(self):
        self.agent.update_Q_network()


    def aggregate(self, solns):
        averaged_solution = torch.zeros_like(self.latest_model)
        accum_sample_num = 0
        for num_sample, local_solution in solns:
            accum_sample_num += num_sample
            averaged_solution += num_sample * local_solution
        averaged_solution /= self.all_train_data_num
        averaged_solution += (1-accum_sample_num/self.all_train_data_num) * self.latest_model
        return averaged_solution.detach()


    def get_client_weights(self, clients):

        client_weights = []
        for c in clients:
            c = c.get_flat_model_params()
            client_weights.append(c.cpu().detach().numpy())
        return np.array(client_weights)

    def select_clients(self, round_n):
        """
        Get weight of all clients. Compute Q values
        Choose top-K clients based on Q values
        """
        num_clients = min(self.clients_per_round, len(self.clients))
        client_weights = self.get_client_weights(self.clients)

        # Train PCA model on client weights during 1st round
        if round_n == 0:
            self.pca_model = PCA(n_components=100)
            self.pca_model.fit(client_weights)

        # Pass weight of server along-with all clients (101x100) to DQN
        client_pca_weights = self.pca_model.transform(client_weights)
        server_pca_weight = self.pca_model.transform(self.latest_model.cpu().detach().numpy().reshape(1,-1))
        agent_input_weight = torch.Tensor(np.concatenate([server_pca_weight, client_pca_weights]))
        k_clients = self.agent.select_k_agents(agent_input_weight)
        return [self.clients[int(i)] for i in k_clients]



