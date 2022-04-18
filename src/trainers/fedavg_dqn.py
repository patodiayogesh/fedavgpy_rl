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
        worker = LrdWorker(model, self.optimizer, options)
        super(FedAvgDQNTrainer, self).__init__(options, dataset, worker=worker)

    def train(self):
        print('>>> Select {} clients per round \n'.format(self.clients_per_round))

        # Fetch latest flat model parameter
        self.latest_model = self.worker.get_flat_model_params().detach()
        for client in self.clients:
            client.set_model_params(self.latest_model)
            # update base.py and client local_train to accept n_epoch = 1
            client.local_train(n_epoch=1)

        for round_i in range(self.num_round):

            # Test latest model on train data
            self.test_latest_model_on_traindata(round_i)
            self.test_latest_model_on_evaldata(round_i)

            # Choose K clients prop to data size
            selected_clients = self.select_clients(round_i)

            # Solve minimization locally
            solns, stats = self.local_train(round_i, selected_clients, n_epoch=1)

            # Track communication cost
            self.metrics.extend_commu_stats(round_i, stats)

            # Update latest model
            self.latest_model = self.aggregate(solns)
            self.optimizer.inverse_prop_decay_learning_rate(round_i)

            for client in self.clients:
                client.set_model_params(self.latest_model)
                client.local_train()

        # Test final model on train data
        self.test_latest_model_on_traindata(self.num_round)
        self.test_latest_model_on_evaldata(self.num_round)

        # Save tracked information
        self.metrics.write()


    def aggregate(self, solns):
        averaged_solution = torch.zeros_like(self.latest_model)
        accum_sample_num = 0
        for num_sample, local_solution in solns:
            accum_sample_num += num_sample
            averaged_solution += num_sample * local_solution
        averaged_solution /= self.all_train_data_num
        averaged_solution += (1-accum_sample_num/self.all_train_data_num) * self.latest_model
        return averaged_solution.detach()

    def train_pca_model(self, input):
        weight_vecs = []
        for _, weight in input:
            weight_vecs.extend(weight.flatten())

        return PCA.fit(weight_vecs, n_components=2)

    def select_clients(self, round_n):

        num_clients = min(self.clients_per_round, len(self.clients))
        client_weights = [c.get_model_params for c in self.clients]

        if round_n == 1:
            self.pca_model = self.train_pca_model(client_weights)

        client_pca_weights = self.pca_model.transform(client_weights)
        k_clients = self.agent.select_k_agents(client_pca_weights)
        return [self.clients[i] for i in k_clients]



