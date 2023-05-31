from abc import abstractmethod

import mlflow
import numpy as np
import torch

from src.models.loss_function.abstract_loss_function import AbstractLossFunction


class DotLossFunctionCentroids(AbstractLossFunction):
    def __init__(self, device="cpu", json=None):
        super(DotLossFunctionCentroids, self).__init__()
        self.device = device
        self.data_loader = None
        self.model = None
        self.distances = None
        self.sum = None
        self.count = None
        self.centroids = None
        self.class_number = self.get_class_number(json)
        self.class_weights_matrix = self.build_class_weight_matrix(json)
        self.epoch_count = 0
        self.recalculate_period = 0
        self.log_loss = False
        self.gamma = 1.5

    def set_recalculate_period(self, recalculate_period):
        self.recalculate_period = int(recalculate_period)

    def set_log_loss(self, log_loss):
        self.log_loss = log_loss

    def assign_model(self, model):
        self.model = model

    def assign_data_loader(self, data_loader):
        self.data_loader = data_loader

    def get_centroids(self):
        counts = torch.zeros(self.class_number, 1)
        dataset_size = len(self.data_loader.dataset)
        sample_size = int(np.ceil(dataset_size * 0.3))
        indx = np.random.randint(len(self.data_loader), size=sample_size)
        subset = torch.utils.data.Subset(self.data_loader.dataset, indx)
        testloader_subset = torch.utils.data.DataLoader(
            subset, batch_size=sample_size - 1, num_workers=0, shuffle=False
        )
        sums = None
        with torch.no_grad():
            for i, data in enumerate(testloader_subset, 0):
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model.forward(inputs)
                outputs.to(self.device)
                if sums is None:
                    sums = torch.zeros(self.class_number, outputs.shape[1])

                sums.index_add_(0, labels, outputs.float())
                count = torch.bincount(labels).to(self.device)
                count = torch.nn.functional.pad(
                    count, pad=(0, self.class_number - count.shape[0])
                )
                counts[:, 0] += count
        avg = sums / counts
        self.centroids = avg
        return avg

    def init_tensors(self, predicted, target):
        out_dimension = predicted.shape[1]
        self.sum = torch.zeros(self.class_number, out_dimension).to(self.device)
        self.count = torch.ones(self.class_number, 1).to(
            self.device
        )  # for numeric stability?
        self.centroids = torch.zeros(self.class_number, 1).to(self.device)
        self.distances = torch.zeros(self.class_number, 1).to(self.device)

    def build_class_weight_matrix(self, json):
        matrix = torch.ones((self.class_number, self.class_number))
        for i in range(0, self.class_number):
            for j in range(0, self.class_number):
                for key in json.keys():
                    if str(i) in json[key] and str(j) in json[key]:
                        matrix[i][j] = 1.5
        return matrix.to(self.device)

    @staticmethod
    def get_class_number(json):
        class_number = 0
        for i in json.keys():
            class_number += len(json[i])
        return class_number

    def calculate_loss(self, predicted, target):
        alphas = torch.zeros(1).to(self.device)
        betas = torch.zeros(1).to(self.device)
        centroids = self.get_centroids()
        for c in range(0, self.class_number):
            idx = (target == c).nonzero(as_tuple=False)
            idx_neg = (target != c).nonzero(as_tuple=False)
            if idx.shape[0] > 0:
                examples = predicted[idx]
                examples_neg = predicted[idx_neg]

                examples_normalized = torch.nn.functional.normalize(
                    examples.squeeze(1)
                ).squeeze(1)
                examples_neg_normalized = torch.nn.functional.normalize(
                    examples_neg.squeeze(1)
                ).squeeze(1)

                alphas += (
                    (idx.shape[0] / target.shape[0])
                    * (1 + (torch.mm(examples_normalized, centroids[c].T).sum()))
                    / 2
                    * ((examples_normalized.shape[0]))
                )

                betas += (
                    (1 + torch.mm(examples_neg_normalized, examples_normalized.T).sum())
                    / 2
                    * (examples_normalized.shape[0] * examples_neg_normalized.shape[0])
                )
        return (self.gamma - alphas) / (self.gamma - betas)

    def log_loss_details(self, predicted, target):
        centroids = self.centroids.detach().clone().to(self.device)
        s = self.calculate_distances_update(predicted, target, centroids)
        m = torch.cdist(centroids, centroids, p=2).to(
            self.device
        )  # class centrioids separation
        mlflow.log_metric("loss_function-points_distances", torch.sum(s))
        mlflow.log_metric("loss_function-centroids_distances", torch.sum(m))

    def forward(self, predicted, target, epoch=0):
        self.epoch_count += 1
        if self.epoch_count > self.recalculate_period or self.sum is None:
            self.epoch_count = 0
        loss = self.calculate_loss(predicted, target)
        if self.log_loss:
            ...
            # self.log_loss_details(predicted, target)

        return loss
