from abc import abstractmethod

import mlflow
import numpy as np
import torch

from src.models.loss_function.abstract_loss_function import AbstractLossFunction


class CoulombLossFunction(AbstractLossFunction):
    def __init__(self, device="cpu", json=None):
        super(CoulombLossFunction, self).__init__()
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
        self.approx_size = 0.1
        self.epsilon = 1
        #self.class_number =4

    def set_recalculate_period(self, recalculate_period):
        self.recalculate_period = int(recalculate_period)

    def set_log_loss(self, log_loss):
        self.log_loss = log_loss

    def assign_model(self, model):
        self.model = model

    def assign_data_loader(self, data_loader):
        self.data_loader = data_loader

    def get_centroids(self):
        self.calculate_centroids()
        return self.centroids

    def init_tensors(self, predicted, target):
        out_dimension = predicted.shape[1]
        self.sum = torch.zeros(self.class_number, out_dimension).to(self.device)
        self.count = torch.ones(self.class_number, 1).to(
            self.device
        )  # for numeric stability?
        self.centroids = 10*torch.rand(self.class_number, 1).to(self.device)
        self.distances = torch.zeros(self.class_number, 1).to(self.device)

    def build_class_weight_matrix(self, json):
        matrix = torch.ones((self.class_number, self.class_number))
        for i in range(0, self.class_number):
            for j in range(0, self.class_number):
                for key in json.keys():
                    if str(i) in json[key] and str(j) in json[key]:
                        matrix[i][j] = 1.5
        return matrix.to(self.device)

    def update_centroids(self, predicted, target):
        self.sum.index_add_(0, target, predicted.float())
        count = torch.bincount(target).to(self.device)
        count = torch.nn.functional.pad(
            count, pad=(0, self.class_number - count.shape[0])
        )
        self.count[:, 0] += count

    def update_distances(self, predicted, target):
        t = torch.index_select(self.centroids, 0, target)
        t = torch.norm(t - predicted, dim=1)
        t = t.reshape(target.shape[0], 1)
        self.distances.index_add_(0, target, t)

    def calculate_centroids(self):
        self.centroids = self.sum / self.count
        #print(self.centroids)

    @staticmethod
    def get_class_number(json):
        class_number = 0
        for i in json.keys():
            class_number += len(json[i])
        return class_number

    def calculate_squared_distances(self, a, b):
        """returns the squared distances between all elements in a and in b as a matrix
        of shape #a * #b"""
        na = a.data.shape[0]
        nb = b.data.shape[0]
        dim = a.data.shape[-1]
        a = a.view([na, 1, -1])
        b = b.view([1, nb, -1])
        d = a - b
        return (d * d).sum(2)

    def plummer_kernel(self, a, b, dimension, epsilon):
        r = self.calculate_squared_distances(a, b)
        r += epsilon * epsilon
        f1 = dimension - 2
        return torch.pow(r, -f1 / 2)

    def calculate_loss(self, predicted, target):
        centroids = self.centroids.detach().clone().to(self.device)
        sum = torch.zeros(1).to(self.device)
        for c in range(0, self.class_number):
            idx = (target == c).nonzero(as_tuple=False)
            if idx.shape[0] > 0:
                examples = predicted[idx]
                p = self.plummer_kernel(
                    centroids[c].unsqueeze(0), examples, 3, self.epsilon
                )
                p2 = self.plummer_kernel(
                    torch.zeros((1, predicted.shape[1])).to(self.device),
                    examples,
                    3,
                    self.epsilon,
                )
                for c2 in range(0, self.class_number):
                    if c2 != c:
                        p1 = self.plummer_kernel(
                            examples, centroids[c2].unsqueeze(0), 3, self.epsilon
                        )
                        sum += p1.sum(0) / (10 * idx.shape[0])

                sum -= p.sum(1) / idx.shape[0]
                sum -= p2.sum(1) / (100000 * idx.shape[0])
        #print(sum)
        return sum + 100

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
            self.init_tensors(predicted, target)
            self.recalculate_centroids()
            self.recalculate_distances()
            self.epoch_count = 0
        loss = self.calculate_loss(predicted, target)
        if self.log_loss:
            ...
            # self.log_loss_details(predicted, target)

        return loss

    def recalculate_centroids(self):
        dataset_size = len(self.data_loader.dataset)
        sample_size = int(np.ceil(dataset_size * self.approx_size))
        indx = np.random.randint(len(self.data_loader), size=sample_size)
        subset = torch.utils.data.Subset(self.data_loader.dataset, indx)
        testloader_subset = torch.utils.data.DataLoader(
            subset,
            batch_size=sample_size - 1,
            num_workers=0,
            shuffle=True,
            collate_fn=self.data_loader.collate_fn,
        )
        with torch.no_grad():
            for i, data in enumerate(testloader_subset, 0):
                if len(data) == 2:
                    inputs, labels = data
                else:
                    labels, inputs, offsets = data
                    offsets = offsets.to(self.device)
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                if len(data) == 2:
                    outputs = self.model.forward(inputs)
                else:
                    outputs = self.model.forward(inputs, offsets)
                outputs.to(self.device)
                self.update_centroids(outputs, labels)
        self.calculate_centroids()

    def recalculate_distances(self):
        dataset_size = len(self.data_loader.dataset)
        sample_size = int(np.ceil(dataset_size * self.approx_size))
        indx = np.random.randint(len(self.data_loader), size=sample_size)
        subset = torch.utils.data.Subset(self.data_loader.dataset, indx)
        testloader_subset = torch.utils.data.DataLoader(
            subset,
            batch_size=sample_size - 1,
            num_workers=0,
            shuffle=False,
            collate_fn=self.data_loader.collate_fn,
        )
        with torch.no_grad():
            for i, data in enumerate(testloader_subset, 0):
                if len(data) == 2:
                    inputs, labels = data
                else:
                    labels, inputs, offsets = data
                    offsets = offsets.to(self.device)
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                if len(data) == 2:
                    outputs = self.model.forward(inputs)
                else:
                    outputs = self.model.forward(inputs, offsets)
                outputs.to(self.device)
                self.update_distances(outputs, labels)
