import torch
from .abstract_loss_function import AbstractLossFunction
from abc import abstractmethod


class DBLossFunctionAbstract(AbstractLossFunction):

    def __init__(self, device='cpu', json=None):
        super(DBLossFunctionAbstract, self).__init__()
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

    def set_recalculate_period(self, recalculate_period):
        self.recalculate_period = int(recalculate_period)

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
        self.count = torch.ones(self.class_number, 1).to(self.device) #for numeric stability?
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

    def update_centroids(self, predicted, target):
        self.sum.index_add_(0, target, predicted.float())
        count = torch.bincount(target).to(self.device)
        count = torch.nn.functional.pad(count, pad=(0, self.class_number - count.shape[0]))
        self.count[:, 0] += count

    def update_distances(self, predicted, target):
        t = torch.index_select(self.centroids, 0, target)
        t = torch.norm(t - predicted, dim=1)
        t = t.reshape(target.shape[0], 1)
        self.distances.index_add_(0, target, t)

    def calculate_centroids(self):
        self.centroids = self.sum / self.count

    @staticmethod
    def get_class_number(json):
        class_number = 0
        for i in json.keys():
            class_number += len(json[i])
        return class_number

    def calculate_distances_update(self, predicted, target, centroids):
        s = self.distances.detach().clone().to(self.device)  # coherence in class
        selected_counts = torch.index_select(self.count, 0, target)
        selected_centroids = torch.index_select(centroids, 0, target)
        centroids.index_add_(0, target, (predicted - selected_centroids) / selected_counts)
        pr = predicted / selected_counts
        vec = torch.linalg.vector_norm(selected_centroids - pr, dim=1)
        s.index_add_(0, target, vec.reshape(target.shape[0], 1))
        s = torch.sqrt(s)
        s = s / self.count

        return s

    def calculate_loss(self, predicted, target):
        centroids = self.centroids.detach().clone().to(self.device)
        s = self.calculate_distances_update(predicted, target, centroids)
        m = torch.cdist(centroids, centroids, p=2).to(self.device)  # class centrioids separation
        sum_ = torch.zeros(1).to(self.device)
        for i in range(0, self.class_number):
            for j in range(0, self.class_number):
                if i != j:
                    sum_ += self.class_weights_matrix[i][j] * (s[i] + s[j])/(m[i][j])
        loss = sum_ / self.class_number ** 2
        return loss

    def forward(self, predicted, target, epoch=0):
        self.epoch_count += 1
        if self.epoch_count > self.recalculate_period or self.sum is None:
            self.init_tensors(predicted, target)
            self.recalculate_centroids()
            self.recalculate_distances()
            self.epoch_count = 0
        loss = self.calculate_loss(predicted, target)
        return loss

    @abstractmethod
    def recalculate_centroids(self):
        raise NotImplementedError

    @abstractmethod
    def recalculate_distances(self):
        raise NotImplementedError
