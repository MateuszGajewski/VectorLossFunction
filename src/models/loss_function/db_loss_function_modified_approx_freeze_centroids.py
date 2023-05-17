import torch
from .db_loss_function_approx import DBLossFunctionApprox
from abc import abstractmethod
import mlflow


class DBLossFunctionModifiedApproxFreezeCentroids(DBLossFunctionApprox):

    def __init__(self, device='cpu', json=None):
        super(DBLossFunctionModifiedApproxFreezeCentroids, self).__init__(device, json)
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
        self.freeze_period = 2
        mlflow.log_param('freeze_period', self.freeze_period)

    def calculate_distances_update(self, predicted, target, centroids):
        s = torch.zeros(self.class_number, 1).to(self.device)
        selected_counts = torch.index_select(self.count, 0, target)
        selected_centroids = torch.index_select(centroids, 0, target)
        #centroids.index_add_(0, target, (predicted - selected_centroids) / selected_counts)
        pr = predicted# / selected_counts
        vec = torch.linalg.vector_norm(selected_centroids - pr, dim=1)
        s.index_add_(0, target, vec.reshape(target.shape[0], 1))
        s = torch.sqrt(s)
        s = s #/ self.count

        return s

    def calculate_centroids_update(self, predicted, target, centroids):
        selected_counts = torch.index_select(self.count, 0, target)
        selected_centroids = torch.index_select(centroids, 0, target)
        centroids.index_add_(0, target, predicted / selected_counts)
        # vec = torch.linalg.vector_norm(centroids, dim=1)
        # sum_ = torch.sum(vec)
        return centroids

    def calculate_loss(self, predicted, target):
        centroids = self.centroids.detach().clone().to(self.device)
        #centroids = self.calculate_centroids_update(predicted, target, centroids)
        s = self.calculate_distances_update(predicted, target, centroids)
        centroids = self.calculate_centroids_update(predicted, target, centroids)
        m = torch.cdist(centroids, centroids, p=2).to(self.device)  # class centrioids separation
        sum_ = torch.zeros(1).to(self.device)
        for i in range(0, self.class_number):
            for j in range(0, self.class_number):
                if i != j:
                    sum_ +=  (s[i] + s[j]) / (m[i][j])
                    #self.class_weights_matrix[i][j] *
        loss = sum_ / self.class_number ** 2
        return loss

    def calculate_loss_(self, predicted, target):
        centroids = self.centroids.detach().clone().to(self.device)
        #centroids = self.calculate_centroids_update(predicted, target, centroids)
        s = self.calculate_distances_update(predicted, target, centroids)
        centroids = self.calculate_centroids_update(predicted, target, centroids)
        m = torch.cdist(centroids, centroids, p=2).to(self.device)  # class centrioids separation
        sum_ = torch.zeros(1).to(self.device)
        for i in range(0, self.class_number):
            for j in range(0, self.class_number):
                if i != j:
                    sum_ +=  (s[i] + s[j]) #/ (m[i][j])
                    #self.class_weights_matrix[i][j] *
        loss = sum_ / self.class_number ** 2
        return loss




    def forward(self, predicted, target, epoch):
        self.epoch_count += 1
        if self.epoch_count > self.recalculate_period or self.sum is None:# or epoch == 0:
            self.init_tensors(predicted, target)
            self.recalculate_centroids()
            self.recalculate_distances()
            self.epoch_count = 0

        if epoch > self.freeze_period:
            loss = self.calculate_loss_(predicted, target)
            self.epoch_count = 0
        else:
            loss = self.calculate_loss(predicted, target)

        if self.log_loss:
            self.log_loss_details(predicted, target)
        return loss

