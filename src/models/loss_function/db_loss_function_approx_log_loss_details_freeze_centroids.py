from .db_loss_function_approx import DBLossFunctionApprox
import mlflow
import torch


class DBLossFunctionApproxLogLossDetailFreezeCentroids(DBLossFunctionApprox):
    def __init__(self, device='cpu', json=None, approx_size=0.05):
        super(DBLossFunctionApproxLogLossDetailFreezeCentroids, self).\
            __init__(device, json, approx_size)
        self.freeze_period = 2
        mlflow.log_param('freeze_period', self.freeze_period)


    def log_loss_details(self, predicted, target):
        centroids = self.centroids.detach().clone().to(self.device)
        s = self.calculate_distances_update(predicted, target, centroids)
        m = torch.cdist(centroids, centroids, p=2).to(self.device)  # class centrioids separation
        mlflow.log_metric('loss_function-points_distances', torch.sum(s))
        mlflow.log_metric('loss_function-centroids_distances', torch.sum(m))

    def calculate_freeze_loss(self, predicted, target):
        centroids = self.centroids.detach().clone().to(self.device)
        s = self.calculate_distances_update(predicted, target, centroids)
        #m = torch.cdist(centroids, centroids, p=2).to(self.device)  # class centrioids separation
        sum_ = torch.zeros(1).to(self.device)
        for i in range(0, self.class_number):
            for j in range(0, self.class_number):
                if i != j:
                    sum_ += self.class_weights_matrix[i][j] * (s[i] + s[j])#/(m[i][j])
        loss = sum_ / self.class_number ** 2
        return loss

    def forward(self, predicted, target, epoch=0):
        self.epoch_count += 1
        if self.epoch_count > self.recalculate_period or self.sum is None:
            self.init_tensors(predicted, target)
            self.recalculate_centroids()
            self.recalculate_distances()
            self.epoch_count = 0
        if epoch < self.freeze_period:
            loss = self.calculate_loss(predicted, target)
        else:
            loss = self.calculate_freeze_loss(predicted, target)
        self.log_loss_details(predicted, target)
        return loss
