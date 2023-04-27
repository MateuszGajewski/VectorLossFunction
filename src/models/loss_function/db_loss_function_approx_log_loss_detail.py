from .db_loss_function_approx import DBLossFunctionApprox
import mlflow
import torch


class DBLossFunctionApproxLogLossDetail(DBLossFunctionApprox):
    def __init__(self, device='cpu', json=None, approx_size=0.05):
        super(DBLossFunctionApproxLogLossDetail, self).__init__(device, json, approx_size)

    def log_loss_details(self, predicted, target):
        centroids = self.centroids.detach().clone().to(self.device)
        s = self.calculate_distances_update(predicted, target, centroids)
        m = torch.cdist(centroids, centroids, p=2).to(self.device)  # class centrioids separation
        mlflow.log_metric('loss_function-points_distances', torch.sum(s))
        mlflow.log_metric('loss_function-centroids_distances', torch.sum(m))

    def freeze_centroids(self):
        ...

    def forward(self, predicted, target):
        self.epoch_count += 1
        if self.epoch_count > self.recalculate_period or self.sum is None:
            self.init_tensors(predicted, target)
            self.recalculate_centroids()
            self.recalculate_distances()
            self.epoch_count = 0
        loss = self.calculate_loss(predicted, target)
        self.log_loss_details(predicted, target)
        return loss
