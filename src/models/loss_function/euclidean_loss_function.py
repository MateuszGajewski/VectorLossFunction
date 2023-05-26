import torch
from .abstract_loss_function import AbstractLossFunction


class EuclideanLossFunction(AbstractLossFunction):
    def __init__(self, device='cpu', json=None) -> None:
        super(EuclideanLossFunction, self).__init__()
        self.class_number = self.get_class_number(json)

    def get_centroids(self):
        centroids = torch.tensor([range(0, self.class_number)])
        centroids = centroids.reshape(self.class_number, 1)
        return centroids.float()

    @staticmethod
    def get_class_number(json):
        class_number = 0
        for i in json.keys():
            class_number += len(json[i])
        return class_number

    def forward(self, predicted, target):
        target = target.reshape(target.shape[0], 1)
        loss = torch.linalg.vector_norm(target - predicted, dim=1)
        loss = torch.sum(loss)
        return loss
