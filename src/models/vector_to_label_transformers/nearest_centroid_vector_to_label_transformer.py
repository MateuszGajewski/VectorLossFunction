import torch
import torch.nn as nn


class NearestCentroidVectorToLabelTransformer:
    def __init__(self):
        self.centroids = None

    def fit(self, criterion):
        self.centroids = criterion.get_centroids()

    def predict(self, input_batch):
        dist = torch.cdist(self.centroids, input_batch)
        labels = torch.argmin(dist.t(), dim=1)
        return labels
