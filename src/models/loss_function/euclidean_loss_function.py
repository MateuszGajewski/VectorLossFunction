import torch
import torch.nn as nn


class EuclideanLossFunction(nn.Module):

    #def __init__(self) -> None:
    #    super(EuclideanLoss, self).__init__()

    def forward(self, predicted, target):
        loss = torch.sqrt((predicted - target) ** 2).sum()
        return loss