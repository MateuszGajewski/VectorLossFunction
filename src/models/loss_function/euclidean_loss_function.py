import torch
import torch.nn as nn


class EuclideanLossFunction(nn.Module):

    #def __init__(self) -> None:
    #    super(EuclideanLoss, self).__init__()

    def forward(self, predicted, target):
        loss = torch.norm(predicted-target)
        return loss