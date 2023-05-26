from abc import ABC, abstractmethod

import torch.nn as nn


class AbstractLossFunction(nn.Module):
    @abstractmethod
    def __init__(self, device="cpu", json=None):
        super(AbstractLossFunction, self).__init__()

    @abstractmethod
    def forward(self, predicted, targets, epoch=0):
        raise NotImplementedError

    def assign_data_loader(self, data_loader):
        pass

    def assign_model(self, model):
        pass
