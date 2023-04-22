from abc import ABC, abstractmethod
import torch.nn as nn


class AbstractLossFunction(nn.Module):
    @abstractmethod
    def forward(self, predicted, targets):
        raise NotImplementedError
