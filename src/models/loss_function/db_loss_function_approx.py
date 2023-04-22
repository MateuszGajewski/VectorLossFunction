import torch
import torch.nn as nn
import numpy as np
from .db_loss_function_abstract import DBLossFunctionAbstract


class DBLossFunctionApprox(DBLossFunctionAbstract):

    def recalculate_centroids(self):
        dataset_size = len(self.data_loader.dataset)
        batch_num = len(self.data_loader)
        sample_size = int(np.ceil(dataset_size * 0.01 / (dataset_size/batch_num)))
        indx = np.random.randint(len(self.data_loader), size=sample_size)
        with torch.no_grad():
            for i, data in enumerate(self.data_loader, 0):
                if i in indx:
                    inputs, labels = data
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model.forward(inputs)
                    outputs.to(self.device)
                    self.update_centroids(outputs, labels)
        self.calculate_centroids()

    def recalculate_distances(self):
        dataset_size = len(self.data_loader.dataset)
        batch_num = len(self.data_loader)
        sample_size = int(np.ceil(dataset_size * 0.01 / (dataset_size / batch_num)))
        indx = np.random.randint(len(self.data_loader), size=sample_size)
        with torch.no_grad():
            for i, data in enumerate(self.data_loader, 0):
                if i in indx:
                    inputs, labels = data
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model.forward(inputs)
                    outputs.to(self.device)
                    self.update_distances(outputs, labels)

