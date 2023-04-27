import mlflow
import torch
import numpy as np
from .db_loss_function_abstract import DBLossFunctionAbstract


class DBLossFunctionApprox(DBLossFunctionAbstract):
    def __init__(self, device='cpu', json=None, approx_size=0.01):
        super(DBLossFunctionApprox, self).__init__(device, json)
        mlflow.log_param('approx_size', approx_size)
        self.approx_size = approx_size

    def recalculate_centroids(self):
        dataset_size = len(self.data_loader.dataset)
        sample_size = int(np.ceil(dataset_size * self.approx_size))
        indx = np.random.randint(len(self.data_loader), size=sample_size)
        subset = torch.utils.data.Subset(self.data_loader.dataset, indx)
        testloader_subset = torch.utils.data.DataLoader(subset, batch_size=1024, num_workers=0, shuffle=False)
        with torch.no_grad():
            for i, data in enumerate(testloader_subset, 0):
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
        sample_size = int(np.ceil(dataset_size * self.approx_size))
        indx = np.random.randint(len(self.data_loader), size=sample_size)
        subset = torch.utils.data.Subset(self.data_loader.dataset, indx)
        testloader_subset = torch.utils.data.DataLoader(subset, batch_size=1024, num_workers=0, shuffle=False)
        with torch.no_grad():
            for i, data in enumerate(testloader_subset, 0):
                if i in indx:
                    inputs, labels = data
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model.forward(inputs)
                    outputs.to(self.device)
                    self.update_distances(outputs, labels)

