import mlflow
import torch
import numpy as np
from .db_loss_function_abstract import DBLossFunctionAbstract


class DBLossFunctionApprox(DBLossFunctionAbstract):
    def __init__(self, device='cpu', json=None, approx_size=0.5):
        super(DBLossFunctionApprox, self).__init__(device, json)
        mlflow.log_param('approx_size', approx_size)
        self.approx_size = approx_size

    def recalculate_centroids(self):
        dataset_size = len(self.data_loader.dataset)
        sample_size = int(np.ceil(dataset_size * self.approx_size))
        indx = np.random.randint(len(self.data_loader), size=sample_size)
        subset = torch.utils.data.Subset(self.data_loader.dataset, indx)
        testloader_subset = torch.utils.data.DataLoader(subset, batch_size=sample_size-1,
                                                        num_workers=0, shuffle=True,
                                                        collate_fn = self.data_loader.collate_fn)
        with torch.no_grad():
            for i, data in enumerate(testloader_subset, 0):
                if len(data) == 2:
                    inputs, labels = data
                else:
                    labels, inputs, offsets = data
                    offsets = offsets.to(self.device)
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                if len(data) == 2:
                    outputs = self.model.forward(inputs)
                else:
                    outputs = self.model.forward(inputs, offsets)
                outputs.to(self.device)
                self.update_centroids(outputs, labels)
        self.calculate_centroids()

    def recalculate_distances(self):
        dataset_size = len(self.data_loader.dataset)
        sample_size = int(np.ceil(dataset_size * self.approx_size))
        indx = np.random.randint(len(self.data_loader), size=sample_size)
        subset = torch.utils.data.Subset(self.data_loader.dataset, indx)
        testloader_subset = torch.utils.data.DataLoader(subset, batch_size=sample_size-1,
                                                        num_workers=0, shuffle=False,
                                                        collate_fn=self.data_loader.collate_fn)
        with torch.no_grad():
            for i, data in enumerate(testloader_subset, 0):
                if len(data) == 2:
                    inputs, labels = data
                else:
                    labels, inputs, offsets = data
                    offsets = offsets.to(self.device)
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                if len(data) == 2:
                    outputs = self.model.forward(inputs)
                else:
                    outputs = self.model.forward(inputs, offsets)
                outputs.to(self.device)
                self.update_distances(outputs, labels)

