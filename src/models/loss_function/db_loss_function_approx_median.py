import numpy as np
import torch
from geom_median.torch import compute_geometric_median
# PyTorch API
from .db_loss_function_approx import DBLossFunctionApprox


class DBLossFunctionApproxMedian(DBLossFunctionApprox):
    def __init__(self, device='cpu', json=None, approx_size=0.05):
        super(DBLossFunctionApproxMedian, self).__init__(device, json, approx_size)

    def calculate_centroids(self):
        dataset_size = len(self.data_loader.dataset)
        sample_size = int(np.ceil(dataset_size * self.approx_size))
        indx = np.random.randint(len(self.data_loader), size=sample_size)
        subset = torch.utils.data.Subset(self.data_loader.dataset, indx)
        testloader_subset = torch.utils.data.DataLoader(subset, batch_size=sample_size, num_workers=0,
                                                        shuffle=True,
                                                        collate_fn = self.data_loader.collate_fn
                                                        )
        with torch.no_grad():
            for i, data in enumerate(testloader_subset, 0):
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model.forward(inputs)
                outputs.to(self.device)
                reults = []
                for i in range(0, self.class_number):
                    indices = labels == i
                    indices = indices.nonzero()
                    indices = torch.squeeze(indices)
                    selected = torch.index_select(outputs, 0, indices)
                    m = compute_geometric_median(selected)
                    reults.append(m.median)
                results = torch.stack(reults)
                self.centroids = results
