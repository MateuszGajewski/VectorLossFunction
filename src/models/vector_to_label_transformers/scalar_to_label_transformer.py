import numpy as np
import torch
import torch.nn as nn


class ScalarToLabelTransformer:
    def __init__(self):
        self.avg = None

    def fit(self, criterion):
        counts = torch.zeros(criterion.class_number, 1)
        dataset_size = len(criterion.data_loader.dataset)
        sample_size = int(np.ceil(dataset_size * 0.8))
        indx = np.random.randint(len(criterion.data_loader), size=sample_size)
        subset = torch.utils.data.Subset(criterion.data_loader.dataset, indx)
        testloader_subset = torch.utils.data.DataLoader(
            subset, batch_size=sample_size - 1, num_workers=0, shuffle=False
        )
        sums = None
        with torch.no_grad():
            for i, data in enumerate(testloader_subset, 0):
                inputs, labels = data
                inputs = inputs.to(criterion.device)
                labels = labels.to(criterion.device)
                outputs = criterion.model.forward(inputs)
                outputs.to(criterion.device)
                if sums is None:
                    sums = torch.zeros(criterion.class_number, outputs.shape[1])

                sums.index_add_(0, labels, outputs.float())
                count = torch.bincount(labels).to(criterion.device)
                count = torch.nn.functional.pad(
                    count, pad=(0, criterion.class_number - count.shape[0])
                )
                counts[:, 0] += count
        self.avg = sums / counts

    def predict(self, input_batch):
        dot_products = torch.mm(
            torch.nn.functional.normalize(input_batch),
            torch.nn.functional.normalize(self.avg).T,
        )

        labels = torch.argmax(dot_products, dim=1)
        return labels
