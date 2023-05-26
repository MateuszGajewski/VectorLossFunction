import mlflow
import numpy as np
import torch

from .db_loss_function_abstract import DBLossFunctionAbstract


class DBLossFunctionApproxCosine(DBLossFunctionAbstract):
    def __init__(self, device="cpu", json=None, approx_size=0.1):
        super(DBLossFunctionApproxCosine, self).__init__(device, json)
        mlflow.log_param("approx_size", approx_size)
        self.approx_size = approx_size

    def calculate_distances_update(self, predicted, target, centroids):
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        s = self.distances.detach().clone().to(self.device)  # coherence in class
        selected_counts = torch.index_select(self.count, 0, target)
        selected_centroids = torch.index_select(centroids, 0, target)
        centroids.index_add_(
            0, target, cos(predicted, selected_centroids) / selected_counts
        )
        pr = predicted / selected_counts
        vec = torch.linalg.vector_norm(selected_centroids - pr, dim=1)
        s.index_add_(0, target, vec.reshape(target.shape[0], 1))
        s = torch.sqrt(s)
        s = s / self.count

        return s

    def calculate_centroids_update(self, predicted, target, centroids):
        selected_counts = torch.index_select(self.count, 0, target)
        selected_centroids = torch.index_select(centroids, 0, target)
        centroids.index_add_(0, target, predicted / selected_counts)
        # vec = torch.linalg.vector_norm(centroids, dim=1)
        # sum_ = torch.sum(vec)
        return centroids

    def sim_matrix(self, c, eps=1e-8):
        """
        added eps for numerical stability
        """
        a_n = c.norm(dim=1)[:, None]
        a_norm = c / torch.max(a_n, eps * torch.ones_like(a_n))
        sim_mt = torch.mm(a_norm, a_norm.transpose(0, 1))
        return sim_mt

    def calculate_loss(self, predicted, target):
        centroids = self.centroids.detach().clone().to(self.device)
        centroids = self.calculate_centroids_update(predicted, target, centroids)
        s = self.calculate_distances_update(predicted, target, centroids)
        m = self.sim_matrix(centroids)
        sum_ = torch.zeros(1).to(self.device)
        for i in range(0, self.class_number):
            for j in range(0, self.class_number):
                if i != j:
                    sum_ += self.class_weights_matrix[i][j] * (s[i] + s[j]) / (m[i][j])
        loss = sum_ / self.class_number**2
        return loss

    def recalculate_centroids(self):
        dataset_size = len(self.data_loader.dataset)
        sample_size = int(np.ceil(dataset_size * self.approx_size))
        indx = np.random.randint(len(self.data_loader), size=sample_size)
        subset = torch.utils.data.Subset(self.data_loader.dataset, indx)
        testloader_subset = torch.utils.data.DataLoader(
            subset,
            batch_size=sample_size - 1,
            num_workers=0,
            shuffle=True,
            collate_fn=self.data_loader.collate_fn,
        )
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
        testloader_subset = torch.utils.data.DataLoader(
            subset,
            batch_size=sample_size - 1,
            num_workers=0,
            shuffle=False,
            collate_fn=self.data_loader.collate_fn,
        )
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
