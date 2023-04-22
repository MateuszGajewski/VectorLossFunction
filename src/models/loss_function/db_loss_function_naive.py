from .db_loss_function_abstract import DBLossFunctionAbstract
import torch


class DBLossFunctionNaive(DBLossFunctionAbstract):
    def recalculate_centroids(self):
        with torch.no_grad():
            for i, data in enumerate(self.data_loader, 0):
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model.forward(inputs)
                outputs.to(self.device)
                self.update_centroids(outputs, labels)
        self.calculate_centroids()

    def recalculate_distances(self):
        with torch.no_grad():
            for i, data in enumerate(self.data_loader, 0):
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model.forward(inputs)
                outputs.to(self.device)
                self.update_distances(outputs, labels)