from .db_loss_function_abstract import DBLossFunctionAbstract
import torch


class DBLossFunctionNaive(DBLossFunctionAbstract):
    def recalculate_centroids(self):
        with torch.no_grad():
            for i, data in enumerate(self.data_loader, 0):
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
        with torch.no_grad():
            for i, data in enumerate(self.data_loader, 0):
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