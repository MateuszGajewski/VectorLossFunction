from .abstract_metric import AbstractMetric
import torch


class WeightedMultiClassAccuracy(AbstractMetric):
    def __init__(self, json, device='cpu'):
        super(WeightedMultiClassAccuracy, self).__init__(json, device)
        self.group_confusion_weight = 0.5
        self.weight_matrix = self.build_class_weight_matrix(json)

    def build_class_weight_matrix(self, json):
        matrix = torch.eye(self.class_number)
        for i in range(0, self.class_number):
            for j in range(0, self.class_number):
                for key in json.keys():
                    if str(i) in json[key] and str(j) in json[key]:
                        if i != j:
                            matrix[i][j] = self.group_confusion_weight
        return matrix.to(self.device)

    def calculate(self, predicted, target):
        indexes = torch.stack((target, predicted), 1)
        sum_ = torch.sum(self.weight_matrix[indexes[:, 0], indexes[:, 1]])

        return sum_
