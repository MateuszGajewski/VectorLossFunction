from abc import abstractmethod


class AbstractMetric:
    def __init__(self, json, device='cpu'):
        self.class_number = self.get_class_number(json)
        self.device = device

    @staticmethod
    def get_class_number(json):
        class_number = 0
        for i in json.keys():
            class_number += len(json[i])
        return class_number

    @abstractmethod
    def calculate(self, predicted, target):
        raise NotImplementedError
