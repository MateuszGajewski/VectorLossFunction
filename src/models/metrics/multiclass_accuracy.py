from torchmetrics.classification import MulticlassAccuracy

from .abstract_metric import AbstractMetric


class MultiClassAccuracy(AbstractMetric):
    def __init__(self, json, device="cpu"):
        super(MultiClassAccuracy, self).__init__(json, device)
        self.metric = MulticlassAccuracy(
            num_classes=self.class_number, average="micro"
        ).to(self.device)

    def calculate(self, predicted, target):
        return self.metric(predicted, target) * target.shape[0]
