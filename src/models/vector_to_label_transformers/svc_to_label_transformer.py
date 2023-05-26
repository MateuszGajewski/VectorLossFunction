import torch
import torch.nn as nn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np


class SVCToLabelTransformer:
    def __init__(self):
        self.centroids = None
        self.clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))

    def fit(self, criterion):
        X = []
        y = []
        dataset_size = len(criterion.data_loader.dataset)
        sample_size = int(np.ceil(dataset_size * 0.3))
        indx = np.random.randint(len(criterion.data_loader), size=sample_size)
        subset = torch.utils.data.Subset(criterion.data_loader.dataset, indx)
        testloader_subset = torch.utils.data.DataLoader(subset, batch_size=sample_size - 1, num_workers=0,
                                                        shuffle=False)
        with torch.no_grad():
            for i, data in enumerate(testloader_subset, 0):
                inputs, labels = data
                inputs = inputs.to(criterion.device)
                outputs = criterion.model.forward(inputs)
                outputs.to(criterion.device)
                y += labels.tolist()
                X += outputs.tolist()
        self.clf.fit(X, y)


    def predict(self, input_batch):
        pred = self.clf.predict(input_batch.tolist())
        return torch.tensor(pred)
