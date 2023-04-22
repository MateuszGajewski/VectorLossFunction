import time

import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAccuracy


class SimpleVisualClassifier(nn.Module):
    def __str__(self):
        return "Simple_Visual_Cassifier"

    def __init__(self, out_dim=3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, out_dim)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                #m.bias.data.fill_(0.01)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def fit(self, config, optimizer, criterion, train_loader, vector_to_label_transformer = None):
        since = time.time()
        criterion.assign_data_loader(train_loader)
        criterion.assign_model(self)
        for epoch in range(
            int(config["training"]["epochs"])
        ):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs = inputs.to(config["training"]["device"])
                labels = labels.to(config["training"]["device"])
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.forward(inputs)
                outputs = outputs.to(config["training"]["device"])
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if i % 10 == 1:
                    print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.5f}")
                    running_loss = 0.0
        if vector_to_label_transformer:
            vector_to_label_transformer.fit(criterion)
            print(criterion.get_centroids())
        print("Finished Training")
        time_elapsed = time.time() - since
        mlflow.log_metric("Training time", time_elapsed)

    def validate(self, config, test_loader, metrics, vector_to_label_transformer=None):
        total_loss = 0.0
        total_number = 0
        losses = {}
        for i in metrics.keys():
            losses[i] = 0
        with torch.no_grad():
            for i, data in enumerate(test_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs = inputs.to(config["training"]["device"])
                labels = labels.to(config["training"]["device"])
                # forward + backward + optimize
                outputs = self.forward(inputs)
                outputs = outputs.to(config["training"]["device"])
                if vector_to_label_transformer:
                    outputs = vector_to_label_transformer.predict(outputs)
                for i in metrics.keys():
                    losses[i] += metrics[i].calculate(outputs, labels)
                total_number += labels.size(0)

        for i in metrics.keys():
            print(f"Finished validaion")
            print(f"{i}: {losses[i]/total_number}")
            mlflow.log_metric(i, losses[i]/total_number)
