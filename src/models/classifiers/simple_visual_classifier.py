import time

import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAccuracy


class SimpleVisualClassifier(nn.Module):
    def __str__(self):
        return "Simple_Visual_Cassifier"

    def init_max_metrics(self, metrics):
        self.max_metrics = {}
        for i in metrics.keys():
            self.max_metrics[i] = 0

    def __init__(self, out_dim=3, softmax_layer=False, tanh_layer=False):
        self.max_metrics = None
        super().__init__()
        self.softmax_layer = softmax_layer
        self.tanh_layer =  tanh_layer
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, out_dim)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # m.bias.data.fill_(0.01)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        if self.tanh_layer:
            x = F.tanh(x)
            #x = F.sigmoid(x)
            #print(x)
        #print(x)
        if self.softmax_layer:
            x = F.log_softmax(x, dim=1)
        return x

    def fit(
        self,
        config,
        optimizer,
        criterion,
        train_loader,
        vector_to_label_transformer=None,
        test_loader=None,
        metrics=None,
    ):
        if metrics is not None:
            self.init_max_metrics(metrics)
        since = time.time()
        for epoch in range(
            int(config["training"]["epochs"])
        ):  # loop over the dataset multiple times
            running_loss = 0.0
            epoch_loss = 0.0
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
                loss = criterion(outputs, labels, epoch)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                epoch_loss += loss.item()
                if i % 10 == 1:
                    print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.5f}")
                    running_loss = 0.0
            mlflow.log_metric("loss", epoch_loss, epoch)
            if test_loader is not None:
                if vector_to_label_transformer:
                    vector_to_label_transformer.fit(criterion)
                self.validate(
                    config, test_loader, metrics, vector_to_label_transformer, epoch
                )

        if vector_to_label_transformer:
            vector_to_label_transformer.fit(criterion)
            #print(criterion.get_centroids())
        print("Finished Training")
        time_elapsed = time.time() - since
        mlflow.log_metric("Training time", time_elapsed)
        if test_loader is not None:
            for i in metrics.keys():
                mlflow.log_metric("max_value_" + i, self.max_metrics[i])

    def validate(
        self, config, test_loader, metrics, vector_to_label_transformer=None, step=None
    ):
        total_loss = 0.0
        total_number = 0
        losses = {}
        for i in metrics.keys():
            losses[i] = 0
        print(losses)
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
                elif outputs.shape[1] == 10:
                    _, outputs = torch.max(outputs.data, 1)
                for j in metrics.keys():
                    losses[j] += metrics[j].calculate(outputs, labels).to(config["training"]["device"])
                total_number += labels.size(0)

        for i in metrics.keys():
            print(f"Finished validaion")
            print(f"{i}: {losses[i] / total_number}")
            m = losses[i] / total_number
            if m > self.max_metrics[i]:
                self.max_metrics[i] = m
            if step is not None:
                mlflow.log_metric(i, m, step)
            else:
                mlflow.log_metric(i, m)
