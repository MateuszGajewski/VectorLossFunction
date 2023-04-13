import time

import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def fit(self, config, optimizer, criterion, train_loader):
        since = time.time()
        for epoch in range(
            int(config["training"]["epochs"])
        ):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.forward(inputs)
                loss = criterion(outputs, labels, epoch)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if i % 10 == 1:
                    print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.5f}")
                    running_loss = 0.0

        print("Finished Training")
        time_elapsed = time.time() - since
        mlflow.log_metric("Training time", time_elapsed)

    def validate(self, config, optimizer, criterion, test_loader):
        total_loss = 0.0
        total_number = 0
        with torch.no_grad():
            for i, data in enumerate(test_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs = inputs.to(config["training"]["device"])
                labels = labels.to(config["training"]["device"])
                # forward + backward + optimize
                outputs = self.forward(inputs)
                outputs = outputs.to(config["training"]["device"])
                loss = criterion(outputs, labels, 0)
                total_loss += loss.item()
                print(loss, outputs, labels)
                total_number += labels.size(0)

        print(f"Finished validaion, avg loss: {total_loss / total_number}")
        mlflow.log_metric("Avg test loss", total_loss / total_number)
