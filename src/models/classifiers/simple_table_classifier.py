from torch import nn
import time
import torch
import mlflow


class SimpleTableClassifier(nn.Module):

    def __init__(self, features_size, out_dim):
        super(SimpleTableClassifier, self).__init__()
        # Setting a standard number of hidden units for each layer
        # Hidden layer
        self.fc1 = nn.Linear(features_size, 50)
        self.relu = nn.ReLU()
        # Output layer
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, out_dim)

    def forward(self, x):
        # l1
        x = self.fc1(x)
        x = self.relu(x)

        # l2
        x = self.fc2(x)
        x = self.relu(x)

        # l3
        x = self.fc3(x)

        # # output
        # x = torch.log_softmax(x, dim=1)
        return x

    def fit(self, config, optimizer, criterion, train_loader):
        since = time.time()
        self.train()
        for epoch in range(int(config['training']['epochs'])):
            running_loss = 0.0
            for i, (features, label) in enumerate(train_loader):
                optimizer.zero_grad()
                predicted_label = self.forward(features)
                loss = criterion(predicted_label, label)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if i % 10 == 1:
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0

            print('Finished Training')
            time_elapsed = time.time() - since
            mlflow.log_metric('Training time', time_elapsed)

    def validate(self, config, optimizer, criterion, test_loader):
        total_loss = 0.0
        total_number = 0
        with torch.no_grad():
            for i, (features, labels) in enumerate(test_loader):
                # get the inputs; data is a list of [inputs, labels]
                features = features.to(config['training']['device'])
                labels = labels.to(config['training']['device'])
                # forward + backward + optimize
                outputs = self.forward(features)
                outputs = outputs.to(config['training']['device'])
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                total_number += labels.size(0)

        print(f'Finished validaion, avg loss: {total_loss / total_number}')
        mlflow.log_metric('Avg test loss', total_loss/total_number)


