import time

import mlflow
import torch
from torch import nn
import torch.nn.functional as F


class SimpleTextClassifier(nn.Module):
    def __str__(self):
        return "Simple_Text_Cassifier"

    def __init__(self, vocab_size, num_class, softmax_layer=False, tanh_layer=False):
        super(SimpleTextClassifier, self).__init__()
        self.softmax_layer = softmax_layer
        self.tanh_layer =  tanh_layer
        self.model = nn.Sequential(
            nn.Linear(vocab_size, 128),
            nn.Linear(128, 64),
            nn.Linear(64, 128),
            nn.Linear(128, 64),
            nn.Linear(64, num_class),
        )
    def init_weights(self):
        initrange = 0.5
        #self.embedding.weight.data.uniform_(-initrange, initrange)
        #self.model.weight.data.uniform_(-initrange, initrange)
        #self.model.bias.data.zero_()
        # nn.init.xavier_uniform_(self.fc.weight)
    def init_max_metrics(self, metrics):
        self.max_metrics = {}
        for i in metrics.keys():
            self.max_metrics[i] = 0

    def forward(self, x):
        x = self.model(x)
        if self.softmax_layer:
            x = F.log_softmax(x, dim=1)
        if self.tanh_layer:
            x = F.tanh(x)
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
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 0.1)
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
            print(criterion.get_centroids())
        print("Finished Training")
        time_elapsed = time.time() - since
        mlflow.log_metric("Training time", time_elapsed)
        if test_loader is not None:
            for i in metrics.keys():
                mlflow.log_metric("max_value_" + i, self.max_metrics[i])

    def validate(self, config, test_loader, metrics, vector_to_label_transformer=None):
        total_loss = 0.0
        total_number = 0
        losses = {}
        for i in metrics.keys():
            losses[i] = 0
        print(losses)
        with torch.no_grad():
            for i, data in enumerate(test_loader, 0):
                inputs, label = data
                # get the inputs; data is a list of [inputs, labels]
                inputs = inputs.to(config["training"]["device"])
                labels = label.to(config["training"]["device"])
                outputs = self.forward(inputs)
                outputs = outputs.to(config["training"]["device"])
                if vector_to_label_transformer:
                    outputs = vector_to_label_transformer.predict(outputs)
                elif outputs.shape[1] == 31:
                    _, outputs = torch.max(outputs.data, 1)
                for j in metrics.keys():
                    losses[j] += metrics[j].calculate(outputs, labels)
                total_number += labels.size(0)

            for i in metrics.keys():
                print(f"Finished validaion")
                print(f"{i}: {losses[i] / total_number}")
                mlflow.log_metric(i, losses[i] / total_number)
