import time

import mlflow
import torch
from torch import nn


class SimpleTextClassifier(nn.Module):
    def __str__(self):
        return "Simple_Text_Cassifier"

    def __init__(self, vocab_size, embed_dim, num_class):
        super(SimpleTextClassifier, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

    def fit(self, config, optimizer, criterion, train_loader, vector_to_label_transformer = None):
        since = time.time()
        self.train()
        for epoch in range(int(config["training"]["epochs"])):
            running_loss = 0.0
            for i, (label, text, offsets) in enumerate(train_loader, 0):
                optimizer.zero_grad()
                text = text.to(config["training"]["device"])
                label = label.to(config["training"]["device"])
                offsets = offsets.to(config["training"]["device"])
                outputs = self.forward(text, offsets)
                outputs = outputs.to(config["training"]["device"])
                loss = criterion(outputs, label)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 0.1)
                optimizer.step()
                running_loss += loss.item()
                if i % 10 == 1:
                    print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
                    running_loss = 0.0

            print("Finished Training")
            time_elapsed = time.time() - since
            mlflow.log_metric("Training time", time_elapsed)

    def validate(self, config, test_loader, metrics, vector_to_label_transformer=None):
        total_loss = 0.0
        total_number = 0
        losses = {}
        for i in metrics.keys():
            losses[i] = 0
        print(losses)
        with torch.no_grad():
            for i, data in enumerate(test_loader, 0):
                label, text, offsets = data
                # get the inputs; data is a list of [inputs, labels]
                text = text.to(config["training"]["device"])
                labels = label.to(config["training"]["device"])
                offsets = offsets.to(config["training"]["device"])
                outputs = self.forward(text, offsets)
                outputs = outputs.to(config["training"]["device"])
                if vector_to_label_transformer:
                    outputs = vector_to_label_transformer.predict(outputs)
                elif outputs.shape[1] == 10:
                    _, outputs = torch.max(outputs.data, 1)
                for j in metrics.keys():
                    losses[j] += metrics[j].calculate(outputs, labels)
                total_number += labels.size(0)

            for i in metrics.keys():
                print(f"Finished validaion")
                print(f"{i}: {losses[i] / total_number}")
                mlflow.log_metric(i, losses[i] / total_number)
