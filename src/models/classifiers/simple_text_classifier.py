from torch import nn
import time
import torch

class SimpleTextClassifier(nn.Module):

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

    def fit(self, config, optimizer, criterion, train_loader):
        self.train()
        total_acc, total_count = 0, 0
        log_interval = 500
        start_time = time.time()
        for epoch in range(int(config['training']['epochs'])):
            for idx, (label, text, offsets) in enumerate(train_loader):
                optimizer.zero_grad()
                predicted_label = self.forward(text, offsets)
                loss = criterion(predicted_label, label)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 0.1)
                optimizer.step()
                total_count += label.size(0)
                if idx % log_interval == 0 and idx > 0:
                    elapsed = time.time() - start_time
                    print('| epoch {:3d} | {:5d}/{:5d} batches '
                          '| accuracy {:8.3f}'.format(epoch, idx, len(train_loader),
                                                      total_acc / total_count))
                    total_acc, total_count = 0, 0
                    start_time = time.time()
