from pathlib import Path

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import pandas as pd
import src.data.datasets as dataset
import src.models.classifiers as classifiers
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from torchtext.vocab import GloVe
import torchtext
from torchtext.data import get_tokenizer
from torchtext.data.functional import to_map_style_dataset


class AGNewsDataLoader:
    def __init__(self):
        self.vocab = None
        self.tokenizer = None
        self.device = None
        self.global_vectors = GloVe(name='840B', dim=300)
        self.tokenizer = get_tokenizer("basic_english")
        self.max_words = 50
        self.embed_len = 300

    @staticmethod
    def simple_f(l, lh):
        return torch.tensor([l, lh])

    def collate_batch(self, batch):

        Y, X = list(zip(*batch))
        X = [self.tokenizer(x) for x in X]
        X = [tokens + [""] * (self.max_words - len(tokens)) if len(tokens) < self.max_words else tokens[:self.max_words] for tokens in
             X]
        X_tensor = torch.zeros(len(batch), self.max_words, self.embed_len)
        for i, tokens in enumerate(X):
            X_tensor[i] = self.global_vectors.get_vecs_by_tokens(tokens)
        return X_tensor.reshape(len(batch), -1), torch.tensor(Y) - 1

    def get_data_loaders(self, config):
        self.device = config["training"]["device"]
        train_dataset, test_dataset  = torchtext.datasets.AG_NEWS()
        #train_dataset, test_dataset = to_map_style_dataset(train_dataset), to_map_style_dataset(test_dataset)
        train_dataset, test_dataset = to_map_style_dataset(train_dataset), to_map_style_dataset(test_dataset)

        train_loader = DataLoader(train_dataset, batch_size=int(config["training"]["batch_size"]), collate_fn=self.collate_batch, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=int(config["training"]["batch_size"]), collate_fn=self.collate_batch)
        #print(len(self.vectorizer.get_feature_names_out()))
        if (
                config.has_option("training", "softmax_layer")
                and config["training"]["softmax_layer"] == "True"
        ):
            cls = eval(config["training"]["classifier"])(
                self.max_words*self.embed_len,
                int(config["training"]["out_dim"]), True
            ).to(config["training"]["device"])
        else:
            cls = eval(config["training"]["classifier"])(
                self.max_words*self.embed_len,
                int(config["training"]["out_dim"]),
            ).to(config["training"]["device"])

        return train_loader, test_loader, cls
