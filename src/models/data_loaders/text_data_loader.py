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


class TextDataLoader:
    def __init__(self):
        self.vocab = None
        self.tokenizer = None
        self.device = None
        self.global_vectors = GloVe(name='6B', dim=300)
        self.tokenizer = get_tokenizer("basic_english")
        self.max_words = 50
        self.embed_len = 300

    @staticmethod
    def simple_f(l, lh):
        return torch.tensor([l, lh])

    def collate_batch(self, batch):

        Y, X = list(zip(*batch))
        Y = [1 if y == 5 else 0 for y in Y]
        X = [self.tokenizer(x) for x in X]
        X = [tokens + [""] * (self.max_words - len(tokens)) if len(tokens) < self.max_words else tokens[:self.max_words] for tokens in
             X]
        X_tensor = torch.zeros(len(batch), self.max_words, self.embed_len)
        for i, tokens in enumerate(X):
            X_tensor[i] = self.global_vectors.get_vecs_by_tokens(tokens)
        return X_tensor.reshape(len(batch), -1), torch.tensor(Y)

    def get_data_loaders(self, config):
        self.device = config["training"]["device"]
        train_df = pd.read_csv(Path('../../../data/processed/hierarchical_labels/'
                                    'news_category/News_Category_Dataset_v3_train.csv'))
        train_df = train_df[train_df['category'].isin(['POLITICS', 'WEIRD NEWS'])]
        ds = []
        for index, row in train_df.iterrows():
            ds.append((row[6], str(row[1]) + str(row[3])))


        train_loader = DataLoader(ds, batch_size=int(config["training"]["batch_size"]),
                                  collate_fn=self.collate_batch, shuffle=True)

        train_df = pd.read_csv(Path('../../../data/processed/hierarchical_labels/'
                                    'news_category/News_Category_Dataset_v3_test.csv'))
        train_df = train_df[train_df['category'].isin(['POLITICS', 'WEIRD NEWS'])]
        ds = []
        for index, row in train_df.iterrows():
            ds.append((row[6], row[1] + str(row[3])))

        test_loader = DataLoader(ds, batch_size=int(config["training"]["batch_size"]),
                                  collate_fn=self.collate_batch, shuffle=True)
        return train_loader, test_loader
