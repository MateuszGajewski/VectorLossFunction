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

class TextDataLoader:
    def __init__(self):
        self.vocab = None
        self.tokenizer = None
        self.device = None

    @staticmethod
    def simple_f(l, lh):
        return torch.tensor([l, lh])

    def collate_batch(self, batch):
        label_list, text_list = [], []
        for _text, _label in batch:
            label_list.append(_label)
            tfidf = self.vectorizer.transform([_text])
            processed_text = torch.tensor(tfidf.toarray(), dtype=torch.float32)
            text_list.append(processed_text)
        label_list = torch.tensor(label_list)
        text_list = torch.cat(text_list)
        return (
            text_list.to(self.device),
            label_list.to(self.device),

        )

    def get_data_loaders(self, config):
        self.device = config["training"]["device"]
        dataset_class = eval(config["data"]["dataset"])
        dataset_train = dataset_class(Path(config["data"]["train_data"]))
        dataset_test = dataset_class(Path(config["data"]["test_data"]))
        self.vectorizer = TfidfVectorizer()
        trainXV = self.vectorizer.fit_transform(dataset_train.data['clean_text'].values.astype('U'))

        train_loader = DataLoader(
            dataset_train,
            batch_size=int(config["training"]["batch_size"]),
            shuffle=True,
            collate_fn=self.collate_batch,
        )
        test_loader = DataLoader(
            dataset_test,
            batch_size=int(config["training"]["batch_size"]),
            shuffle=True,
            collate_fn=self.collate_batch,
        )
        print(len(self.vectorizer.get_feature_names_out()))
        if (
                config.has_option("training", "softmax_layer")
                and config["training"]["softmax_layer"] == "True"
        ):
            cls = eval(config["training"]["classifier"])(
                len(self.vectorizer.get_feature_names_out()),
                int(config["training"]["out_dim"]), True
            ).to(config["training"]["device"])
        else:
            cls = eval(config["training"]["classifier"])(
                len(self.vectorizer.get_feature_names_out()),
                int(config["training"]["out_dim"]),
            ).to(config["training"]["device"])

        return train_loader, test_loader, cls
