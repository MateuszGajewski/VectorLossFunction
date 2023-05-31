from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer


class TextDataLoader:
    def __init__(self):
        self.vocab = None
        self.tokenizer = None
        self.device = None

    @staticmethod
    def simple_f(l, lh):
        return torch.tensor([l, lh])

    def collate_batch(self, batch):
        text_pipeline = lambda x: self.vocab(self.tokenizer(x))
        label_list, text_list, offsets = [], [], [0]
        for _text, _label in batch:
            label_list.append(_label)
            processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))
        label_list = torch.tensor(label_list)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)
        return (
            label_list.to(self.device),
            text_list.to(self.device),
            offsets.to(self.device),
        )

    def get_data_loaders(self, config):
        self.device = config["training"]["device"]
        dataset_class = eval(config["data"]["dataset"])
        dataset_train = dataset_class(Path(config["data"]["train_data"]))
        dataset_test = dataset_class(Path(config["data"]["test_data"]))


        vectorizer = TfidfVectorizer()
        # fit_transform for train data
        trainXV = vectorizer.fit_transform(train['clean_text'])
        trainY = train['category']
        # transform for test data
        testXV = vectorizer.transform(test['clean_text'])
        testY = test['category']
        print(trainXV.shape, testXV.shape)

        train_loader = DataLoader(
            dataset_train,
            batch_size=int(config["training"]["batch_size"]),
            shuffle=True,
            pin_memory=True,
            collate_fn=self.collate_batch,
        )
        test_loader = DataLoader(
            dataset_test,
            batch_size=int(config["training"]["batch_size"]),
            shuffle=True,
            pin_memory=True,
            collate_fn=self.collate_batch,
        )
        cls = eval(config["training"]["classifier"])(
            len(self.vocab),
            int(config["data"]["embed_dim"]),
            int(config["training"]["out_dim"]),
        ).to(config["training"]["device"])

        return train_loader, test_loader, cls
