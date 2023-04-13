from pathlib import Path

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import src.data.datasets as dataset
import src.models.classifiers as classifiers


class VisualDataLoader:
    @staticmethod
    def simple_f(l, lh):
        return torch.tensor([l, lh])

    def get_data_loaders(self, config):
        transform = transforms.Compose(  # composing several transforms together
            [
                transforms.ToTensor(),  # to tensor object
                transforms.Normalize((0.5), (0.5)),
            ]
        )
        dataset_class = eval(config["data"]["dataset"])
        dataset_train = dataset_class(
            Path(config["data"]["train_data"]),
            transform=transform,
            #label_to_vec_function=self.simple_f,
        )
        dataset_test = dataset_class(
            Path(config["data"]["test_data"]),
            transform=transform,
            #label_to_vec_function=self.simple_f,
        )

        train_loader = DataLoader(
            dataset_train,
            batch_size=int(config["training"]["batch_size"]),
            shuffle=True,
            pin_memory=True,
        )
        test_loader = DataLoader(
            dataset_test,
            batch_size=int(config["training"]["batch_size"]),
            shuffle=True,
            pin_memory=True,
        )
        cls = eval(config["training"]["classifier"])(
            int(config["training"]["out_dim"])
        ).to(config["training"]["device"])

        return train_loader, test_loader, cls
