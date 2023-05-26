import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split


class WineQualityDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data = pd.read_csv(data_path)
        self.data["type"] = self.data["type"].replace({"white": 0, "red": 1})
        self.data = self.data.fillna(-1)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        attributes = self.data.iloc[index, 0:12].values.astype(np.float)
        label = self.data.iloc[index, 12]

        return torch.tensor(attributes, dtype=torch.float), label
