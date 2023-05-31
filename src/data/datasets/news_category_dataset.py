import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset, random_split


class NewsCategoryDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data = pd.read_csv(data_path)[0:20000]
        print("len", len(self.data))
        #self.data = pd.read_csv(data_path)
        selected_cat = ['POLITICS', 'WELNESS', 'ENTERTAINMENT', 'TRAVEL']
        #self.data = self.data[self.data['category'].isin(selected_cat)].reset_index(drop=True)

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Get the image and label at the specified index
        text = str(self.data.iloc[index, 8])  # str(self.data.iloc[index, 1]) +
        label = self.data.iloc[index, 7]

        return text, label
