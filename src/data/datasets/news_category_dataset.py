import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset, random_split


class NewsCategoryDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data = pd.read_csv(data_path)[0:10000]
        selected_cat = ['POLITICS',  'ENTERTAINMENT', 'U.S. NEWS', 'WORLD NEWS']
        self.data = self.data[self.data['category'].isin(selected_cat)].reset_index(drop=True)
        print("len", len(self.data))
        print(self.data['category'].value_counts())
        print(self.data['label'].value_counts())
        #self.data = pd.read_csv(data_path)[0:15000]
        #self.data = pd.read_csv(data_path)


        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Get the image and label at the specified index
        text = str(self.data.iloc[index, 8])  # str(self.data.iloc[index, 1]) +
        label = self.data.iloc[index, 7]

        return text, label
