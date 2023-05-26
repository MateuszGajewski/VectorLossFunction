import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset, random_split


class FashionMNISTDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data = pd.read_csv(data_path)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Get the image and label at the specified index
        image = self.data.iloc[index, 2:].values.astype(np.uint8).reshape(28, 28, 1)
        label = self.data.iloc[index, 1]

        # Apply the transformation (if any) to the image
        if self.transform:
            image = self.transform(image)

        return image, label
