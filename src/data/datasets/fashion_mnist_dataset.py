import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split


class FashionMNISTDataset(Dataset):
    def __init__(self, data_path, transform=None, label_to_vec_function=None):
        self.data = pd.read_csv(data_path)
        self.transform = transform
        self.label_to_vec_function = label_to_vec_function

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Get the image and label at the specified index
        image = self.data.iloc[index, 2:].values.astype(np.uint8).reshape(28, 28, 1)
        label = self.data.iloc[index, 0]

        if self.label_to_vec_function:
            label = self.label_to_vec_function(self.data.iloc[index, 0], self.data.iloc[index, 1])

        # Apply the transformation (if any) to the image
        if self.transform:
            image = self.transform(image)

        return image, label