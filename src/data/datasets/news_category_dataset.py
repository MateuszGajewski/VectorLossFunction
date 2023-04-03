import pandas as pd
from torch.utils.data import Dataset


class NewsCategoryDataset(Dataset):
    def __init__(self, data_path, transform=None, label_to_vec_function=None):
        self.data = pd.read_csv(data_path)
        self.transform = transform
        self.label_to_vec_function = label_to_vec_function

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Get the image and label at the specified index
        text = str(self.data.iloc[index, 1]) + str(self.data.iloc[index, 3])
        label = self.data.iloc[index, 7]

        if self.label_to_vec_function:
            label = self.label_to_vec_function(
                self.data.iloc[index, 7], self.data.iloc[index, 6]
            )

        return text, label
