import json
from pathlib import Path

import pandas as pd

from src.data.add_hierarchical_labels.hierarchical_label_transformer import \
    HierarchicalLabelTransformer


class HierarchicalLabelTransformerFashionMnist(HierarchicalLabelTransformer):
    def __init__(self):
        self.new_categories_dict = {
            "0": ["9", "7", "5"],  # category for "Ankle boot", "Sneaker" and "Sandal"
            "1": [
                "0",
                "2",
                "4",
                "6",
            ],  # category for "T-shirt/top", "Pullover", "Coat", "Shirt"
            "2": ["1", "3"],  # category for "Trouser" and "Dress"
            "3": ["8"],
        }  # category for "Trouser" and "Bag"

    def new_label(self, row) -> str:
        for i in self.new_categories_dict:
            if str(row["label"]) in self.new_categories_dict[i]:
                return i

    def add_label(self, src: Path, dst: Path) -> None:
        df = pd.read_csv(src)
        df["hierarchical_label"] = df.apply(lambda row: self.new_label(row), axis=1)
        df.to_csv(dst)
        with open(str(dst.parent.resolve()) + "/hierarchical_labels.json", "w") as fp:
            json.dump(self.new_categories_dict, fp)

        f = open(str(src.parent.resolve()) + "/labels.json")
        labels = json.load(f)
        with open(str(dst.parent.resolve()) + "/labels.json", "w") as fp:
            json.dump(labels, fp)
