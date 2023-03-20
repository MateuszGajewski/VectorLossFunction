from src.data.add_hierarchical_labels.hierarchical_label_transformer import HierarchicalLabelTransformer
from pathlib import Path
import pandas as pd
import json
import numpy as np

class HierarchicalLabelTransformerWineQuality(HierarchicalLabelTransformer):

    def __init__(self):
        self.new_categories_dict = {'0': ['3', '4'],
                                    '1': ['5', '6'],
                                    '2': ['7', '8', '9'],
                                    }

    def new_label(self, row) -> str:
        for i in self.new_categories_dict:
            if str(row['quality']) in self.new_categories_dict[i]:
                return i

    def add_label(self, src: Path, dst: Path, dst_test: Path = None) -> None:
        df = pd.read_csv(src)
        df['hierarchical_label'] = df.apply(lambda row: self.new_label(row), axis=1)
        msk = np.random.rand(len(df)) < 0.8
        train_df = df[msk]
        test_df = df[~msk]
        train_df.to_csv(dst)
        test_df.to_csv(dst_test)
        with open(str(dst.parent.resolve()) + '/hierarchical_labels.json', 'w') as fp:
            json.dump(self.new_categories_dict, fp)
