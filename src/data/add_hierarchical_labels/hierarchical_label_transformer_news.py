from src.data.add_hierarchical_labels.hierarchical_label_transformer import HierarchicalLabelTransformer
from pathlib import Path
import pandas as pd
import json
import numpy as np

class HierarchicalLabelTransformerNews(HierarchicalLabelTransformer):
    def __init__(self):
        self.new_categories_dict = {'0': ['POLITICS', 'BUSINESS', 'WORLD NEWS', 'THE WORLDPOST', 'WORLDPOST', 'MONEY', 'U.S. NEWS'],
                                    '1': ['EDUCATION', 'TECH', 'SCIENCE', 'COLLEGE'],
                                    '2': ['CULTURE & ARTS', 'ARTS & CULTURE', 'ARTS', 'STYLE'],
                                    '3': ['WELLNESS', 'STYLE & BEAUTY', 'PARENTING', 'HEALTHY LIVING',
                                          'PARENTS', 'TRAVEL', 'HOME & LIVING', 'DIVORCE', ],
                                    '4': ['LATINO VOICES', 'QUEER VOICES', 'BLACK VOICES'],
                                    '5': ['ENTERTAINMENT', 'COMEDY', 'WEIRD NEWS'],
                                    '6': ['FOOD & DRINK', 'TASTE']}

    def drop_rows_not_in_dict(self, df):
        all_categories = []
        for i in self.new_categories_dict:
            all_categories += self.new_categories_dict[i]
        df = df[df['category'].isin(all_categories)]
        return df

    def new_label(self, row) -> str:
        for i in self.new_categories_dict:
            if str(row['category']) in self.new_categories_dict[i]:
                return i

    def add_label(self, src: Path, dst: Path, dst_test: Path = None) -> None:
        df = pd.read_json(src, lines=True)
        df = self.drop_rows_not_in_dict(df)
        df['hierarchical_label'] = df.apply(lambda row: self.new_label(row), axis=1)
        msk = np.random.rand(len(df)) < 0.8
        train_df = df[msk]
        test_df = df[~msk]
        train_df.to_csv(dst)
        test_df.to_csv(dst_test)
        with open(str(dst.parent.resolve()) + '/hierarchical_labels.json', 'w') as fp:
            json.dump(self.new_categories_dict, fp)

h = HierarchicalLabelTransformerNews()
h.add_label(Path('../../../data/raw/news_category/News_Category_Dataset_v3.json'),
            Path('../../../data/processed/hierarchical_labels/news_category/News_Category_Dataset_v3_train.csv'),
            Path('../../../data/processed/hierarchical_labels/news_category/News_Category_Dataset_v3_test.csv'))