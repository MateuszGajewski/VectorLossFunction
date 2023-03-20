from src.data.add_hierarchical_labels.hierarchical_label_transformer import HierarchicalLabelTransformer
from pathlib import Path
import pandas as pd
import json


class HierarchicalLabelTransformerAmazonReviews(HierarchicalLabelTransformer):

    def add_label(self, src: Path, dst: Path) -> None:
        #all labels are included
        df = pd.read_csv(src)
        df.to_csv(dst)






