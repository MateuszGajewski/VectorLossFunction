from src.data.add_hierarchical_labels.hierarchical_label_transformer_news import HierarchicalLabelTransformerNews
from src.data.add_hierarchical_labels.hierarchical_label_transformer_fashion_mnist import HierarchicalLabelTransformerFashionMnist
from src.data.add_hierarchical_labels.hierarchical_label_transformer_amazon_rewievs import HierarchicalLabelTransformerAmazonReviews

from pathlib import Path

if __name__ == "__main__":
    h = HierarchicalLabelTransformerNews()
    h.add_label(Path('../../../data/raw/news_category/News_Category_Dataset_v3.json'),
                Path('../../../data/processed/hierarchical_labels/news_category/News_Category_Dataset_v3_train.csv'),
                Path('../../../data/processed/hierarchical_labels/news_category/News_Category_Dataset_v3_test.csv'))

    h = HierarchicalLabelTransformerFashionMnist()
    h.add_label(Path('../../../data/raw/fashion_mnist/fashion-mnist_test.csv'),
                Path('../../../data/processed/hierarchical_labels/fashion_mnist/fashion-mnist_test.csv'))

    h = HierarchicalLabelTransformerAmazonReviews()
    h.add_label(Path('../../../data/raw/amazon_reviews/train_40k.csv'),
                     Path('../../../data/processed/hierarchical_labels/amazon_rewievs/train_40k.csv'))

    h = HierarchicalLabelTransformerAmazonReviews()
    h.add_label(Path('../../../data/raw/amazon_reviews/val_10k.csv'),
                     Path('../../../data/processed/hierarchical_labels/amazon_rewievs/val_10k.csv'))
