import configparser
from pathlib import Path

import torch

from src.models.data_loaders.text_data_loader import TextDataLoader
from src.models.run_experiment_from_config import Experiment

def run(file):
    config = configparser.ConfigParser()
    config.read(file)
    t = TextDataLoader()
    train_data_loader, test_data_loader = t.get_data_loaders(config)


    experiment = Experiment(file)
    experiment.build_objects()
    experiment.build_metrics()
    experiment.train_and_log_model()
    experiment.test_loader = test_data_loader
    experiment.criterion.data_loader = train_data_loader
    experiment.criterion.class_number = 2
    experiment.criterion.init_tensors(torch.zeros((1, int(config['training']['out_dim']))), 0)
    experiment.criterion.recalculate_centroids()
    experiment.vector_to_label_transformer.fit(experiment.criterion)
    experiment.validate()
    print(experiment.criterion.centroids)

if __name__ == '__main__':

    run(Path("../configs/text_config_coulomb_grid_search.ini"))