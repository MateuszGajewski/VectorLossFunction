import configparser
from pathlib import Path
from src.models.data_loaders.text_data_loader import TextDataLoader
from src.models.run_experiment_from_config import Experiment

def run(file):
    config = configparser.ConfigParser()
    config.read(file)
    t = TextDataLoader()
    train_data_loader, test_ata_loader = t.get_data_loaders(config)

    experiment = Experiment(file)
    experiment.build_objects()
    experiment.build_metrics()
    experiment.train_and_log_model()
    experiment.test_loader = test_ata_loader
    experiment.cls.criterion.data_loader = train_data_loader


if __name__ == '__main__':

    run(Path("../configs/text_config_coulomb_grid_search.ini"))