import configparser
from pathlib import Path

from src.models.run_experiment_from_config import Experiment

def run(file):
    experiment = Experiment(file)
    experiment.build_objects()
    experiment.build_metrics()
    experiment.train_and_log_model()


if __name__ == '__main__':
    run(Path("../configs/text_config_coulomb_grid_search.ini"))