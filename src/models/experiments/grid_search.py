import configparser
from pathlib import Path

from src.models.run_experiment_from_config import Experiment


def run_grid_search(file: Path):
    batch_size = [32, 128, 512, 2048]
    batch_size.reverse()
    recalculate_period = [0]
    recalculate_period.reverse()
    out_dim = [10]
    out_dim.reverse()
    learning_rate = [0.001, 0.0001]
    for batch_size_ in batch_size:
        for recalculate_period_ in recalculate_period:
            for out_dim_ in out_dim:
                for learning_rate_ in learning_rate:
                    set_value_in_property_file(
                        file, batch_size_, recalculate_period_, out_dim_, learning_rate_
                    )

                    experiment = Experiment(file)
                    experiment.build_objects()
                    experiment.build_metrics()
                    experiment.train_and_log_model()
                    del experiment


def set_value_in_property_file(
    file_path, batch_size, recalculate_period, out_dim, learning_rate
):
    config = configparser.ConfigParser()
    config.read(file_path)
    config.set("training", "batch_size", str(batch_size))
    config.set("training", "recalculate_period", str(recalculate_period))
    config.set("training", "lr", str(learning_rate))
    config.set("training", "out_dim", str(out_dim))
    cfgfile = open(file_path, "w")
    config.write(cfgfile, space_around_delimiters=False)
    cfgfile.close()


if __name__ == "__main__":
    run_grid_search(
        Path("../configs/visual_config_cross_entropy_grid_search.ini")
    )
