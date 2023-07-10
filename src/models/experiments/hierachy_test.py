import configparser
from pathlib import Path

import mlflow

from src.models.run_experiment_from_config import Experiment


def run_stability_test(
    server_uri, experiment_name, file, functions
):
    runs_params = []
    parameters = [
        "params.Batch size",
        "params.recalculate_period",
        "params.number of dimensions",
        "params.learning_rate",
    ]
    mlflow.set_tracking_uri(server_uri)
    exp = mlflow.search_runs(experiment_names=[experiment_name])
    tmp_exp = exp.sort_values(
        by=["metrics.metrics.MultiClassAccuracy"], ascending=False
    )
    tmp_exp = tmp_exp.head(n=1)
    print(tmp_exp['params.Batch size'])
    for index, row in tmp_exp.iterrows():
        params = {}
        for p in parameters:
            params[p] = row[p]
        runs_params.append(params)
    for f in functions:
        for i in runs_params:
            set_value_in_property_file(
                file,
                i["params.Batch size"],
                i["params.recalculate_period"],
                i["params.number of dimensions"],
                i["params.learning_rate"],
                8,
                f
            )
            experiment = Experiment(file)
            experiment.build_objects()
            experiment.build_metrics()
            experiment.train_and_log_model()
            del experiment


def set_value_in_property_file(
    file_path, batch_size, recalculate_period, out_dim, learning_rate, epochs, f
):
    config = configparser.ConfigParser()
    config.read(file_path)
    config.set("training", "batch_size", str(batch_size))
    config.set("training", "recalculate_period", str(recalculate_period))
    config.set("training", "lr", str(learning_rate))
    config.set("training", "out_dim", str(out_dim))
    config.set("training", "epochs", str(epochs))
    config.set("training", "loss_function", str(f))

    cfgfile = open(file_path, "w")
    config.write(cfgfile, space_around_delimiters=False)
    cfgfile.close()



if __name__ == "__main__":
    run_stability_test(
        "http://127.0.0.1:9999",
        "visual_coulomb_grid_search",
        Path("../configs/visual_config_coulomb_grid_search.ini"),
        ['loss_function.CoulombLossFunction']
    )
